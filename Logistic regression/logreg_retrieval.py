"""
Logistic Regression Retrieval for DGEB Benchmark
Corrected version: trains a separate model per dataset (Arch and Euk),
uses saga solver (matching actual runs), and evaluates each task
against its own corpus independently.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datasets import load_dataset
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


# ---------------------------------------------------------------------------
# STEP 1: K-MER FEATURE EXTRACTION
# ---------------------------------------------------------------------------
# For a given protein sequence string and integer k, slide a window of
# length k across the sequence and count every contiguous subsequence
# (k-mer). Returns a Counter dict: {"ACDE": 3, "CDEF": 1, ...}
# This is the raw term-frequency input before TF-IDF weighting.

def kmer_counts(sequence, k):
    return Counter(sequence[i:i+k] for i in range(len(sequence) - k + 1))


def build_feature_matrix(sequences, k, vectorizer=None, tfidf=None):
    """
    Converts a list of protein sequences into a TF-IDF-weighted sparse matrix.

    If vectorizer is None, it is fit on the provided sequences (training mode).
    If vectorizer is provided, it is applied without refitting (inference mode).
    Same logic applies to the tfidf transformer.

    This distinction is critical: when encoding query or corpus sequences at
    evaluation time, we must use the vocabulary and IDF weights learned from
    the TRAINING corpus, not refit on new data.
    """
    # Count k-mers for every sequence
    kmer_dicts = [
        kmer_counts(seq, k=k)
        for seq in tqdm(sequences, desc=f"  {k}-mer extraction", unit="seq")
    ]

    if vectorizer is None:
        # Training mode: learn the vocabulary (all observed k-mers) and
        # convert count dicts into a sparse count matrix
        vectorizer = DictVectorizer(sparse=True)
        X = vectorizer.fit_transform(kmer_dicts)
    else:
        # Inference mode: apply the existing vocabulary. K-mers not seen
        # during training are silently ignored (set to 0).
        X = vectorizer.transform(kmer_dicts)

    if tfidf is None:
        # Training mode: compute IDF weights from the training corpus and
        # apply them along with L2 normalization
        tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
        tfidf.fit(X)

    X = tfidf.transform(X)
    return X, vectorizer, tfidf


# ---------------------------------------------------------------------------
# STEP 2: DATA DOWNLOAD
# ---------------------------------------------------------------------------
# Downloads all four HuggingFace datasets needed:
#   arch_retrieval       -> bacterial corpus (train) + archaeal queries (test)
#   arch_retrieval_qrels -> relevance judgments for Arch task
#   euk_retrieval        -> bacterial corpus (train) + eukaryotic queries (test)
#   euk_retrieval_qrels  -> relevance judgments for Euk task

def download_data():
    print("Downloading datasets from HuggingFace...")
    arch_ds    = load_dataset("tattabio/arch_retrieval")
    arch_qrels = load_dataset("tattabio/arch_retrieval_qrels")
    euk_ds     = load_dataset("tattabio/euk_retrieval")
    euk_qrels  = load_dataset("tattabio/euk_retrieval_qrels")
    print("Download complete.")
    return arch_ds, arch_qrels, euk_ds, euk_qrels


# ---------------------------------------------------------------------------
# STEP 3: BUILD GROUND TRUTH LOOKUP
# ---------------------------------------------------------------------------
# The qrels datasets map each query_id to the set of corpus_ids that are
# considered relevant (functionally similar). We store this as a dict:
#   { "Q001": {"C042", "C107"}, "Q002": {"C019"}, ... }
# This is used at evaluation time to check whether retrieved sequences
# are actually relevant to each query.

def build_ground_truth(qrels_ds):
    qrel_split   = list(qrels_ds.keys())[0]
    ground_truth = {}
    for row in qrels_ds[qrel_split]:
        qid = str(row["query_id"])
        cid = str(row["corpus_id"])
        ground_truth.setdefault(qid, set()).add(cid)
    return ground_truth


# ---------------------------------------------------------------------------
# STEP 4: LABEL SIMPLIFICATION
# ---------------------------------------------------------------------------
# Protein name fields in UniProt are long and specific, e.g.:
#   "ATP synthase subunit alpha (EC 3.6.3.14) [Includes: ...]"
# Using the full name as a class label would create thousands of near-unique
# classes, making classification intractable. We keep only the first n_words
# words of the base name (before any parenthetical), reducing "ATP synthase
# subunit alpha" -> "ATP synthase". This creates broader functional groupings
# that the classifier can actually learn.

def simplify_label(label, n_words=2):
    base = label.split("(")[0].strip()
    return " ".join(base.split()[:n_words])


# ---------------------------------------------------------------------------
# STEP 5: TRAIN LOGISTIC REGRESSION
# ---------------------------------------------------------------------------
# Takes a corpus (bacterial sequences with protein name labels), builds
# TF-IDF k-mer features, and trains a multiclass logistic regression
# classifier. The classifier learns to predict functional class from
# k-mer composition.
#
# KEY CORRECTION vs. original: this function now accepts any corpus
# (arch_corpus OR euk_corpus), so it can be called separately for each
# task. Previously it was always called only on arch_corpus.
#
# Solver note: we use "saga" (not "lbfgs" as in the Methods section) because
# saga scales better to the large number of classes and features here.
# lbfgs requires computing the full Hessian and becomes slow/memory-intensive
# at this scale. The paper's Methods section should reflect "saga".

def train_logreg(corpus, k, n_words=2):
    sequences  = [row["Sequence"]      for row in corpus]
    raw_labels = [row["Protein names"] for row in corpus]
    corpus_ids = [str(row["Entry"])    for row in corpus]

    labels = [simplify_label(l, n_words=n_words) for l in raw_labels]
    print(f"  Unique labels: {len(set(labels))}")

    print(f"  Building {k}-mer TF-IDF features for {len(sequences)} sequences...")
    X, vectorizer, tfidf = build_feature_matrix(sequences, k=k)
    print(f"  Feature matrix shape: {X.shape}")

    print(f"  Training logistic regression (this may take several minutes)...")
    model = LogisticRegression(solver="saga", max_iter=200, n_jobs=-1,
                               C=1.0, tol=1e-2, verbose=1)
    model.fit(X, labels)
    print(f"  Training complete.")

    return model, vectorizer, tfidf, corpus_ids


# ---------------------------------------------------------------------------
# STEP 6: RETRIEVAL METRICS
# ---------------------------------------------------------------------------
# All metrics are computed at cutoff k=10, meaning we only look at the top
# 10 retrieved sequences for each query.
#
# nDCG@10: rewards finding relevant sequences AND finding them early.
#   A relevant sequence at rank 1 scores 1/log2(2)=1.0; at rank 2 it
#   scores 1/log2(3)~0.63, etc. The raw DCG is divided by the ideal DCG
#   (what you'd get if all relevant sequences were ranked first).
#
# MRR: simply 1/rank of the FIRST relevant sequence found. Captures whether
#   the model can find at least one correct answer near the top.
#
# Precision@10: fraction of the 10 retrieved sequences that are relevant.
#
# Recall@10: fraction of ALL relevant sequences that appear in top 10.
#   Note: if a query has 50 relevant sequences in the corpus, recall@10
#   is capped at 10/50=0.2 no matter how well the model ranks.
#
# F1@10: harmonic mean of Precision@10 and Recall@10.
#
# HitRate@10: binary -- 1 if ANY relevant sequence is in top 10, else 0.

def ndcg_at_k(ranked_ids, relevant_ids, k=10):
    relevances = [1 if seq_id in relevant_ids else 0
                  for seq_id in ranked_ids[:k]]
    dcg   = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
    ideal = sorted(relevances, reverse=True)
    idcg  = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked_ids, relevant_ids):
    for rank, seq_id in enumerate(ranked_ids, start=1):
        if seq_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def precision_at_k(ranked_ids, relevant_ids, k=10):
    hits = sum(1 for seq_id in ranked_ids[:k] if seq_id in relevant_ids)
    return hits / k


def recall_at_k(ranked_ids, relevant_ids, k=10):
    if not relevant_ids:
        return 0.0
    hits = sum(1 for seq_id in ranked_ids[:k] if seq_id in relevant_ids)
    return hits / len(relevant_ids)


def f1_at_k(ranked_ids, relevant_ids, k=10):
    p = precision_at_k(ranked_ids, relevant_ids, k)
    r = recall_at_k(ranked_ids, relevant_ids, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def hit_rate_at_k(ranked_ids, relevant_ids, k=10):
    return 1.0 if any(seq_id in relevant_ids for seq_id in ranked_ids[:k]) else 0.0


# ---------------------------------------------------------------------------
# STEP 7: EVALUATE
# ---------------------------------------------------------------------------
# How retrieval works with logistic regression (the "blended scores" approach):
#
# The classifier was trained to predict functional class from k-mer features.
# At retrieval time, we use it in an indirect way:
#
#   1. Encode the query sequence into k-mer TF-IDF features.
#   2. Run decision_function on the query -> scores over all classes.
#      Pick the top-3 classes the query is most likely to belong to.
#   3. Run decision_function on every corpus sequence -> a matrix of
#      shape (n_corpus, n_classes).
#   4. For each corpus sequence, average its scores on those top-3 classes.
#      This "blended score" measures how strongly each corpus sequence
#      belongs to the same functional classes as the query.
#   5. Rank corpus sequences by blended score descending and evaluate.
#
# This is why the vectorizer and tfidf passed in MUST have been fit on the
# same corpus that the model was trained on. Mixing arch and euk here
# (the original bug) gives the euk corpus sequences arch-vocabulary features,
# and scores them using a classifier that has never seen euk protein labels.

def evaluate_logreg(queries, ground_truth, model, vectorizer, tfidf,
                    corpus_ids, corpus_sequences, k_mer, k=10, top_n_classes=3):
    """
    Evaluates logistic regression retrieval.
    Returns mean scores and per-query records for all 6 metrics.

    All arguments (model, vectorizer, tfidf, corpus_ids, corpus_sequences)
    must come from the SAME dataset (either all Arch or all Euk).
    """
    per_query_data = []
    total          = len(queries)

    # Encode all corpus sequences using the training vocabulary + IDF weights
    print("  Pre-computing corpus features...")
    X_corpus, _, _ = build_feature_matrix(
        corpus_sequences, k=k_mer,
        vectorizer=vectorizer, tfidf=tfidf
    )

    # Get the classifier's raw class scores for every corpus sequence
    # Shape: (n_corpus_sequences, n_classes)
    print("  Pre-computing corpus decision scores...")
    corpus_scores_matrix = model.decision_function(X_corpus)

    print(f"  Evaluating {total} queries...")
    for row in tqdm(queries, desc="  Evaluating queries", unit="query"):
        query_id  = str(row["Entry"])
        query_seq = row["Sequence"]

        if query_id not in ground_truth:
            continue

        # Encode the single query sequence using the training vocabulary
        X_query, _, _ = build_feature_matrix(
            [query_seq], k=k_mer,
            vectorizer=vectorizer, tfidf=tfidf
        )

        # Find which functional classes the query most likely belongs to
        query_scores      = model.decision_function(X_query)[0]
        top_class_indices = np.argsort(query_scores)[::-1][:top_n_classes]

        # Score each corpus sequence by how strongly it belongs to the
        # same top-3 classes as the query, averaged across those classes
        blended_scores = np.mean(
            corpus_scores_matrix[:, top_class_indices], axis=1
        )

        # Rank corpus sequences from highest to lowest blended score
        ranked_idx   = np.argsort(blended_scores)[::-1]
        ranked_ids   = [corpus_ids[j] for j in ranked_idx[:k]]
        relevant_ids = ground_truth[query_id]

        per_query_data.append({
            "query_id":   query_id,
            "ndcg":       ndcg_at_k(ranked_ids, relevant_ids, k),
            "mrr":        mrr(ranked_ids, relevant_ids),
            "precision":  precision_at_k(ranked_ids, relevant_ids, k),
            "recall":     recall_at_k(ranked_ids, relevant_ids, k),
            "f1":         f1_at_k(ranked_ids, relevant_ids, k),
            "hit_rate":   hit_rate_at_k(ranked_ids, relevant_ids, k),
            "seq_length": len(query_seq),
        })

    def mean_of(key):
        return float(np.mean([d[key] for d in per_query_data])) if per_query_data else 0.0

    means = {m: mean_of(m) for m in
             ["ndcg", "mrr", "precision", "recall", "f1", "hit_rate"]}

    print(f"\n  Scored {len(per_query_data)}/{total} queries")
    print(f"  nDCG@{k}={means['ndcg']:.4f}  MRR={means['mrr']:.4f}  "
          f"P@{k}={means['precision']:.4f}  R@{k}={means['recall']:.4f}  "
          f"F1@{k}={means['f1']:.4f}  HitRate@{k}={means['hit_rate']:.4f}")

    return means, per_query_data


# ---------------------------------------------------------------------------
# STEP 8: PLOTTING
# ---------------------------------------------------------------------------

def plot_score_distribution(all_per_query, method_labels, dataset_label, filename):
    """Histogram of per-query nDCG@10 scores for each method."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for per_query, label in zip(all_per_query, method_labels):
        scores = [d["ndcg"] for d in per_query]
        ax.hist(scores, bins=20, alpha=0.5, label=label)
    ax.set_xlabel("nDCG@10 per query", fontsize=12)
    ax.set_ylabel("Number of queries", fontsize=12)
    ax.set_title(f"Score Distribution -- {dataset_label} Dataset", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved -> {filename}")


def plot_length_vs_score(per_query_data, metric, method_label, dataset_label, filename):
    """Scatter plot of sequence length vs a chosen metric."""
    lengths = [d["seq_length"] for d in per_query_data]
    scores  = [d[metric]       for d in per_query_data]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(lengths, scores, alpha=0.3, s=10)
    ax.set_xlabel("Query sequence length (aa)", fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(
        f"Sequence Length vs {metric.upper()} -- {method_label} ({dataset_label})",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved -> {filename}")


def plot_metrics_bar(all_means, method_labels, dataset_label, filename):
    """Bar chart comparing all 6 metrics across k values."""
    metrics      = ["ndcg", "mrr", "precision", "recall", "f1", "hit_rate"]
    metric_names = ["nDCG@10", "MRR", "P@10", "R@10", "F1@10", "HitRate@10"]
    x            = np.arange(len(metrics))
    width        = 0.8 / len(all_means)

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (means, label) in enumerate(zip(all_means, method_labels)):
        values = [means[m] for m in metrics]
        ax.bar(x + i * width, values, width, label=label, alpha=0.8)

    ax.set_xticks(x + width * (len(all_means) - 1) / 2)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"All Metrics -- {dataset_label} Dataset", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved -> {filename}")


# ---------------------------------------------------------------------------
# STEP 9: MAIN -- CORRECTED PIPELINE
# ---------------------------------------------------------------------------
# The key structural fix: for each k value, we train TWO separate models:
#
#   model_arch: trained on arch_corpus (bacterial sequences from Arch task)
#               evaluated on arch_queries against arch_corpus
#
#   model_euk:  trained on euk_corpus (bacterial sequences from Euk task)
#               evaluated on euk_queries against euk_corpus
#
# Each model has its own vectorizer and tfidf fit on its own corpus, so
# the vocabulary and IDF weights are always appropriate for the task at hand.
# This is the correct way to evaluate cross-domain retrieval independently
# for each benchmark task.

if __name__ == "__main__":

    K_LIST  = [3, 4]
    N_WORDS = 2        # Number of words to keep from protein name as class label

    # --- Download all data ---
    arch_ds, arch_qrels, euk_ds, euk_qrels = download_data()

    # --- Build ground truth relevance dicts ---
    arch_ground_truth = build_ground_truth(arch_qrels)
    euk_ground_truth  = build_ground_truth(euk_qrels)

    # --- Extract splits ---
    # Each dataset has:
    #   "train" -> the bacterial reference corpus to retrieve from
    #   "test"  -> the query sequences (archaeal for Arch, eukaryotic for Euk)
    arch_corpus  = arch_ds["train"]
    arch_queries = arch_ds["test"]
    euk_corpus   = euk_ds["train"]
    euk_queries  = euk_ds["test"]

    # Pre-extract sequences and IDs from each corpus as plain Python lists
    # so they can be passed into build_feature_matrix and evaluate_logreg
    arch_corpus_sequences = [row["Sequence"] for row in arch_corpus]
    arch_corpus_ids       = [str(row["Entry"]) for row in arch_corpus]
    euk_corpus_sequences  = [row["Sequence"] for row in euk_corpus]
    euk_corpus_ids        = [str(row["Entry"]) for row in euk_corpus]

    # --- Ablation loop ---
    ablation_results    = {}
    arch_means_all      = []
    euk_means_all       = []
    method_labels       = []
    best_arch_per_query = None
    best_euk_per_query  = None

    for k_mer in K_LIST:
        run_label = f"LogReg (k={k_mer}, TF-IDF)"

        print(f"\n{'='*60}")
        print(f"  k={k_mer}  |  Features: TF-IDF  |  Label width: {N_WORDS} words")
        print(f"{'='*60}")

        # --- ARCH TASK ---
        # Train on arch bacterial corpus, evaluate arch archaeal queries
        # The vectorizer and tfidf are fit on arch_corpus here.
        print(f"\n[Arch] Training model on arch_corpus (k={k_mer})...")
        model_arch, vec_arch, tfidf_arch, corpus_ids_arch = train_logreg(
            arch_corpus, k=k_mer, n_words=N_WORDS
        )

        print(f"\n--- Arch Dataset Evaluation (k={k_mer}) ---")
        arch_means, arch_per_query = evaluate_logreg(
            arch_queries, arch_ground_truth,
            model_arch, vec_arch, tfidf_arch,
            corpus_ids_arch, arch_corpus_sequences, k_mer=k_mer
        )

        # --- EUK TASK ---
        # Train a SEPARATE model on euk bacterial corpus.
        # This gives a fresh vectorizer and tfidf fit on euk vocabulary,
        # and a classifier trained on euk protein name labels.
        # Previously the arch model was incorrectly reused here.
        print(f"\n[Euk] Training model on euk_corpus (k={k_mer})...")
        model_euk, vec_euk, tfidf_euk, corpus_ids_euk = train_logreg(
            euk_corpus, k=k_mer, n_words=N_WORDS
        )

        print(f"\n--- Euk Dataset Evaluation (k={k_mer}) ---")
        euk_means, euk_per_query = evaluate_logreg(
            euk_queries, euk_ground_truth,
            model_euk, vec_euk, tfidf_euk,
            corpus_ids_euk, euk_corpus_sequences, k_mer=k_mer
        )

        ablation_results[k_mer] = {
            "arch": {m: round(v, 4) for m, v in arch_means.items()},
            "euk":  {m: round(v, 4) for m, v in euk_means.items()},
        }
        arch_means_all.append(arch_means)
        euk_means_all.append(euk_means)
        method_labels.append(run_label)

        # Save per-query data for the best k (k=4) for detailed plots
        if k_mer == 4:
            best_arch_per_query = arch_per_query
            best_euk_per_query  = euk_per_query

    # --- Print ablation table ---
    print("\n" + "=" * 95)
    print("LOGREG K-MER ABLATION -- All Metrics (TF-IDF, 2-word labels)")
    print("=" * 95)
    print(f"{'Method':<28} {'Dataset':<6} {'nDCG':>7} {'MRR':>7} "
          f"{'P@10':>7} {'R@10':>7} {'F1@10':>7} {'HR@10':>7}")
    print("-" * 95)

    for k_mer in K_LIST:
        r    = ablation_results[k_mer]
        name = f"LogReg (k={k_mer}, TF-IDF)"
        for ds_label, ds_key in [("Arch", "arch"), ("Euk", "euk")]:
            m = r[ds_key]
            print(f"{name:<28} {ds_label:<6} {m['ndcg']:>7.4f} {m['mrr']:>7.4f} "
                  f"{m['precision']:>7.4f} {m['recall']:>7.4f} "
                  f"{m['f1']:>7.4f} {m['hit_rate']:>7.4f}")
        print()

    print("--- Reference baselines (nDCG only) ---")
    print(f"{'k-NN (k=4, TF-IDF) Arch':<35} {'0.7456':>7}")
    print(f"{'k-NN (k=4, TF-IDF) Euk':<35}  {'0.7425':>7}")
    print(f"{'BLAST (blastp) Arch':<35}       {'0.9310':>7}")
    print(f"{'BLAST (blastp) Euk':<35}        {'0.8692':>7}")

    # --- Plots (using k=4 results) ---
    plot_score_distribution(
        [best_arch_per_query], ["LogReg (k=4, TF-IDF)"],
        "Arch", "dist_arch_logreg_corrected.png"
    )
    plot_score_distribution(
        [best_euk_per_query], ["LogReg (k=4, TF-IDF)"],
        "Euk", "dist_euk_logreg_corrected.png"
    )
    plot_metrics_bar(
        arch_means_all, method_labels, "Arch",
        "metrics_bar_arch_logreg_corrected.png"
    )
    plot_metrics_bar(
        euk_means_all, method_labels, "Euk",
        "metrics_bar_euk_logreg_corrected.png"
    )
    plot_length_vs_score(
        best_arch_per_query, "ndcg",
        "LogReg (k=4, TF-IDF)", "Arch",
        "length_vs_ndcg_arch_logreg_corrected.png"
    )
    plot_length_vs_score(
        best_euk_per_query, "ndcg",
        "LogReg (k=4, TF-IDF)", "Euk",
        "length_vs_ndcg_euk_logreg_corrected.png"
    )
    plot_length_vs_score(
        best_arch_per_query, "f1",
        "LogReg (k=4, TF-IDF)", "Arch",
        "length_vs_f1_arch_logreg_corrected.png"
    )
    plot_length_vs_score(
        best_euk_per_query, "f1",
        "LogReg (k=4, TF-IDF)", "Euk",
        "length_vs_f1_euk_logreg_corrected.png"
    )