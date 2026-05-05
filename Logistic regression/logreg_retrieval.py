"""
Logistic Regression Retrieval for DGEB Benchmark
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datasets import load_dataset
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


# Extract k-mer features using TF-IDF

def kmer_counts(sequence, k):
    return Counter(sequence[i:i+k] for i in range(len(sequence) - k + 1))


def build_feature_matrix(sequences, k, vectorizer=None, tfidf=None):
    """Build TF-IDF weighted k-mer matrix for a single k, with progress bar."""
    kmer_dicts = [
        kmer_counts(seq, k=k)
        for seq in tqdm(sequences, desc=f"  {k}-mer extraction", unit="seq")
    ]

    if vectorizer is None:
        vectorizer = DictVectorizer(sparse=True)
        X = vectorizer.fit_transform(kmer_dicts)
    else:
        X = vectorizer.transform(kmer_dicts)

    if tfidf is None:
        tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
        tfidf.fit(X)

    X = tfidf.transform(X)
    return X, vectorizer, tfidf


# Download data

def download_data():
    print("Downloading datasets from HuggingFace...")
    arch_ds    = load_dataset("tattabio/arch_retrieval")
    arch_qrels = load_dataset("tattabio/arch_retrieval_qrels")
    euk_ds     = load_dataset("tattabio/euk_retrieval")
    euk_qrels  = load_dataset("tattabio/euk_retrieval_qrels")
    print("Download complete.")
    return arch_ds, arch_qrels, euk_ds, euk_qrels


# Build ground truth dict from qrels

def build_ground_truth(qrels_ds):
    qrel_split   = list(qrels_ds.keys())[0]
    ground_truth = {}
    for row in qrels_ds[qrel_split]:
        qid = str(row["query_id"])
        cid = str(row["corpus_id"])
        ground_truth.setdefault(qid, set()).add(cid)
    return ground_truth


# Label simplification

def simplify_label(label, n_words=2):
    """Keep only the first n_words words of the protein name."""
    base = label.split("(")[0].strip()
    return " ".join(base.split()[:n_words])


# Train logistic regression for a single k

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


# Metrics

def ndcg_at_k(ranked_ids, relevant_ids, k=10):
    """Rewards relevant items ranked higher. 1.0 = perfect, 0.0 = none found."""
    relevances = [1 if seq_id in relevant_ids else 0
                  for seq_id in ranked_ids[:k]]
    dcg   = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
    ideal = sorted(relevances, reverse=True)
    idcg  = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked_ids, relevant_ids):
    """Returns 1/rank of the first relevant item found, else 0."""
    for rank, seq_id in enumerate(ranked_ids, start=1):
        if seq_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def precision_at_k(ranked_ids, relevant_ids, k=10):
    """Fraction of top-k results that are relevant."""
    hits = sum(1 for seq_id in ranked_ids[:k] if seq_id in relevant_ids)
    return hits / k


def recall_at_k(ranked_ids, relevant_ids, k=10):
    """Fraction of all relevant items found in top-k."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for seq_id in ranked_ids[:k] if seq_id in relevant_ids)
    return hits / len(relevant_ids)


def f1_at_k(ranked_ids, relevant_ids, k=10):
    """Harmonic mean of Precision@k and Recall@k."""
    p = precision_at_k(ranked_ids, relevant_ids, k)
    r = recall_at_k(ranked_ids, relevant_ids, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def hit_rate_at_k(ranked_ids, relevant_ids, k=10):
    """1 if at least one relevant item is in top-k, else 0."""
    return 1.0 if any(seq_id in relevant_ids for seq_id in ranked_ids[:k]) else 0.0


# Evaluate

def evaluate_logreg(queries, ground_truth, model, vectorizer, tfidf,
                    corpus_ids, corpus_sequences, k_mer, k=10, top_n_classes=3):
    """
    Evaluates logistic regression retrieval with tqdm progress bar.
    Returns mean scores and per-query records for all 6 metrics.
    """
    per_query_data = []
    total          = len(queries)

    print("  Pre-computing corpus features...")
    X_corpus, _, _ = build_feature_matrix(
        corpus_sequences, k=k_mer,
        vectorizer=vectorizer, tfidf=tfidf
    )

    print("  Pre-computing corpus decision scores...")
    corpus_scores_matrix = model.decision_function(X_corpus)

    print(f"  Evaluating {total} queries...")
    for row in tqdm(queries, desc="  Evaluating queries", unit="query"):
        query_id  = str(row["Entry"])
        query_seq = row["Sequence"]

        if query_id not in ground_truth:
            continue

        X_query, _, _ = build_feature_matrix(
            [query_seq], k=k_mer,
            vectorizer=vectorizer, tfidf=tfidf
        )
        query_scores      = model.decision_function(X_query)[0]
        top_class_indices = np.argsort(query_scores)[::-1][:top_n_classes]
        blended_scores    = np.mean(
            corpus_scores_matrix[:, top_class_indices], axis=1
        )

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


# Plotting

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


if __name__ == "__main__":

    K_LIST   = [3, 4] 
    N_WORDS  = 2       

    # 1. Download data
    arch_ds, arch_qrels, euk_ds, euk_qrels = download_data()

    # 2. Ground truth
    arch_ground_truth = build_ground_truth(arch_qrels)
    euk_ground_truth  = build_ground_truth(euk_qrels)

    # 3. Splits
    arch_corpus      = arch_ds["train"]
    arch_queries     = arch_ds["test"]
    euk_corpus       = euk_ds["train"]
    euk_queries      = euk_ds["test"]

    corpus_sequences     = [row["Sequence"] for row in arch_corpus]
    euk_corpus_sequences = [row["Sequence"] for row in euk_corpus]
    euk_corpus_ids       = [str(row["Entry"]) for row in euk_corpus]

    # 4. Run ablation over k values separately
    ablation_results   = {}
    arch_means_all     = []
    euk_means_all      = []
    method_labels      = []
    best_arch_per_query = None
    best_euk_per_query  = None

    for k_mer in K_LIST:
        run_label = f"LogReg (k={k_mer}, TF-IDF)"

        print(f"\n{'='*60}")
        print(f"  k={k_mer}  |  Features: TF-IDF  |  Label width: {N_WORDS} words")
        print(f"{'='*60}")

        model, vectorizer, tfidf, corpus_ids = train_logreg(
            arch_corpus, k=k_mer, n_words=N_WORDS
        )

        print(f"\n--- Arch Dataset (k={k_mer}) ---")
        arch_means, arch_per_query = evaluate_logreg(
            arch_queries, arch_ground_truth,
            model, vectorizer, tfidf,
            corpus_ids, corpus_sequences, k_mer=k_mer
        )

        print(f"\n--- Euk Dataset (k={k_mer}) ---")
        euk_means, euk_per_query = evaluate_logreg(
            euk_queries, euk_ground_truth,
            model, vectorizer, tfidf,
            euk_corpus_ids, euk_corpus_sequences, k_mer=k_mer
        )

        ablation_results[k_mer] = {
            "arch": {m: round(v, 4) for m, v in arch_means.items()},
            "euk":  {m: round(v, 4) for m, v in euk_means.items()},
        }
        arch_means_all.append(arch_means)
        euk_means_all.append(euk_means)
        method_labels.append(run_label)

        # Save best method (k=4) for detailed plots
        if k_mer == 4:
            best_arch_per_query = arch_per_query
            best_euk_per_query  = euk_per_query

    # 5. Print ablation table
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
    print(f"{'k-NN (k=4, TF-IDF) Euk':<35} {'0.7425':>7}")
    print(f"{'BLAST (blastp) Arch':<35} {'0.9310':>7}")
    print(f"{'BLAST (blastp) Euk':<35} {'0.8692':>7}")

    # 6. Plots
    plot_score_distribution(
        [best_arch_per_query], ["LogReg (k=4, TF-IDF)"],
        "Arch", "dist_arch_logreg_v5.png"
    )
    plot_score_distribution(
        [best_euk_per_query], ["LogReg (k=4, TF-IDF)"],
        "Euk", "dist_euk_logreg_v5.png"
    )

    plot_metrics_bar(
        arch_means_all, method_labels, "Arch",
        "metrics_bar_arch_logreg_v5.png"
    )
    plot_metrics_bar(
        euk_means_all, method_labels, "Euk",
        "metrics_bar_euk_logreg_v5.png"
    )

    plot_length_vs_score(
        best_arch_per_query, "ndcg",
        "LogReg (k=4, TF-IDF)", "Arch",
        "length_vs_ndcg_arch_logreg_v5.png"
    )
    plot_length_vs_score(
        best_euk_per_query, "ndcg",
        "LogReg (k=4, TF-IDF)", "Euk",
        "length_vs_ndcg_euk_logreg_v5.png"
    )
    plot_length_vs_score(
        best_arch_per_query, "f1",
        "LogReg (k=4, TF-IDF)", "Arch",
        "length_vs_f1_arch_logreg_v5.png"
    )
    plot_length_vs_score(
        best_euk_per_query, "f1",
        "LogReg (k=4, TF-IDF)", "Euk",
        "length_vs_f1_euk_logreg_v5.png"
    )