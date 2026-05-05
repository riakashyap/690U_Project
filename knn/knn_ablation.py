"""
k-NN Retrieval Ablation — DGEB Benchmark (Updated v3)
======================================================
Compares two feature representations:
  1. Raw k-mer frequency (L1-normalized) for k = 3, 4, 5
  2. TF-IDF weighted k-mers for k = 3, 4, 5

Metrics:
  - nDCG@10, MRR, Precision@10, Recall@10, F1@10, HitRate@10

Run with:
    python3 knn_ablation_v3.py
"""

import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datasets import load_dataset
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors


# ══════════════════════════════════════════════════════════════
# STEP 1: k-mer feature extraction
# ══════════════════════════════════════════════════════════════

def kmer_counts(sequence, k):
    return Counter(sequence[i:i+k] for i in range(len(sequence) - k + 1))


def build_feature_matrix(sequences, k, vectorizer=None, tfidf=None):
    kmer_dicts = [kmer_counts(seq, k=k) for seq in sequences]

    if vectorizer is None:
        vectorizer = DictVectorizer(sparse=True)
        X = vectorizer.fit_transform(kmer_dicts)
    else:
        X = vectorizer.transform(kmer_dicts)

    if tfidf is not None:
        X = tfidf.transform(X)
    else:
        X = normalize(X, norm="l1")

    return X, vectorizer


def build_tfidf_transformer(X_corpus):
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
    tfidf.fit(X_corpus)
    return tfidf


# ══════════════════════════════════════════════════════════════
# STEP 2: Download data
# ══════════════════════════════════════════════════════════════

def download_data():
    print("Downloading datasets from HuggingFace...")
    arch_ds    = load_dataset("tattabio/arch_retrieval")
    arch_qrels = load_dataset("tattabio/arch_retrieval_qrels")
    euk_ds     = load_dataset("tattabio/euk_retrieval")
    euk_qrels  = load_dataset("tattabio/euk_retrieval_qrels")
    print("Download complete.")
    return arch_ds, arch_qrels, euk_ds, euk_qrels


# ══════════════════════════════════════════════════════════════
# STEP 3: Extract queries, corpus, and ground truth
# ══════════════════════════════════════════════════════════════

def extract_retrieval_data(ds, qrels_ds, label):
    qrel_split = list(qrels_ds.keys())[0]
    qrels      = qrels_ds[qrel_split]

    if len(ds["train"]) > len(ds["test"]):
        corpus_split, query_split = "train", "test"
    else:
        corpus_split, query_split = "test", "train"

    queries = ds[query_split]
    corpus  = ds[corpus_split]

    print(f"[{label}] Queries : {len(queries)}  |  Corpus : {len(corpus)}  |  Qrels : {len(qrels)}")

    ground_truth = {}
    for row in qrels:
        qid = str(row["query_id"])
        cid = str(row["corpus_id"])
        ground_truth.setdefault(qid, set()).add(cid)

    query_records  = [(str(r["Entry"]), r["Sequence"]) for r in queries]
    corpus_records = [(str(r["Entry"]), r["Sequence"]) for r in corpus]
    return query_records, ground_truth, corpus_records


# ══════════════════════════════════════════════════════════════
# STEP 4: Build merged bacterial corpus
# ══════════════════════════════════════════════════════════════

def build_bacterial_corpus(arch_ds, euk_ds):
    seen_ids         = set()
    bacterial_corpus = []

    for ds_split in [arch_ds["train"], euk_ds["train"]]:
        for row in ds_split:
            entry_id = str(row["Entry"])
            if entry_id not in seen_ids:
                bacterial_corpus.append((entry_id, row["Sequence"]))
                seen_ids.add(entry_id)

    print(f"Bacterial corpus: {len(bacterial_corpus)} unique sequences")
    return bacterial_corpus


# ══════════════════════════════════════════════════════════════
# STEP 5: Build k-NN index
# ══════════════════════════════════════════════════════════════

def build_knn_index(corpus_matrix):
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=1)
    nn.fit(corpus_matrix)
    return nn


# ══════════════════════════════════════════════════════════════
# STEP 6: Retrieve top-k for one query
# ══════════════════════════════════════════════════════════════

def knn_retrieve(query_seq, knn_index, vectorizer, corpus_ids,
                 k_mer, tfidf=None, top_k=10):
    X_query, _ = build_feature_matrix(
        [query_seq], k=k_mer, vectorizer=vectorizer, tfidf=tfidf
    )
    distances, indices = knn_index.kneighbors(X_query, n_neighbors=top_k)
    return [corpus_ids[i] for i in indices[0]]


# ══════════════════════════════════════════════════════════════
# STEP 7: Metrics
# ══════════════════════════════════════════════════════════════

def ndcg_at_k(ranked_ids, relevant_ids, k=10):
    """
    Normalized Discounted Cumulative Gain.
    Rewards relevant items ranked higher. 1.0 = perfect, 0.0 = none found.
    """
    relevances = [1 if seq_id in relevant_ids else 0
                  for seq_id in ranked_ids[:k]]
    dcg   = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
    ideal = sorted(relevances, reverse=True)
    idcg  = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked_ids, relevant_ids):
    """
    Mean Reciprocal Rank.
    Returns 1/rank of the first relevant item found, else 0.
    Example: first relevant item at rank 3 -> MRR = 1/3 = 0.333
    """
    for rank, seq_id in enumerate(ranked_ids, start=1):
        if seq_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def precision_at_k(ranked_ids, relevant_ids, k=10):
    """
    Precision@k: fraction of top-k retrieved results that are relevant.
    Example: 3 relevant in top 10 -> 0.30
    """
    hits = sum(1 for seq_id in ranked_ids[:k] if seq_id in relevant_ids)
    return hits / k


def recall_at_k(ranked_ids, relevant_ids, k=10):
    """
    Recall@k: fraction of all relevant items that appear in top-k.
    Example: found 3 out of 20 relevant -> 0.15
    """
    if not relevant_ids:
        return 0.0
    hits = sum(1 for seq_id in ranked_ids[:k] if seq_id in relevant_ids)
    return hits / len(relevant_ids)


def f1_at_k(ranked_ids, relevant_ids, k=10):
    """
    F1@k: harmonic mean of Precision@k and Recall@k.
    Balances quality (precision) with coverage (recall).
    """
    p = precision_at_k(ranked_ids, relevant_ids, k)
    r = recall_at_k(ranked_ids, relevant_ids, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def hit_rate_at_k(ranked_ids, relevant_ids, k=10):
    """
    Hit Rate@k: 1 if at least one relevant item is in top-k, else 0.
    Answers: did the method find anything useful at all?
    """
    return 1.0 if any(seq_id in relevant_ids for seq_id in ranked_ids[:k]) else 0.0


# ══════════════════════════════════════════════════════════════
# STEP 8: Evaluate over all queries
# ══════════════════════════════════════════════════════════════

def evaluate_knn(query_records, ground_truth, knn_index,
                 vectorizer, corpus_ids, k_mer, tfidf=None, k=10):
    """
    Evaluates k-NN retrieval with all 6 metrics.
    Returns mean scores dict and per-query records.
    """
    per_query_data = []
    total          = len(query_records)

    for i, (query_id, seq) in enumerate(query_records):
        if query_id not in ground_truth:
            continue
        try:
            ranked_ids   = knn_retrieve(
                seq, knn_index, vectorizer, corpus_ids,
                k_mer=k_mer, tfidf=tfidf, top_k=k
            )
            relevant_ids = ground_truth[query_id]

            ndcg_score = ndcg_at_k(ranked_ids, relevant_ids, k)
            mrr_score  = mrr(ranked_ids, relevant_ids)
            p_score    = precision_at_k(ranked_ids, relevant_ids, k)
            r_score    = recall_at_k(ranked_ids, relevant_ids, k)
            f1_score   = f1_at_k(ranked_ids, relevant_ids, k)
            hr_score   = hit_rate_at_k(ranked_ids, relevant_ids, k)

            per_query_data.append({
                "query_id":  query_id,
                "ndcg":      ndcg_score,
                "mrr":       mrr_score,
                "precision": p_score,
                "recall":    r_score,
                "f1":        f1_score,
                "hit_rate":  hr_score,
                "seq_length": len(seq),
            })

            print(f"  [{i+1}/{total}] {query_id}: "
                  f"nDCG={ndcg_score:.3f}  MRR={mrr_score:.3f}  "
                  f"P={p_score:.3f}  R={r_score:.3f}  "
                  f"F1={f1_score:.3f}  HR={hr_score:.0f}")

        except Exception as e:
            print(f"  [{i+1}/{total}] {query_id}: ERROR -- {e}")

    def mean_of(key):
        return float(np.mean([d[key] for d in per_query_data])) if per_query_data else 0.0

    means = {m: mean_of(m) for m in
             ["ndcg", "mrr", "precision", "recall", "f1", "hit_rate"]}

    print(f"\n  Scored {len(per_query_data)}/{total} queries")
    print(f"  nDCG@{k}={means['ndcg']:.4f}  MRR={means['mrr']:.4f}  "
          f"P@{k}={means['precision']:.4f}  R@{k}={means['recall']:.4f}  "
          f"F1@{k}={means['f1']:.4f}  HR@{k}={means['hit_rate']:.4f}")

    return means, per_query_data


# ══════════════════════════════════════════════════════════════
# STEP 9: Plotting
# ══════════════════════════════════════════════════════════════

def plot_score_distribution(all_per_query, method_labels, dataset_label, filename):
    """
    Histogram of per-query nDCG@10 scores for each method.
    Shows whether methods are consistently good or have high variance.
    """
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
    """
    Scatter plot of sequence length vs a chosen metric score.
    Shows whether short sequences are harder to retrieve correctly.
    """
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
    """
    Bar chart comparing all 6 metrics across all k-mer methods.
    Gives a quick visual summary of which method wins on each metric.
    """
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
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved -> {filename}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # 1. Download data once
    arch_ds, arch_qrels, euk_ds, euk_qrels = download_data()

    # 2. Extract queries and ground truth once
    arch_query_records, arch_ground_truth, _ = extract_retrieval_data(
        arch_ds, arch_qrels, "Arch"
    )
    euk_query_records, euk_ground_truth, _ = extract_retrieval_data(
        euk_ds, euk_qrels, "Euk"
    )

    # 3. Build bacterial corpus once
    bacterial_corpus = build_bacterial_corpus(arch_ds, euk_ds)
    corpus_ids       = [entry_id for entry_id, _ in bacterial_corpus]
    corpus_sequences = [seq for _, seq in bacterial_corpus]

    # 4. Run ablation for k=3,4,5 with both raw and TF-IDF features
    ablation_results    = {}
    best_arch_per_query = None
    best_euk_per_query  = None
    arch_means_all      = []
    euk_means_all       = []
    method_labels       = []

    for k_mer in [3, 4, 5]:
        ablation_results[k_mer] = {}

        for use_tfidf in [False, True]:
            feat_label = "TF-IDF" if use_tfidf else "Raw"
            run_label  = f"k-NN (k={k_mer}, {feat_label})"

            print(f"\n{'='*60}")
            print(f"  k={k_mer}  |  Features: {feat_label} k-mer frequencies")
            print(f"{'='*60}")

            # Build raw corpus feature matrix
            print(f"Building {k_mer}-mer features for corpus...")
            corpus_matrix, vectorizer = build_feature_matrix(
                corpus_sequences, k=k_mer
            )
            print(f"Corpus matrix shape: {corpus_matrix.shape}")

            # Fit TF-IDF if needed
            tfidf = None
            if use_tfidf:
                print("Fitting TF-IDF transformer on corpus...")
                tfidf         = build_tfidf_transformer(corpus_matrix)
                corpus_matrix = tfidf.transform(corpus_matrix)

            knn_index = build_knn_index(corpus_matrix)

            # Evaluate Arch
            print(f"\n--- Arch Dataset (k={k_mer}, {feat_label}) ---")
            arch_means, arch_per_query = evaluate_knn(
                arch_query_records, arch_ground_truth,
                knn_index, vectorizer, corpus_ids,
                k_mer=k_mer, tfidf=tfidf
            )

            # Evaluate Euk
            print(f"\n--- Euk Dataset (k={k_mer}, {feat_label}) ---")
            euk_means, euk_per_query = evaluate_knn(
                euk_query_records, euk_ground_truth,
                knn_index, vectorizer, corpus_ids,
                k_mer=k_mer, tfidf=tfidf
            )

            ablation_results[k_mer][feat_label] = {
                "arch": {k: round(v, 4) for k, v in arch_means.items()},
                "euk":  {k: round(v, 4) for k, v in euk_means.items()},
                "mean_ndcg": round((arch_means["ndcg"] + euk_means["ndcg"]) / 2, 4),
            }

            arch_means_all.append(arch_means)
            euk_means_all.append(euk_means)
            method_labels.append(run_label)

            # Save per-query data from best method (k=4, TF-IDF) for plots
            if k_mer == 4 and use_tfidf:
                best_arch_per_query = arch_per_query
                best_euk_per_query  = euk_per_query

    # 5. Print full ablation table with all 6 metrics
    print("\n" + "=" * 95)
    print("K-MER ABLATION -- All Metrics")
    print("=" * 95)
    print(f"{'Method':<28} {'Dataset':<6} {'nDCG':>7} {'MRR':>7} "
          f"{'P@10':>7} {'R@10':>7} {'F1@10':>7} {'HR@10':>7}")
    print("-" * 95)

    for k_mer in [3, 4, 5]:
        for feat_label in ["Raw", "TF-IDF"]:
            r    = ablation_results[k_mer][feat_label]
            name = f"k-NN (k={k_mer}, {feat_label})"
            for ds_label, ds_key in [("Arch", "arch"), ("Euk", "euk")]:
                m = r[ds_key]
                print(f"{name:<28} {ds_label:<6} {m['ndcg']:>7.4f} {m['mrr']:>7.4f} "
                      f"{m['precision']:>7.4f} {m['recall']:>7.4f} "
                      f"{m['f1']:>7.4f} {m['hit_rate']:>7.4f}")
        print()

    # 6. Print full comparison table (nDCG only, for reference vs baselines)
    print("\n" + "=" * 70)
    print("FULL COMPARISON TABLE (nDCG@10)")
    print("=" * 70)
    print(f"{'Method':<30} {'Arch nDCG@10':>14} {'Euk nDCG@10':>13}")
    print("-" * 70)
    print(f"{'BLAST (blastp)':<30} {'0.9310':>14} {'0.8692':>13}")
    for k_mer in [3, 4, 5]:
        for feat_label in ["Raw", "TF-IDF"]:
            r    = ablation_results[k_mer][feat_label]
            name = f"k-NN (k={k_mer}, {feat_label})"
            print(f"{name:<30} {r['arch']['ndcg']:>14.4f} {r['euk']['ndcg']:>13.4f}")
    print(f"{'LogReg (improved)':<30} {'0.0632':>14} {'0.1158':>13}")
    print(f"{'LogReg (baseline)':<30} {'0.0030':>14} {'0.1363':>13}")

    # 7. Save full results to JSON
    with open("knn_ablation_results_v3.json", "w") as f:
        json.dump({
            "generated_at":        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ablation":            ablation_results,
            "best_arch_per_query": best_arch_per_query,
            "best_euk_per_query":  best_euk_per_query,
        }, f, indent=2)
    print("\nSaved -> knn_ablation_results_v3.json")

    # 8. Score distribution plots for best method (k=4, TF-IDF)
    if best_arch_per_query:
        plot_score_distribution(
            all_per_query = [best_arch_per_query],
            method_labels = ["k-NN (k=4, TF-IDF)"],
            dataset_label = "Arch",
            filename      = "dist_arch_knn_v3.png"
        )
        plot_length_vs_score(
            best_arch_per_query, "ndcg",
            "k-NN (k=4, TF-IDF)", "Arch",
            "length_vs_ndcg_arch_v3.png"
        )
        plot_length_vs_score(
            best_arch_per_query, "f1",
            "k-NN (k=4, TF-IDF)", "Arch",
            "length_vs_f1_arch_v3.png"
        )

    if best_euk_per_query:
        plot_score_distribution(
            all_per_query = [best_euk_per_query],
            method_labels = ["k-NN (k=4, TF-IDF)"],
            dataset_label = "Euk",
            filename      = "dist_euk_knn_v3.png"
        )
        plot_length_vs_score(
            best_euk_per_query, "ndcg",
            "k-NN (k=4, TF-IDF)", "Euk",
            "length_vs_ndcg_euk_v3.png"
        )
        plot_length_vs_score(
            best_euk_per_query, "f1",
            "k-NN (k=4, TF-IDF)", "Euk",
            "length_vs_f1_euk_v3.png"
        )

    # 9. Metrics bar chart across all methods
    plot_metrics_bar(
        arch_means_all, method_labels, "Arch",
        "metrics_bar_arch_knn_v3.png"
    )
    plot_metrics_bar(
        euk_means_all, method_labels, "Euk",
        "metrics_bar_euk_knn_v3.png"
    )
