# DeCrypt: Architecture & Model Evolution (CHANGES.md)

This document tracks verified changes, architectural fixes, and empirical benchmark evaluations conducted on the Bitcoin transaction dataset (`data/bitcoin_transactions.csv`).

---

## 1. Before vs. After Comparison Table

All metrics below are derived directly from real executions on `data/bitcoin_transactions.csv` (8,388 transactions, 646 unique addresses):

| Aspect | Legacy Baseline (`--legacy-features`) | Upgraded Architecture (Phase 1–4) |
| :--- | :--- | :--- |
| **Transaction Volume** | Net flow (`abs(sum(amount))`). Deposit 100 + withdraw 100 resulted in `0.0`. | True volume (`sum(abs(amount))`). Deposit 100 + withdraw 100 correctly yields `200.0`. |
| **Transaction Features** | 15 features. Groupby with slow Python lambdas. | 23 vectorized behavioral & temporal features (includes `net_flow`, `pass_through_ratio`, Goh-Barabási `burstiness`, Shannon `amount_entropy`). |
| **Transaction Graph** | Grouped by hash looking for dual withdraw/deposit. Produced **0 nodes, 0 edges**. | **Heuristic flow graph** linking withdrawals to deposits within 24h & 2% amount tolerance: **137 nodes, 137 edges**. |
| **Network Feature Handling** | Network columns had empty object dtype; silently dropped by `select_dtypes(['int64','float64'])`. | Explicit numeric typing via `select_dtypes('number')`. If edges < `min_edges` (10), network features are skipped with an explicit warning; zero-variance columns logged. |
| **Recency Calculation** | Measured from dataset start (`first_tx - min_time`), distorting inactivity. | `days_until_end = (dataset_end - last_tx)`. |
| **Cluster Selection** | Naive elbow heuristic defaulting to the cap ($k=10$, inertia diffs). | Dynamic silhouette maximization ($k=2..10$). Optimal $k=2$ (tabular silhouette = 0.718). |
| **Anomaly Scoring** | Arbitrary affine transformation `((1 - raw) * 50).clip(0, 100)`. | Percentile-rank calibrated risk score normalized strictly across `[0.0, 100.0]`. |
| **Sequence Modeling** | None. | **BERT-style Masked Event Transformer** (64-dim, 4 heads, 2 layers, 15% masking). Anomaly score = leave-one-out event NLL surprise. |
| **Explainability** | None. Only an overall scalar score. | Top-3 most surprising events per address stored with timestamp, amount, direction, and NLL. |
| **Embedding Clustering**| None. | Mean-pooled sequence embeddings clustered via **HDBSCAN** (PCA-16) and KMeans. |
| **Ensemble Scoring** | None. | Rank-average of Isolation Forest and Transformer surprise (0.5 / 0.5 default), calibrated 0–100. |
| **Logging** | Duplicate handler output (every log line printed twice). | Single clean stream logger (`propagate=False`). |
| **Package Management** | Traditional pip. | Fast package management via `uv`. |

---

## 2. Empirical Benchmark Evaluation (5 Seeds, Mean +/- Std)

Evaluated on 5 independent splits (Seeds 42, 43, 44, 45, 46). Test sets consisted of held-out real addresses (assumed normal) combined with 50 synthetic anomalous addresses (10 per archetype) sampled from empirical train distributions:

| Model | AUROC | Average Precision (AP) | Recall@K (Overall) | Structuring | Burst | Pass-Through | Bot-Periodic | Dormant-Spike |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Legacy IF** | 0.890 ± 0.010 | 0.664 ± 0.067 | 0.612 ± 0.030 | 0.90 | 0.82 | 0.34 | 0.04 | 0.96 |
| **Fixed IF** | 0.914 ± 0.013 | 0.656 ± 0.051 | 0.652 ± 0.032 | **0.94** | 0.80 | 0.24 | **0.38** | 0.90 |
| **Transformer**| 0.803 ± 0.049 | 0.566 ± 0.119 | 0.496 ± 0.069 | 0.22 | 0.42 | **1.00** | 0.10 | 0.74 |
| **Ensemble** | **0.919 ± 0.019** | **0.728 ± 0.043** | **0.660 ± 0.022** | 0.60 | 0.56 | 0.94 | 0.22 | **0.98** |

### Benchmark Insights & Findings:
1. **Pass-Through / Mixer Detection:** The Sequence Transformer achieved a **100% recall (1.00)** on rapid deposit-withdrawal pass-through sequences, whereas tabular Isolation Forest only identified 24–34%.
2. **Periodic Bot Activity:** The addition of Goh-Barabási burstiness and inter-event variance improved periodic bot detection from 4% (Legacy IF) to 38% (Fixed IF).
3. **Ensemble Performance:** The Ensemble achieved the highest overall Average Precision (**0.728**) and highest overall Recall@K (**0.660**), demonstrating strong complementarity between tabular deviance and temporal surprise.
4. **Honest Limitations:** Standalone Transformer surprise (AUROC 0.803) did not surpass Isolation Forest on its own because volume-centric anomalies (structuring and high-value bursts) are more directly separable via aggregated tabular statistics. The ensemble bridges this gap.
