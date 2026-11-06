# DeCrypt: Architecture & Model Evolution (CHANGES.md)

This document tracks verified changes, architectural fixes, and empirical benchmark evaluations conducted on the Bitcoin transaction dataset (`data/bitcoin_transactions.csv`).

---

## 1. Before vs. After Comparison Table

All metrics below are derived directly from real executions on `data/bitcoin_transactions.csv` (8,388 transactions, 646 unique addresses):

| Aspect | Legacy Baseline (`--legacy-features`) | Upgraded Architecture (Phase 1–4 + Validation v2) |
| :--- | :--- | :--- |
| **Transaction Volume** | Net flow (`abs(sum(amount))`). Deposit 100 + withdraw 100 resulted in `0.0`. | True volume (`sum(abs(amount))`). Deposit 100 + withdraw 100 correctly yields `200.0`. |
| **Transaction Features** | 15 features. Groupby with slow Python lambdas. | 26 vectorized behavioral & temporal features (includes `net_flow`, `pass_through_ratio`, Goh-Barabási `burstiness`, Shannon `amount_entropy`, `cv_inter_event_time`, `modal_amount_share`, `lag1_autocorr_inter_event_time`). |
| **Transaction Graph** | Grouped by hash looking for dual withdraw/deposit. Produced **0 nodes, 0 edges**. | **Negative Result**: Heuristic flow graph yields 137 edges, but null permutation models produce equal or more edges (shuffled time: 183.5, shuffled amount: 154.5). Ablation proves network features degrade AUROC (-0.0208) and AP (-0.0566); **excluded from scoring by default**. |
| **Network Feature Handling** | Network columns had empty object dtype; silently dropped by `select_dtypes(['int64','float64'])`. | Explicit numeric typing via `select_dtypes('number')`. Retained behind optional `--include-network-features` flag; zero-variance columns logged. |
| **Recency Calculation** | Measured from dataset start (`first_tx - min_time`), distorting inactivity. | `days_until_end = (dataset_end - last_tx)`. |
| **Cluster Selection** | Naive elbow heuristic defaulting to the cap ($k=10$, inertia diffs). | Dynamic silhouette maximization ($k=2..10$). Optimal $k=2$ on tabular features. Flagged: Cluster 1 contains 98.14% of addresses (>90% single cluster dominance). |
| **Anomaly Scoring** | Arbitrary affine transformation `((1 - raw) * 50).clip(0, 100)`. | Percentile-rank calibrated risk score normalized strictly across `[0.0, 100.0]`. |
| **Sequence Modeling** | None. | **BERT-style Masked Event Transformer** (64-dim, 4 heads, 2 layers, 15% masking). Anomaly score = leave-one-out event NLL surprise. |
| **Explainability** | None. Only an overall scalar score. | Top-3 most surprising events per address stored with timestamp, amount, direction, and NLL. |
| **Embedding Clustering**| None. | Mean-pooled sequence embeddings clustered via **HDBSCAN** strictly on addresses with $\ge 3$ events ($N=372$). Silhouette = 0.0984, sizes: 339, 7, and 26 noise points. |
| **Ensemble Scoring** | None. | Multi-strategy rank calibration: Mean-Rank (0.5/0.5), Max-Rank ($\max(\text{rank}_{\text{IF}}, \text{rank}_{\text{TF}})$), and Tuned Weighted Fusion ($0.8 \times \text{rank}_{\text{IF}} + 0.2 \times \text{rank}_{\text{TF}}$). |
| **Logging** | Duplicate handler output (every log line printed twice). | Single clean stream logger (`propagate=False`). |
| **Package Management** | Traditional pip. | Fast package management via `uv`. |

---

## 2. Empirical Benchmark Evaluation (10 Seeds, 20/archetype = 100 Anomalies/Seed, 95% Bootstrap CI)

> [!IMPORTANT]
> **Generator Versioning Note**: All benchmark metrics below are produced by **Generator `v2.0`** (recorded in `benchmark_results.json`), incorporating pass-through delay variants (15m, 1h, 6h, 24h), empirical structuring amount caps ($\le 250$ BTC), and bot-detection features (`cv_inter_event_time`, `modal_amount_share`, `lag1_autocorr_inter_event_time`). Benchmark numbers from earlier generator versions (`v1.0`) are not directly comparable due to these generator updates.

Evaluated across 10 independent test splits (Seeds 42 through 51) with 20 synthetic anomalies injected per archetype (100 anomalies per seed; 1,000 total evaluations). Each cell displays the empirical mean and [95% bootstrap confidence interval] computed over 1,000 bootstrap resamples:

| Model | AUROC [95% CI] | AP [95% CI] | Recall@K [95% CI] | Structuring | Burst | Pass-Through | Bot-Periodic | Dormant-Spike |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Legacy IF** | 0.884 [0.869, 0.897] | 0.793 [0.763, 0.822] | 0.713 [0.688, 0.736] | 0.92 [0.87, 0.96] | 0.93 [0.88, 0.98] | 0.55 [0.46, 0.65] | 0.18 [0.12, 0.25] | 0.98 [0.96, 0.99] |
| **Fixed IF** | 0.948 [0.942, 0.955] | 0.875 [0.855, 0.894] | 0.824 [0.810, 0.837] | **0.98 [0.97, 1.00]** | **0.95 [0.89, 0.98]** | 0.50 [0.42, 0.58] | **0.70 [0.65, 0.76]** | 0.98 [0.97, 1.00] |
| **Transformer** | 0.813 [0.789, 0.835] | 0.723 [0.688, 0.756] | 0.600 [0.566, 0.629] | 0.27 [0.10, 0.45] | 0.48 [0.23, 0.72] | **1.00 [1.00, 1.00]** | 0.28 [0.18, 0.41] | 0.96 [0.90, 1.00] |
| **Ensemble (Mean-Rank)** | 0.944 [0.936, 0.952] | 0.892 [0.873, 0.909] | 0.793 [0.771, 0.814] | 0.83 [0.75, 0.90] | 0.68 [0.44, 0.87] | 0.99 [0.97, 1.00] | 0.47 [0.34, 0.61] | **0.99 [0.98, 1.00]** |
| **Ensemble (Max-Rank)** | 0.927 [0.917, 0.937] | 0.855 [0.829, 0.881] | 0.764 [0.742, 0.788] | 0.80 [0.74, 0.88] | 0.77 [0.64, 0.87] | 0.97 [0.94, 1.00] | 0.30 [0.24, 0.37] | 0.97 [0.94, 0.99] |
| **Ensemble (Tuned 0.8/0.2)** | **0.960 [0.955, 0.965]** | **0.902 [0.886, 0.917]** | **0.846 [0.832, 0.859]** | 0.94 [0.90, 0.97] | 0.94 [0.86, 0.99] | 0.72 [0.68, 0.77] | 0.63 [0.57, 0.69] | **0.99 [0.97, 1.00]** |

---

## 3. Paired Bootstrap Confidence Intervals of the Difference (Tuned Fusion vs. Fixed IF)

To rigorously test whether the Tuned Weighted Ensemble ($0.8 \times \text{rank}_{\text{IF}} + 0.2 \times \text{rank}_{\text{TF}}$) statistically significantly outperforms standalone Fixed Isolation Forest, paired differences $\Delta = \text{Tuned} - \text{Fixed IF}$ were evaluated across all 10 evaluation seeds using 1,000 bootstrap resamples:

| Metric | Paired Mean Difference ($\Delta$) | 95% Bootstrap CI | Statistical Significance |
| :--- | :---: | :---: | :--- |
| **$\Delta$ AUROC** | **+0.0119** | **[+0.0094, +0.0141]** | **Statistically Significant ($p < 0.05$)** |
| **$\Delta$ Average Precision (AP)** | **+0.0278** | **[+0.0181, +0.0373]** | **Statistically Significant ($p < 0.05$)** |
| **$\Delta$ Recall@K** | **+0.0220** | **[+0.0120, +0.0330]** | **Statistically Significant ($p < 0.05$)** |

Because zero lies outside all three 95% bootstrap confidence intervals, Tuned Ensemble demonstrates a statistically significant superiority over standalone tabular Isolation Forest.

---

## 4. Deep-Dive Audits & Validation Findings

### 4.1 Pass-Through Delay Variant Analysis (15m, 1h, 6h, 24h)
Pass-through synthetic anomalies were injected across four distinct delay variants to measure the persistence of temporal surprise over increasing dormancy:

| Delay Variant | Transformer Recall [95% CI] | Fixed IF Recall [95% CI] | Tuned Ensemble Recall [95% CI] |
| :--- | :---: | :---: | :---: |
| **15 minutes (2–15m)** | **1.00 [1.00, 1.00]** | 0.54 [0.40, 0.66] | 0.68 [0.52, 0.82] |
| **1 hour (50–70m)** | **1.00 [1.00, 1.00]** | 0.40 [0.30, 0.50] | 0.58 [0.46, 0.68] |
| **6 hours (5–7h)** | **1.00 [1.00, 1.00]** | 0.52 [0.42, 0.62] | 0.82 [0.74, 0.90] |
| **24 hours (22–26h)** | **1.00 [1.00, 1.00]** | 0.54 [0.38, 0.68] | 0.82 [0.74, 0.90] |

* **Insight**: The Sequence Transformer achieves **1.00 recall across all delay intervals from 15 minutes to 24 hours**. While tabular IF remains stagnant at ~40–54%, the Transformer's learned conditional event model reliably flags matched-amount reversals regardless of whether the funds dwell for minutes or an entire day.

### 4.2 Network Feature Ablation: Proof of Negative Result
To verify whether the heuristic flow graph adds value or noise, we conducted an empirical benchmark ablation comparing Fixed IF without network features (default) against Fixed IF with network features:

| Configuration | AUROC [95% CI] | AP [95% CI] | Recall@K [95% CI] |
| :--- | :---: | :---: | :---: |
| **Without Network Features (Default)** | **0.948 [0.942, 0.955]** | **0.875 [0.855, 0.894]** | **0.824 [0.810, 0.837]** |
| **With Network Features (Ablation)** | 0.927 [0.919, 0.936] | 0.818 [0.785, 0.847] | 0.774 [0.757, 0.791] |
| **Paired Difference (With - Without)** | **-0.0208 [-0.0286, -0.0135]** | **-0.0566 [-0.0764, -0.0369]** | **-0.0500 [-0.0680, -0.0320]** |

* **Conclusion**: Adding network features degrades AUROC by -0.0208, AP by -0.0566, and Recall@K by -0.0500 with statistical significance. Combined with the permutation test (where shuffled null models produce 154–183 edges vs. real 137), the heuristic graph is formally reclassified as a **Negative Result** and excluded from scoring by default.

### 4.3 Bot-Periodic Feature Engineering Impact
Introducing three targeted behavioral features:
1. **CV of inter-event times** ($\sigma / \mu$, approaching 0 for constant-rate bots)
2. **Modal-amount share** ($\text{count}(\text{mode}) / \text{total\_txs}$, equal to 1.0 for fixed-amount bots)
3. **Lag-1 autocorrelation** of $\Delta t$

* **Impact on Detection**:
  * Fixed IF bot-periodic recall jumped from **0.50 [0.42, 0.58]** in baseline to **0.70 [0.65, 0.76]** (+20 percentage points, a 40% relative increase).
  * Tuned Ensemble bot-periodic recall increased from **0.47** to **0.63 [0.57, 0.69]**.

### 4.4 Empirical Temporal Intervals in Real Data
An audit of all 646 real addresses in `data/bitcoin_transactions.csv` reveals:
* **Deposit followed by withdrawal within 15 minutes**: **29 addresses (4.49%)**.
* **Inter-event gaps under 60 seconds**: **61 addresses (9.44%)**.
These empirical rates confirm that rapid turnarounds and high-frequency bursts exist naturally in Bitcoin transaction flows (e.g. payment relays and automated trading), underscoring why tabular models require sequential modeling to discern illicit pass-throughs from benign activity.

### 4.5 HDBSCAN Clustering on Addresses with $\ge 3$ Events
* When evaluated strictly on addresses with sufficient sequential history ($N = 372$):
  * **HDBSCAN**: Identifies 2 clusters with silhouette = **0.0984** and 26 noise points. Cluster sizes: Cluster 0 contains **339 entities** (91.13%), Cluster 1 contains **7 entities** (1.88%), and **26 points** are noise (-1).
  * **KMeans ($k=2$)**: Achieves silhouette = **0.4180** with balanced clusters (**187** and **185** entities).
* **Correction**: Previous claims that HDBSCAN "partitioned" the dataset were an artifact of clustering zero-padded dummy embeddings of short-history addresses. Evaluated on real embeddings, HDBSCAN groups the vast majority into a single core cluster with a tiny 7-address satellite.
