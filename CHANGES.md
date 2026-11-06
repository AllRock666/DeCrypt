# DeCrypt: Architecture & Model Evolution (CHANGES.md)

This document tracks verified changes, architectural fixes, and empirical benchmark evaluations conducted on the Bitcoin transaction dataset (`data/bitcoin_transactions.csv`).

---

## 1. Before vs. After Comparison Table

All metrics below are derived directly from real executions on `data/bitcoin_transactions.csv` (8,388 transactions, 646 unique addresses):

| Aspect | Legacy Baseline (`--legacy-features`) | Upgraded Architecture (Phase 1–4 + Validation v2) |
| :--- | :--- | :--- |
| **Transaction Volume** | Net flow (`abs(sum(amount))`). Deposit 100 + withdraw 100 resulted in `0.0`. | True volume (`sum(abs(amount))`). Deposit 100 + withdraw 100 correctly yields `200.0`. |
| **Transaction Features** | 15 features. Groupby with slow Python lambdas. | 29 total features (23 vectorized behavioral & temporal features, including `net_flow`, `pass_through_ratio`, Goh-Barabási `burstiness`, Shannon `amount_entropy`, `cv_inter_event_time`, `modal_amount_share`, `lag1_autocorr_inter_event_time`, `direction_switch_count`, `alternation_rate`, and `max_alternating_run_length`). |
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
> **Generator Versioning Note**: All benchmark metrics below are produced by **Generator `v2.0`** (recorded in `benchmark_results.json`), incorporating alternating deposit/withdraw delay variants (15m, 1h, 6h, 24h), empirical structuring amount caps ($\le 250$ BTC), and updated behavioral features (`cv_inter_event_time`, `modal_amount_share`, `lag1_autocorr_inter_event_time`, `direction_switch_count`, `alternation_rate`, `max_alternating_run_length`). Benchmark numbers from earlier generator versions (`v1.0`) are not directly comparable due to these updates.

Evaluated across 10 independent test splits (Seeds 42 through 51) with 20 synthetic anomalies injected per archetype (100 anomalies per seed; 1,000 total evaluations). Each cell displays the empirical mean and [95% bootstrap confidence interval] computed over 1,000 bootstrap resamples:

| Model | AUROC [95% CI] | AP [95% CI] | Recall@K [95% CI] | Structuring | Burst | Alternating Dep/Wd | Bot-Periodic | Dormant-Spike |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Legacy IF** | 0.884 [0.869, 0.897] | 0.793 [0.763, 0.822] | 0.713 [0.688, 0.736] | 0.92 [0.87, 0.96] | 0.93 [0.88, 0.98] | 0.55 [0.46, 0.65] | 0.18 [0.12, 0.25] | 0.98 [0.96, 0.99] |
| **Fixed IF** | 0.966 [0.961, 0.972] | 0.899 [0.874, 0.921] | 0.863 [0.850, 0.875] | **0.88 [0.83, 0.93]** | **0.94 [0.87, 0.99]** | 0.83 [0.76, 0.89] | **0.71 [0.65, 0.76]** | 0.96 [0.93, 0.99] |
| **Transformer** | 0.812 [0.788, 0.833] | 0.717 [0.670, 0.756] | 0.610 [0.581, 0.637] | 0.30 [0.14, 0.47] | 0.48 [0.24, 0.71] | **1.00 [1.00, 1.00]** | 0.30 [0.19, 0.42] | 0.96 [0.91, 1.00] |
| **Ensemble (Mean-Rank)** | 0.941 [0.928, 0.953] | 0.890 [0.867, 0.911] | 0.784 [0.754, 0.811] | 0.80 [0.73, 0.88] | 0.60 [0.38, 0.81] | **1.00 [1.00, 1.00]** | 0.53 [0.41, 0.67] | **0.99 [0.97, 1.00]** |
| **Ensemble (Max-Rank)** | 0.925 [0.907, 0.941] | 0.843 [0.796, 0.883] | 0.759 [0.726, 0.793] | 0.78 [0.72, 0.84] | 0.78 [0.64, 0.89] | 0.99 [0.98, 1.00] | 0.33 [0.26, 0.40] | 0.91 [0.83, 0.98] |
| **Ensemble (Tuned 0.8/0.2)** | **0.969 [0.964, 0.975]** | **0.919 [0.898, 0.937]** | **0.871 [0.852, 0.893]** | 0.86 [0.81, 0.91] | 0.89 [0.79, 0.96] | **0.97 [0.93, 1.00]** | 0.66 [0.61, 0.71] | **0.98 [0.95, 1.00]** |

---

## 3. Paired Bootstrap Confidence Intervals of the Difference (Tuned Fusion vs. Fixed IF)

To rigorously test whether the Tuned Weighted Ensemble ($0.8 \times \text{rank}_{\text{IF}} + 0.2 \times \text{rank}_{\text{TF}}$) statistically significantly outperforms standalone Fixed Isolation Forest, paired differences $\Delta = \text{Tuned} - \text{Fixed IF}$ were evaluated across all 10 evaluation seeds using 1,000 bootstrap resamples:

| Metric | Paired Mean Difference ($\Delta$) | 95% Bootstrap CI | Statistical Significance |
| :--- | :---: | :---: | :--- |
| **$\Delta$ AUROC** | **+0.0031** | **[+0.0019, +0.0044]** | **Statistically Significant ($p < 0.05$)** |
| **$\Delta$ Average Precision (AP)** | **+0.0195** | **[+0.0115, +0.0280]** | **Statistically Significant ($p < 0.05$)** |
| **$\Delta$ Recall@K** | **+0.0080** | **[-0.0020, +0.0200]** | **Overlap with zero ($p \ge 0.05$)** |

Because zero lies strictly outside the 95% bootstrap confidence intervals for AUROC and AP, Tuned Ensemble demonstrates a statistically significant superiority in ranking quality and precision over standalone tabular Isolation Forest.

---

## 4. Deep-Dive Audits & Validation Findings

### 4.1 Alternating Deposit/Withdraw Delay Variant Analysis (15m, 1h, 6h, 24h)
Alternating deposit/withdraw synthetic anomalies were injected across four distinct delay variants to measure detection persistence across turnaround intervals:

| Delay Variant | Transformer Recall [95% CI] | Fixed IF Recall [95% CI] | Tuned Ensemble Recall [95% CI] |
| :--- | :---: | :---: | :---: |
| **15 minutes (2–15m)** | **1.00 [1.00, 1.00]** | 0.82 [0.68, 0.92] | 0.96 [0.88, 1.00] |
| **1 hour (50–70m)** | **1.00 [1.00, 1.00]** | 0.72 [0.60, 0.82] | 0.96 [0.90, 1.00] |
| **6 hours (5–7h)** | **1.00 [1.00, 1.00]** | 0.90 [0.82, 0.98] | 0.98 [0.94, 1.00] |
| **24 hours (22–26h)** | **1.00 [1.00, 1.00]** | 0.88 [0.74, 0.98] | **1.00 [1.00, 1.00]** |

* **Insight**: The Sequence Transformer achieves **1.00 recall across all delay intervals from 15 minutes to 24 hours**. While tabular IF with direction features captures 72–90%, the Transformer's learned conditional event model reliably flags matched-amount reversals across all delay regimes.

### 4.2 Network Feature Ablation: Proof of Negative Result
To verify whether the heuristic flow graph adds value or noise, an empirical benchmark ablation was conducted comparing Fixed IF without network features (default) against Fixed IF with network features:

| Configuration | AUROC [95% CI] | AP [95% CI] | Recall@K [95% CI] |
| :--- | :---: | :---: | :---: |
| **Without Network Features (Default)** | **0.966 [0.961, 0.972]** | **0.899 [0.874, 0.921]** | **0.863 [0.850, 0.875]** |
| **With Network Features (Ablation)** | 0.952 [0.945, 0.960] | 0.855 [0.823, 0.887] | 0.831 [0.817, 0.845] |
| **Paired Difference (With - Without)** | **-0.0143 [-0.0190, -0.0100]** | **-0.0439 [-0.0658, -0.0284]** | **-0.0320 [-0.0410, -0.0220]** |

* **Conclusion**: Adding network features degrades AUROC by -0.0143, AP by -0.0439, and Recall@K by -0.0320 with statistical significance. Combined with permutation testing (shuffled null models produce 154–183 edges vs. real 137), the heuristic graph is formally reclassified as a **Negative Result** and excluded from scoring by default.

### 4.3 Bot-Periodic Feature Engineering Impact
Introducing three targeted behavioral features:
1. **CV of inter-event times** ($\sigma / \mu$, approaching 0 for constant-rate bots)
2. **Modal-amount share** ($\text{count}(\text{mode}) / \text{total\_txs}$, equal to 1.0 for fixed-amount bots)
3. **Lag-1 autocorrelation** of $\Delta t$

* **Impact on Detection**: Fixed IF bot-periodic recall jumped from **0.50** to **0.71 [0.65, 0.76]** (+21 percentage points).

### 4.4 Tabular Direction Features & State on Matching Transformer
Three tabular direction features were added to `extract_behavioral_features`:
1. `direction_switch_count`: count of consecutive event pairs where $t_i \ne t_{i+1}$.
2. `alternation_rate`: $\frac{\text{direction\_switch\_count}}{N - 1}$.
3. `max_alternating_run_length`: longest contiguous sub-sequence of alternating transitions.

* **Does Fixed IF now match the Transformer?**
  **Plain Answer: No, Fixed IF does not fully match the Transformer.**
  While tabular direction features dramatically raised Fixed IF pass-through recall from **0.50** to **0.830 [0.760, 0.890]**, the Sequence Transformer maintains a perfect **1.000 [1.000, 1.000]** recall across all 10 seeds and all delay intervals. However, the Tuned Ensemble (0.8/0.2) achieves **0.975 [0.935, 1.000]**, capturing the Transformer's sensitivity while boosting overall AUROC (0.969) and AP (0.919).

### 4.5 Validation Controls: Length-Control & Amount Perturbation
All validation experiment numbers below are recorded in `output/validation_controls.json`:
1. **Length-Control Distribution**:
   - Real address transaction counts: median 3 txs, 75th percentile 8 txs, max 960 (for addresses with $\ge 3$ txs: median 7, 75th percentile 14).
   - Injected pass-through sequences have 12–22 events.
   - When injected sequences are truncated to a short history of **3–6 events**, Fixed IF recall collapses to **0.200 (std 0.089)**, whereas the Sequence Transformer retains **0.910 (std 0.049)** recall.
2. **Amount Perturbation Sensitivity**:
   - Pass-through variants where withdrawal is scaled to $0.5\times, 0.8\times, 1.0\times, 1.5\times$ of deposit:
   - Transformer recall is **1.000** across all 4 ratios (std 0.000). Fixed IF recall ranges from 0.520 to 0.690.
   - *Fund Conservation Caveat*: The Transformer detects directional flip and delta-t interval transitions, and **does not verify fund conservation**.

### 4.6 In-Situ Contamination Test on Real Addresses ($\ge 5$ events)
Instead of evaluating standalone synthetic addresses, $k \in \{1, 2, 3\}$ alternating deposit/withdraw cycles were directly injected into existing transaction sequences of real addresses with $\ge 5$ transactions:
* **$k = 1$ cycle (2 injected transactions)**:
  - Fixed IF Recall: **0.250 (std 0.055)**
  - Transformer Recall: **0.270 (std 0.081)**
* **$k = 2$ cycles (4 injected transactions)**:
  - Fixed IF Recall: **0.300 (std 0.089)**
  - Transformer Recall: **0.400 (std 0.114)**
* **$k = 3$ cycles (6 injected transactions)**:
  - Fixed IF Recall: **0.310 (std 0.049)**
  - Transformer Recall: **0.400 (std 0.105)**

### 4.7 Empirical Temporal Intervals in Real Data
* **Deposit followed by withdrawal within 15 minutes**: **29 addresses (4.49%)**.
  - 11 have $<3$ events (insufficient history).
  - 18 have $\ge 3$ events.
  - Transformer surprise percentile among valid addresses: mean 67.03%, median 69.09%, min 20.97%, max 98.92%.
  - Ensemble risk score: mean 79.43, median 89.28. **13 of 29 addresses (44.83%)** receive ensemble risk score $\ge 90.0$.
* **Inter-event gaps under 60 seconds**: **61 addresses (9.44%)**.

### 4.8 HDBSCAN Clustering on Addresses with $\ge 3$ Events
* When evaluated strictly on addresses with sufficient sequential history ($N = 372$):
  * **HDBSCAN**: Identifies 2 clusters with silhouette = **0.0984** and 26 noise points. Cluster sizes: Cluster 0 contains **339 entities** (91.13%), Cluster 1 contains **7 entities** (1.88%), and **26 points** are noise (-1).
  * **KMeans ($k=2$)**: Achieves silhouette = **0.4180** with balanced clusters (**187** and **185** entities).
* **Correction**: Previous claims that HDBSCAN "partitioned" the dataset were an artifact of clustering zero-padded dummy embeddings of short-history addresses. Evaluated on real embeddings, HDBSCAN groups the vast majority into a single core cluster with a tiny 7-address satellite.
