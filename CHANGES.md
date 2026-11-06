# DeCrypt: Architecture & Model Evolution (CHANGES.md)

This document tracks verified changes, architectural fixes, and empirical benchmark evaluations conducted on the Bitcoin transaction dataset (`data/bitcoin_transactions.csv`).

---

## 1. Before vs. After Comparison Table

All metrics below are derived directly from real executions on `data/bitcoin_transactions.csv` (8,388 transactions, 646 unique addresses):

| Aspect | Legacy Baseline (`--legacy-features`) | Upgraded Architecture (Phase 1–4 + Validation) |
| :--- | :--- | :--- |
| **Transaction Volume** | Net flow (`abs(sum(amount))`). Deposit 100 + withdraw 100 resulted in `0.0`. | True volume (`sum(abs(amount))`). Deposit 100 + withdraw 100 correctly yields `200.0`. |
| **Transaction Features** | 15 features. Groupby with slow Python lambdas. | 23 vectorized behavioral & temporal features (includes `net_flow`, `pass_through_ratio`, Goh-Barabási `burstiness`, Shannon `amount_entropy`). |
| **Transaction Graph** | Grouped by hash looking for dual withdraw/deposit. Produced **0 nodes, 0 edges**. | **Heuristic flow graph** linking withdrawals to deposits within 24h & 2% amount tolerance: **137 nodes, 137 edges** (classified **Low-Confidence** via null permutation audit). |
| **Network Feature Handling** | Network columns had empty object dtype; silently dropped by `select_dtypes(['int64','float64'])`. | Explicit numeric typing via `select_dtypes('number')`. If edges < `min_edges` (10), network features are skipped with an explicit warning; zero-variance columns logged. |
| **Recency Calculation** | Measured from dataset start (`first_tx - min_time`), distorting inactivity. | `days_until_end = (dataset_end - last_tx)`. |
| **Cluster Selection** | Naive elbow heuristic defaulting to the cap ($k=10$, inertia diffs). | Dynamic silhouette maximization ($k=2..10$). Optimal $k=2$ (tabular silhouette = 0.718). Flagged: Cluster 1 contains 98.1% of addresses (>90% single cluster dominance). |
| **Anomaly Scoring** | Arbitrary affine transformation `((1 - raw) * 50).clip(0, 100)`. | Percentile-rank calibrated risk score normalized strictly across `[0.0, 100.0]`. |
| **Sequence Modeling** | None. | **BERT-style Masked Event Transformer** (64-dim, 4 heads, 2 layers, 15% masking). Anomaly score = leave-one-out event NLL surprise. |
| **Explainability** | None. Only an overall scalar score. | Top-3 most surprising events per address stored with timestamp, amount, direction, and NLL. |
| **Embedding Clustering**| None. | Mean-pooled sequence embeddings clustered via **HDBSCAN** (PCA-16) and KMeans. HDBSCAN achieves silhouette 0.447 with balanced partitions (372 vs 274 addresses). |
| **Ensemble Scoring** | None. | Multi-strategy rank calibration: Mean-Rank (0.5/0.5), Max-Rank ($\max(\text{rank}_{\text{IF}}, \text{rank}_{\text{TF}})$), and Tuned Weighted Fusion ($0.8 \times \text{rank}_{\text{IF}} + 0.2 \times \text{rank}_{\text{TF}}$). |
| **Logging** | Duplicate handler output (every log line printed twice). | Single clean stream logger (`propagate=False`). |
| **Package Management** | Traditional pip. | Fast package management via `uv`. |

---

## 2. Empirical Benchmark Evaluation (10 Seeds, 20/archetype = 100 Anomalies/Seed, 95% Bootstrap CI)

Evaluated across 10 independent test splits (Seeds 42 through 51) with 20 synthetic anomalies injected per archetype (100 anomalies per seed; 1,000 total evaluations). Each cell displays the empirical mean and [95% bootstrap confidence interval] computed over 1,000 bootstrap resamples:

| Model | AUROC [95% CI] | AP [95% CI] | Recall@K [95% CI] | Structuring | Burst | Pass-Through | Bot-Periodic | Dormant-Spike |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Legacy IF** | 0.884 [0.869, 0.898] | 0.793 [0.763, 0.822] | 0.712 [0.687, 0.735] | 0.92 [0.87, 0.96] | 0.93 [0.89, 0.97] | 0.55 [0.45, 0.65] | 0.18 [0.12, 0.25] | 0.98 [0.96, 0.99] |
| **Fixed IF** | 0.920 [0.908, 0.930] | 0.812 [0.778, 0.843] | 0.769 [0.749, 0.787] | **0.96 [0.94, 0.98]** | **0.96 [0.92, 0.98]** | 0.46 [0.36, 0.57] | **0.50 [0.42, 0.58]** | 0.97 [0.94, 0.99] |
| **Transformer** | 0.812 [0.792, 0.831] | 0.716 [0.681, 0.750] | 0.610 [0.586, 0.635] | 0.29 [0.11, 0.48] | 0.48 [0.23, 0.71] | **0.99 [0.98, 1.00]** | 0.33 [0.20, 0.48] | 0.96 [0.91, 1.00] |
| **Ensemble (Mean-Rank)** | 0.924 [0.917, 0.932] | **0.848 [0.829, 0.868]** | 0.752 [0.732, 0.773] | 0.84 [0.76, 0.93] | 0.61 [0.37, 0.85] | 0.91 [0.86, 0.95] | 0.40 [0.27, 0.54] | **0.99 [0.98, 1.00]** |
| **Ensemble (Max-Rank)** | 0.910 [0.900, 0.922] | 0.811 [0.780, 0.844] | 0.747 [0.725, 0.772] | 0.83 [0.76, 0.90] | 0.79 [0.67, 0.89] | 0.96 [0.93, 0.98] | 0.22 [0.15, 0.29] | 0.94 [0.90, 0.98] |
| **Ensemble (Tuned 0.8/0.2)** | **0.933 [0.925, 0.941]** | 0.842 [0.815, 0.869] | **0.786 [0.769, 0.802]** | **0.96 [0.94, 0.98]** | 0.91 [0.86, 0.96] | 0.59 [0.52, 0.67] | 0.47 [0.41, 0.55] | 0.98 [0.97, 1.00] |

---

## 3. Deep-Dive Validation & Diagnostic Audits

### 3.1 Explanation of Address Count with <3 Events
- **Dataset Ground Truth**: The raw CSV contains 646 unique addresses. A strict count shows:
  - **143 addresses** with exactly 1 transaction
  - **131 addresses** with exactly 2 transactions
  - **Total < 3 transactions**: $143 + 131 = \mathbf{274}$ addresses (42.41% of dataset).
  - **Total $\ge$ 3 transactions**: $646 - 274 = \mathbf{372}$ addresses (57.59% of dataset).
- **Origin of the "311" Count**: During initial model training, the sequence transformer used a 90/10 train/validation split on the 372 sequence-eligible addresses ($372 \times 0.90 \approx 335$ training addresses, 37 validation addresses). A prior report erroneously calculated $646 - 335 = 311$ by subtracting training addresses from total addresses and omitting the 37 validation addresses ($335 + 37 = 372$). The true count of short-history addresses in the dataset is exactly **274**. All 274 addresses are correctly tagged with `insufficient_history=True` in `risk_scores.csv` and evaluated solely via Isolation Forest.

### 3.2 Heuristic Flow Graph Sanity & Permutation Test
- **Observed Graph**: 137 nodes, 137 edges (linking withdrawals to deposits within 24 hours with matching amount $\pm 2\%$).
- **Null Permutation Models**:
  - **Timestamp Shuffling (20 runs)**: Produced a mean of **183.45 edges** ($\text{std} = 13.09$, $\min = 163$, $\max = 207$).
  - **Amount Shuffling (20 runs)**: Produced a mean of **154.45 edges** ($\text{std} = 10.16$, $\min = 132$, $\max = 171$).
- **Finding & Classification**: The empirical graph edge count (137) is lower than the mean of both permutation null models. Shuffling timestamps breaks temporal clustering and mixes transactions uniformly across the 3-year observation window, paradoxically creating *more* coincidental matches within 24-hour windows. Because the real graph is not clearly above the shuffled distribution, the graph is formally classified and labeled as **Low-Confidence** in both the PDF report and documentation.

### 3.3 Benchmark Archetype Audit & Pass-Through Investigation
- **Empirical Distribution Range Alignment**:
  - **Amounts**: Real transactions span $[0.00, 4579.29]$ with median $1.00$. In early test drafts, structuring cumulative deposits reached 7,003 BTC (outside empirical range). The structuring threshold was capped at $250.0$ BTC, guaranteeing cumulative withdrawals remain $\le 4,000.0 \le 4,579.29$ BTC.
  - **Inter-Event Times**: Real $\Delta t$ spans $[0, 8.5 \times 10^7]$ seconds (~987 days), median $11,400$s (3.17h). All injected archetypes (burst 5–30s, pass-through 2–15m, dormant 150–250 days) fall strictly within empirical bounds.
- **Pass-Through Recall Analysis (Why Fixed IF recalls 0.46 vs Transformer 0.99)**:
  - Tabular `pass_through_ratio` distribution in the real dataset: **121 real addresses (18.7%)** exhibit `pass_through_ratio > 0.80`, and **70 addresses (10.8%)** exhibit `pass_through_ratio > 0.95`. Normal relay/transit wallets frequently pass through 100% of deposited funds.
  - Because high `pass_through_ratio` is commonplace among benign addresses in the training set, Isolation Forest treats high pass-through as a normal behavioral state when transaction frequency and volume are unexceptional.
  - In contrast, the **Sequence Transformer** evaluates the strict conditional transition probability: an exact deposit followed within minutes by an equivalent withdrawal ($X \to 0.98X$) generates extreme cross-entropy surprise in direction, $\Delta t$ bin, and amount bin simultaneously, achieving **0.99 [0.98, 1.00]** recall.

### 3.4 Fusion Strategy Alternatives
- **Weight Tuning Protocol**: Grid search was executed on 3 dedicated tuning seeds (`101, 102, 103`) strictly separate from benchmark evaluation seeds (`42..51`). A weighting of $w_{\text{IF}} = 0.8, w_{\text{TF}} = 0.2$ proved optimal on the tuning set.
- **Out-of-Sample Performance Comparison**:
  - **Mean-Rank (0.5/0.5)**: Maximizes Average Precision (**0.848 [0.829, 0.868]**) and delivers strong balanced recall across both temporal (0.91 pass-through) and tabular anomalies (0.84 structuring, 0.99 dormant-spike).
  - **Max-Rank ($\max(\text{rank}_{\text{IF}}, \text{rank}_{\text{TF}})$)**: Yields AUROC 0.910 [0.900, 0.922] and AP 0.811 [0.780, 0.844]; excels at pass-through (0.96) and bursts (0.79), but exhibits higher false positives on normal active addresses.
  - **Tuned Weighted Fusion (0.8/0.2)**: Achieves the highest overall AUROC (**0.933 [0.925, 0.941]**) and highest overall Recall@K (**0.786 [0.769, 0.802]**), providing superior tabular detection while leveraging transformer surprise to elevate pass-through recall from 0.46 to 0.59.

### 3.5 Clustering Topology & Cluster Dominance Flags
- **Tabular KMeans ($k=2$, Silhouette = 0.718)**:
  - Cluster 0: 12 addresses (1.86%) — extreme volume outliers.
  - Cluster 1: 634 addresses (98.14%) — all other entities.
  - **Diagnostic Status**: **FLAGGED: >90% single cluster**. Euclidean distance in tabular space is heavily dominated by heavy-tailed volume distributions, lumping nearly the entire dataset into one partition.
- **HDBSCAN on Sequence Embeddings (Silhouette = 0.447)**:
  - Cluster 0: 372 addresses (57.59%) — rich sequence histories ($\ge 3$ transactions).
  - Cluster 1: 274 addresses (42.41%) — short histories ($< 3$ transactions).
  - **Diagnostic Status**: **Balanced (no cluster > 90%)**. HDBSCAN sequence embeddings naturally separate depth of temporal activity without succumbing to volume skew.
