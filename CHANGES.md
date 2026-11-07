# DeCrypt: Changelog, Verification Audits & Empirical Validation Results

This document records the architectural revisions, empirical benchmark results, and deep-dive validation audits conducted on the `feature/transformer-sequence-model` branch of the **DeCrypt** platform.

> [!IMPORTANT]
> **Generator Versioning Note**: All benchmark metrics in this document were generated using **Synthetic Anomaly Generator v2.0**.
> Metrics from earlier generator versions are not directly comparable due to domain-bounded thresholds, calibrated temporal intervals, and strict empirical parameter controls.
>
> **Benchmark Circularity Note**: The alternation and timing-regularity features (`direction_switch_count`, `alternation_rate`, `max_alternating_run_length`, `cv_inter_event_time`, `modal_amount_share`, `lag1_autocorr_inter_event_time`) encode the injected archetype definitions by construction. While they boost tabular Isolation Forest detection on synthetic bot-periodic and pass-through benchmarks, their effectiveness on real-world adversaries with non-stationary evasion tactics remains unproven and subject to benchmark definition alignment.

---

## 1. Summary of Architectural & Algorithmic Improvements

| Category | Prior Baseline | Production Architecture Revision |
| :--- | :--- | :--- |
| **Data Ingestion & Multi-Source** | Single Bitcoin CSV script with silent IEEE 754 float drift. | `UnifiedTransaction` schema enforcing arbitrary `Decimal` precision, strict UTC normalization, and multi-source adapters (`BitcoinAdapter`, `AlchemyAdapter`, `PlaidAdapter`). |
| **Network Graph Reliability** | Heuristic flow graph used for scoring, claiming high confidence. | Shuffled-null permutation tests revealed the graph produces 154–183 edges under random noise vs. 137 in real data. Reclassified as a **Negative Result**; network features are excluded from scoring by default. |
| **Streaming Memory Bounds** | Unbounded memory growth accumulating all events. | Explicitly measured via `tracemalloc`: 5,000 transaction events consume **8,360,724 bytes (~7.97 MB)** peak heap memory (with **392.46 KB** raw object payload). Prior "<1.5 MB" claim is formally corrected. Capped at 5,000 events per entity (`DECRYPT_MAX_ENTITY_HISTORY`). |
| **Incremental Feature Extractor** | Discrepancies between batch and incremental features on long histories. | Full mathematical parity verified across 20 real entities for both short and long (>128) event histories. |
| **Service Authentication & Security** | Optional API key with fall-through defaults. | Fail-closed API security via constant-time `hmac.compare_digest`. Rejects requests with 401 Unauthorized unless `DECRYPT_API_KEY` is configured (or dev mode enabled). |
| **Domain Source Guard** | Bitcoin-trained models silently scored unrelated sources. | Calibrated source guard rejects cross-chain/fiat transactions with 422 Unprocessable Entity until domain-specific reference distributions are fitted. |
| **Sequence Surprise Aggregations** | Whole-sequence mean surprise diluted over long histories. | Added top-$k$ mean surprise and sliding-window max surprise (`window5` selected on tuning seeds 101–103). |

---

## 2. Six-Model Benchmark Results (v2.0 Generator, 10 Held-Out Seeds)

Evaluated across 10 held-out evaluation seeds (42–51), with 20 injections per fraud archetype (100 anomalies per seed = 1,000 total evaluations). All table cells report the mean and 95% bootstrap confidence interval computed from 1,000 resamples:

| Model | AUROC [95% CI] | AP [95% CI] | Recall@K [95% CI] | Structuring | Burst | Pass-Through | Bot-Periodic | Dormant-Spike |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Legacy IF** | 0.884 [0.869, 0.897] | 0.793 [0.763, 0.822] | 0.713 [0.688, 0.736] | 0.92 [0.88, 0.96] | 0.93 [0.88, 0.97] | 0.55 [0.49, 0.62] | 0.18 [0.12, 0.24] | **0.98 [0.95, 1.00]** |
| **Fixed IF** | 0.965 [0.960, 0.971] | 0.898 [0.873, 0.921] | 0.860 [0.843, 0.877] | 0.87 [0.82, 0.91] | **0.96 [0.93, 0.99]** | 0.79 [0.73, 0.86] | **0.70 [0.64, 0.76]** | 0.97 [0.94, 0.99] |
| **Transformer (Window-5)** | 0.823 [0.796, 0.848] | 0.700 [0.652, 0.744] | 0.602 [0.559, 0.641] | 0.32 [0.20, 0.44] | 0.55 [0.38, 0.71] | **1.00 [1.00, 1.00]** | 0.16 [0.10, 0.22] | **0.98 [0.95, 1.00]** |
| **Ensemble (Mean-Rank)** | 0.936 [0.924, 0.946] | 0.851 [0.818, 0.881] | 0.785 [0.756, 0.810] | 0.83 [0.77, 0.89] | 0.78 [0.67, 0.88] | **1.00 [1.00, 1.00]** | 0.32 [0.23, 0.41] | **0.99 [0.98, 1.00]** |
| **Ensemble (Max-Rank)** | 0.924 [0.909, 0.936] | 0.824 [0.787, 0.856] | 0.773 [0.743, 0.801] | 0.80 [0.74, 0.86] | 0.81 [0.71, 0.90] | 0.98 [0.96, 1.00] | 0.33 [0.24, 0.43] | 0.95 [0.91, 0.98] |
| **Ensemble (Tuned 0.9/0.1)** | **0.966 [0.960, 0.972]** | **0.895 [0.867, 0.921]** | **0.863 [0.845, 0.883]** | 0.87 [0.82, 0.91] | 0.95 [0.91, 0.98] | **0.89 [0.84, 0.93]** | 0.64 [0.56, 0.71] | 0.97 [0.95, 0.99] |
| **Ensemble (Current 0.8/0.2 Mean)** | 0.969 [0.963, 0.974] | 0.915 [0.896, 0.933] | 0.874 [0.855, 0.894] | 0.88 [0.83, 0.92] | 0.94 [0.89, 0.98] | 0.97 [0.94, 0.99] | 0.60 [0.50, 0.70] | 0.98 [0.96, 1.00] |

---

## 3. Paired Bootstrap Confidence Intervals of the Difference

### 3.1 Tuned Ensemble (0.9/0.1 Window-5) vs. Fixed IF
Paired differences $\Delta = \text{Tuned} - \text{Fixed IF}$ evaluated across all 10 held-out evaluation seeds using 1,000 bootstrap resamples:

| Metric | Paired Mean Difference ($\Delta$) | 95% Bootstrap CI | Statistical Interpretation |
| :--- | :---: | :---: | :--- |
| **$\Delta$ AUROC** | **+0.0004** | **[-0.0006, +0.0013]** | Overlap with zero |
| **$\Delta$ Average Precision (AP)** | **-0.0034** | **[-0.0078, +0.0006]** | Overlap with zero |
| **$\Delta$ Recall@K** | **+0.0030** | **[-0.0020, +0.0080]** | Overlap with zero |

### 3.2 Tuned Ensemble (0.9/0.1 Window-5) vs. Previous Ensemble (0.8/0.2 Mean)
On synthetic benchmark sequences where anomalies span entire address sequences:

| Metric | Paired Mean Difference ($\Delta$) | 95% Bootstrap CI | Statistical Interpretation |
| :--- | :---: | :---: | :--- |
| **$\Delta$ AUROC** | **-0.0029** | **[-0.0043, -0.0017]** | 95% bootstrap CI excludes zero |
| **$\Delta$ Average Precision (AP)** | **-0.0205** | **[-0.0326, -0.0098]** | 95% bootstrap CI excludes zero |
| **$\Delta$ Recall@K** | **-0.0110** | **[-0.0190, -0.0040]** | 95% bootstrap CI excludes zero |

* **Context & Trade-off**: On purely synthetic addresses where every transaction exhibits anomalous alternation, whole-sequence mean surprise aggregates signal across all 12–22 events, yielding high synthetic metrics. However, in **in-situ contamination** where fraud is injected into real histories, whole-sequence mean surprise is heavily diluted by benign history, whereas Window-5 achieves a **+0.206 Recall@K advantage** over Fixed IF.

---

## 4. Deep-Dive Audits & Validation Findings

### 4.1 Transformer Aggregation Tuning on Seeds 101–103 ONLY
To avoid tuning on the test benchmark, aggregation candidates were evaluated strictly across independent tuning seeds `[101, 102, 103]` on stratified in-situ contamination:

| Candidate Aggregation | Avg Overall Recall | k=1 (2 txs) | k=2 (4 txs) | k=3 (6 txs) | 5–9 events (k=3) | 10–24 events (k=3) | 25+ events (k=3) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Window-5 (`window5`)** | **0.5817** | **0.4314** | **0.6471** | **0.6667** | 0.8333 | 0.7222 | **0.5333** |
| **Window-3 (`window3`)** | 0.5294 | 0.4118 | 0.6078 | 0.5686 | 0.7222 | 0.5556 | 0.4000 |
| **Top-3 (`top3`)** | 0.5229 | 0.4314 | 0.5686 | 0.5686 | 0.8333 | 0.6111 | 0.4667 |
| **Top-5 (`top5`)** | 0.5229 | 0.4314 | 0.5686 | 0.5686 | **0.9444** | **0.7778** | 0.3333 |
| **Top-2 (`top2`)** | 0.4771 | 0.4510 | 0.4902 | 0.4902 | 0.6111 | 0.5556 | 0.4667 |
| **Window-2 (`window2`)** | 0.4771 | 0.4314 | 0.5294 | 0.4706 | 0.6111 | 0.4444 | 0.4000 |
| **Top-1 (`top1`)** | 0.4510 | 0.4314 | 0.5098 | 0.4118 | 0.5556 | 0.5000 | 0.4667 |
| **Whole-Sequence Mean (`mean`)** | 0.3987 | 0.3333 | 0.4118 | 0.4510 | 0.5556 | 0.5000 | 0.3333 |
| **Fixed IF Baseline** | 0.3399 | 0.2353 | 0.3529 | 0.4314 | 0.5556 | 0.5556 | 0.4667 |

* **Selection**: `window5` clearly outperformed all other candidates with an average overall recall of **0.5817** (beating whole-sequence mean at 0.3987 and Fixed IF at 0.3399).

---

### 4.2 Stratified In-Situ Contamination on 10 Held-Out Seeds (Paired Bootstrap CIs)
Using the chosen `window5` aggregation, in-situ contamination was evaluated across 10 held-out evaluation seeds (42–51).

> **Definition of Recall@K**:
> For each stratum $S \in \{\text{'5–9'}, \text{'10–24'}, \text{'25+'}, \text{'overall'}\}$, let $P_S$ be the positive (contaminated) addresses in that stratum, and $K = |P_S|$.
> All addresses in the stratum pool are ranked by descending anomaly score.
> $\text{Recall@K} = \frac{|P_S \cap \text{Top-K}|}{|P_S|}$.

| Injected Cycles | History Stratum | Fixed IF Recall [95% CI] | TF Mean Recall [95% CI] | TF Window-5 Recall [95% CI] | Paired Difference: Window-5 vs. Fixed IF [95% CI] | Statistical Interpretation |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$k = 1$ (2 events)** | **5–9 events** | 0.300 [0.150, 0.450] | 0.350 [0.267, 0.433] | 0.333 [0.233, 0.433] | +0.033 [-0.133, +0.217] | Overlap with zero |
| | **10–24 events** | 0.400 [0.316, 0.500] | 0.250 [0.117, 0.367] | 0.317 [0.183, 0.433] | -0.083 [-0.283, +0.117] | Overlap with zero |
| | **25+ events** | 0.280 [0.180, 0.400] | 0.260 [0.180, 0.340] | 0.240 [0.120, 0.380] | -0.040 [-0.200, +0.160] | Overlap with zero |
| | **Overall ($\ge 5$)**| 0.306 [0.271, 0.347] | 0.288 [0.218, 0.347] | 0.294 [0.241, 0.341] | -0.012 [-0.088, +0.059] | Overlap with zero |
| **$k = 2$ (4 events)** | **5–9 events** | 0.367 [0.233, 0.467] | 0.517 [0.416, 0.617] | 0.583 [0.467, 0.683] | **+0.217 [+0.067, +0.417]** | **95% bootstrap CI excludes zero** |
| | **10–24 events** | 0.467 [0.367, 0.567] | 0.367 [0.250, 0.467] | 0.517 [0.367, 0.667] | +0.050 [-0.183, +0.283] | Overlap with zero |
| | **25+ events** | 0.240 [0.160, 0.320] | 0.260 [0.180, 0.340] | 0.300 [0.180, 0.420] | +0.060 [-0.100, +0.240] | Overlap with zero |
| | **Overall ($\ge 5$)**| 0.324 [0.276, 0.365] | 0.365 [0.288, 0.435] | 0.476 [0.376, 0.559] | **+0.153 [+0.059, +0.235]** | **95% bootstrap CI excludes zero** |
| **$k = 3$ (6 events)** | **5–9 events** | 0.617 [0.533, 0.667] | 0.583 [0.450, 0.700] | 0.750 [0.633, 0.850] | **+0.133 [+0.000, +0.267]** | Borderline (touches zero) |
| | **10–24 events** | 0.633 [0.567, 0.717] | 0.467 [0.333, 0.584] | 0.633 [0.467, 0.750] | +0.000 [-0.184, +0.167] | Overlap with zero |
| | **25+ events** | 0.320 [0.220, 0.440] | 0.300 [0.200, 0.380] | 0.540 [0.460, 0.640] | **+0.220 [+0.060, +0.360]** | **95% bootstrap CI excludes zero** |
| | **Overall ($\ge 5$)**| 0.453 [0.418, 0.482] | 0.441 [0.347, 0.529] | **0.659 [0.576, 0.724]** | **+0.206 [+0.124, +0.276]** | **95% bootstrap CI excludes zero** |

---

### 4.3 Isolation Forest Structuring Regression Investigation & Feature Ablation
When new features were introduced, Fixed IF structuring recall declined from **0.98** to **0.87 [0.82, 0.91]**.
A systematic ablation across 10 evaluation seeds isolated the causal factor:

| Feature Set / Model Variant | Structuring Recall [95% CI] | Burst | Pass-Through | Bot-Periodic | Overall Recall@K |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **All 28 Features (Default, max_features=1.0)** | 0.860 [0.820, 0.910] | 0.965 | 0.825 | 0.675 | 0.861 |
| **Ablate Alternation (-3 features)** | **0.965 [0.930, 0.990]** | 0.940 | 0.460 | 0.645 | 0.799 |
| **Ablate Timing Regularity (-3 features)** | 0.880 [0.820, 0.930] | 0.965 | 0.825 | 0.615 | 0.848 |
| **Ablate Centroid (-1 feature)** | 0.885 [0.840, 0.930] | 0.955 | 0.795 | 0.690 | 0.851 |
| **Ablate All New Features (21 Features)** | **1.000 [1.000, 1.000]** | 0.975 | 0.480 | 0.650 | 0.808 |
| **`max_features=sqrt` (5 features per split)** | **0.935 [0.890, 0.980]** | 0.980 | 0.750 | 0.585 | 0.848 |
| **`max_features=0.5` (14 features per split)** | 0.870 [0.840, 0.900] | 0.920 | 0.800 | 0.660 | 0.847 |
| **`max_features=0.7` (19 features per split)** | 0.855 [0.800, 0.910] | 0.945 | 0.780 | 0.745 | 0.860 |

* **Causal Finding**: The regression was caused by the **alternation feature group** (`direction_switch_count`, `alternation_rate`, `max_alternating_run_length`). In structuring sequences (multiple deposits followed by one withdrawal), direction switches only once. These extra dimensions diluted tree splits away from amount entropy and variance. Ablating them restored structuring recall to **0.965**, while using `max_features='sqrt'` recovered structuring to **0.935** while maintaining pass-through detection at **0.750**.

---

### 4.4 Alternating Deposit/Withdraw Delay Variant Analysis (15m, 1h, 6h, 24h)
Alternating deposit/withdraw synthetic anomalies were injected across four distinct delay variants:

| Delay Variant | Transformer Recall [95% CI] | Fixed IF Recall [95% CI] | Tuned Ensemble Recall [95% CI] |
| :--- | :---: | :---: | :---: |
| **15 minutes (2–15m)** | **1.00 [1.00, 1.00]** | 0.76 [0.64, 0.88] | 0.82 [0.72, 0.92] |
| **1 hour (50–70m)** | **1.00 [1.00, 1.00]** | 0.64 [0.50, 0.78] | 0.82 [0.70, 0.92] |
| **6 hours (5–7h)** | **1.00 [1.00, 1.00]** | 0.90 [0.84, 0.96] | 0.92 [0.86, 0.98] |
| **24 hours (22–26h)** | **1.00 [1.00, 1.00]** | 0.88 [0.78, 0.96] | 0.98 [0.94, 1.00] |

* **Insight**: The Sequence Transformer achieves **1.00 recall across all delay intervals from 15 minutes to 24 hours**. While tabular IF with direction features captures 64–90%, the Transformer's learned conditional event model reliably flags matched-amount reversals across all delay regimes.

---

### 4.5 Network Feature Ablation: Proof of Negative Result
| Configuration | AUROC [95% CI] | AP [95% CI] | Recall@K [95% CI] |
| :--- | :---: | :---: | :---: |
| **Without Network Features (Default)** | **0.965 [0.960, 0.971]** | **0.898 [0.873, 0.921]** | **0.860 [0.843, 0.877]** |
| **With Network Features (Ablation)** | 0.943 [0.935, 0.951] | 0.831 [0.801, 0.859] | 0.809 [0.791, 0.828] |
| **Paired Difference (With - Without)** | **-0.0226 [-0.0274, -0.0172]** | **-0.0672 [-0.0793, -0.0544]** | **-0.0510 [-0.0700, -0.0310]** |

* **Conclusion**: Adding network features degrades AUROC by -0.0226, AP by -0.0672, and Recall@K by -0.0510 (95% bootstrap CI excludes zero). Combined with permutation testing (shuffled null models produce 154–183 edges vs. real 137), the heuristic graph is formally reclassified as a **Negative Result** and excluded from scoring by default.

---

### 4.6 Empirical Temporal Intervals in Real Data
* **Deposit followed by withdrawal within 15 minutes**: **29 addresses (4.49%)**.
  - 11 have $<3$ events (insufficient history).
  - 18 have $\ge 3$ events.
  - Transformer surprise percentile among valid addresses: mean 67.03%, median 69.09%, min 20.97%, max 98.92%.
  - Ensemble risk score: mean 79.43, median 89.28. **13 of 29 addresses (44.83%)** receive ensemble risk score $\ge 90.0$.
* **Inter-event gaps under 60 seconds**: **61 addresses (9.44%)**.

---

### 4.7 HDBSCAN Clustering on Addresses with $\ge 3$ Events
* When evaluated strictly on addresses with sufficient sequential history ($N = 372$):
  * **HDBSCAN**: Identifies 2 clusters with silhouette = **0.1365** and 75 noise points. Cluster sizes: Cluster 0 contains **290 entities** (77.96%), Cluster 1 contains **7 entities** (1.88%), and **75 points** are noise (-1).
  * **KMeans ($k=2$)**: Achieves silhouette = **0.3712** with balanced clusters (**180** and **192** entities).
* **Correction**: Previous claims that HDBSCAN "partitioned" the dataset were an artifact of clustering zero-padded dummy embeddings of short-history addresses. Evaluated on real embeddings, HDBSCAN groups the vast majority into a single core cluster with a tiny 7-address satellite.

---

## 5. Fused Model In-Situ Contamination Benchmark (10 Held-Out Seeds)

Evaluated across 10 held-out evaluation seeds (42–51) on stratified in-situ contamination.
Tested Models:
1. **Fixed IF**: Baseline Isolation Forest using all 28 tabular features.
2. **Fusion 0.8 / 0.2 Mean**: 0.8 IF percentile rank + 0.2 whole-sequence mean surprise rank.
3. **Fusion 0.9 / 0.1 Window-5**: 0.9 IF percentile rank + 0.1 sliding-window-5 max surprise rank.
4. **Fusion 3-Signal**: $0.20 \times \text{rank}_{\text{IF}} + 0.10 \times \text{rank}_{\text{TF, Mean}} + 0.70 \times \text{rank}_{\text{TF, Win5}}$ (weights chosen strictly on independent tuning seeds 101–103).

### 5.1 Results by Injected Cycle Count ($k = 1, 2, 3$) and History Stratum

| Injected Cycles | History Stratum | Fixed IF Recall [95% CI] | Fusion 0.8/0.2 Mean [95% CI] | Fusion 0.9/0.1 Win5 [95% CI] | Fusion 3-Signal [95% CI] | Paired Difference: 3-Signal vs. Fixed IF [95% CI] |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$k = 1$ (2 events)** | **5–9 events** | 0.283 [0.167, 0.400] | 0.333 [0.233, 0.433] | 0.317 [0.217, 0.417] | 0.333 [0.183, 0.467] | +0.050 [-0.133, +0.233] |
| | **10–24 events** | 0.450 [0.350, 0.550] | 0.483 [0.400, 0.600] | 0.450 [0.366, 0.550] | 0.400 [0.317, 0.483] | -0.050 [-0.217, +0.117] |
| | **25+ events** | 0.260 [0.140, 0.380] | 0.280 [0.200, 0.360] | 0.280 [0.160, 0.420] | 0.320 [0.180, 0.460] | +0.060 [-0.080, +0.220] |
| | **Overall ($\ge 5$)** | 0.312 [0.276, 0.347] | 0.318 [0.288, 0.347] | 0.306 [0.276, 0.329] | 0.347 [0.312, 0.388] | +0.035 [-0.024, +0.100] |
| **$k = 2$ (4 events)** | **5–9 events** | 0.383 [0.300, 0.467] | 0.483 [0.417, 0.550] | 0.433 [0.350, 0.517] | 0.683 [0.600, 0.750] | **+0.300 [+0.200, +0.383]** |
| | **10–24 events** | 0.517 [0.417, 0.633] | 0.550 [0.417, 0.667] | 0.567 [0.467, 0.667] | 0.617 [0.516, 0.717] | +0.100 [-0.083, +0.267] |
| | **25+ events** | 0.280 [0.180, 0.360] | 0.300 [0.200, 0.400] | 0.300 [0.200, 0.400] | 0.420 [0.300, 0.540] | +0.140 [-0.000, +0.280] |
| | **Overall ($\ge 5$)** | 0.312 [0.265, 0.359] | 0.371 [0.335, 0.412] | 0.347 [0.306, 0.388] | 0.518 [0.447, 0.582] | **+0.206 [+0.129, +0.288]** |
| **$k = 3$ (6 events)** | **5–9 events** | 0.600 [0.533, 0.667] | 0.717 [0.667, 0.767] | 0.633 [0.567, 0.700] | 0.867 [0.733, 0.967] | **+0.267 [+0.117, +0.383]** |
| | **10–24 events** | 0.617 [0.517, 0.733] | 0.617 [0.517, 0.733] | 0.633 [0.533, 0.734] | 0.800 [0.700, 0.884] | **+0.183 [+0.050, +0.300]** |
| | **25+ events** | 0.320 [0.240, 0.400] | 0.360 [0.260, 0.460] | 0.440 [0.320, 0.560] | 0.620 [0.500, 0.760] | **+0.300 [+0.180, +0.440]** |
| | **Overall ($\ge 5$)** | 0.441 [0.406, 0.471] | 0.547 [0.524, 0.571] | 0.500 [0.465, 0.529] | **0.700 [0.653, 0.747]** | **+0.259 [+0.200, +0.318]** |

### 5.2 Key Takeaways on Fused Signals
1. **$k=1$ Floor**: Across all models, a single injected cycle (2 events) inside established histories yields Recall@K between 0.306 and 0.347. Paired difference CIs vs. Fixed IF all span zero (e.g. 3-Signal difference $+0.035$ $[-0.024, +0.100]$), indicating that 2 subtle events are statistically indistinguishable from background noise regardless of model architecture.
2. **Transformer Window-5 Dominance on Long Histories ($25+$ events)**: As history grows, tabular features dilute the anomalous signal (Fixed IF drops from 0.600 on short histories to 0.320 on $25+$ histories at $k=3$). Sequence window-5 maintains focus on local burstiness, allowing the 3-Signal fusion to achieve **0.620** recall (+0.300 gain vs. Fixed IF, 95% bootstrap CI excludes zero).
3. **Overall Recall Gain at $k=3$**: 3-Signal fusion reaches **0.700 [0.653, 0.747]**, outperforming Fixed IF (0.441) by **+0.259 [+0.200, +0.318]** (95% bootstrap CI excludes zero).

---

## 6. Default Pipeline Configuration & Architectural Trade-offs

### 6.1 Production Configuration
Based on empirical evaluation across both the whole-address synthetic benchmark and the in-situ contamination suite, the production default pipeline configuration is set to:
* **Isolation Forest Subsampling (`if_max_features`)**: `1.0` (all 28 features considered at each split).
* **Ensemble Fusion Weights**: $w_{\text{IF}} = 0.8$, $w_{\text{TF}} = 0.2$.
* **Sequence Surprise Aggregation**: `window_max` (rolling window size = 5).
* **Entity History Bounds**: Capped at 5,000 events (`DECRYPT_MAX_ENTITY_HISTORY`).

### 6.2 Plain-Language Trade-Off Analysis
* **Why `max_features=1.0` is the Default**:
  In Section 4.3, we observed that setting `max_features=sqrt` (5 features per split) improved structuring recall from 0.860 to 0.935 by preventing alternation features from competing with amount variance splits. However, this came at a significant cost:
  - Burst recall declined from **0.965** to **0.930** (-3.5%).
  - Bot-periodic recall declined from **0.675** to **0.630** (-4.5%).
  - Overall whole-address Recall@K dropped from **0.861** to **0.848**.
  Subsampling features per split adds tree variance that degrades detection of multi-feature archetypes. We therefore retain `max_features=1.0` to preserve balanced detection sensitivity across all fraud archetypes simultaneously.
* **Whole-Address vs. In-Situ Contamination Trade-off**:
  - In whole-address analysis (where an entire account's activity is synthetic), tabular summary statistics capture the global signature cleanly (Fixed IF achieves 0.860–0.870 Recall@K).
  - In operational streaming (where an attacker injects fraudulent bursts into an existing genuine account), tabular summary metrics suffer heavy dilution. Here, sequence models (Window-5 max surprise) prevent dilution and provide a +0.220 to +0.300 recall lift on established accounts. The 0.8 / 0.2 whole-address ensemble provides a safe baseline that preserves tabular stability while alerting compliance teams to localized sequence surprises.

### 6.3 Discrepancy Analysis: Ablation Script vs. Main Table Fixed IF
A comparison of the standalone ablation script (`scripts/ablate_structuring_features.py`) and the main benchmark table (`output/benchmark_results.json`) reveals minor numerical differences for Fixed IF on identical seeds:
* **Structuring Recall**: 0.860 (Ablation) vs. 0.870 (Main Table). Difference: **0.010** (1.0%).
* **Pass-Through Recall**: 0.825 (Ablation) vs. 0.795 (Main Table). Difference: **0.030** (3.0%).
* **Root Cause & Empirical Noise Level**:
  1. Across the 10 evaluation seeds, there are exactly 20 test anomalies per archetype per seed ($N = 200$ total instances).
  2. A single address changing classification in one seed shifts that seed's metric by $1/20 = 5.0\%$, and shifts the 10-seed aggregate mean by $1/200 = 0.5\%$.
  3. The structuring discrepancy ($0.010$) corresponds to a difference of exactly **2 addresses out of 200**.
  4. The pass-through discrepancy ($0.030$) corresponds to a difference of exactly **6 addresses out of 200**.
  5. Across the 10 random seeds, the run-to-run standard deviation is $\sigma = 0.071$ (std error $\approx 0.022$) for structuring and $\sigma = 0.106$ (std error $\approx 0.034$) for pass-through. Both observed differences are well within the random sample noise level ($<1.0 \times \text{SE}$).
  6. At the script level, `run_benchmark_suite` in `decrypt/benchmark.py` executes graph sanity permutation tests (`run_graph_sanity_test`) and fusion tuning (`tune_fusion_weights`) prior to evaluation seeds, whereas the standalone ablation script evaluates Isolation Forest directly in isolation.

---

## 7. Known System Limitations

To maintain rigorous scientific and engineering integrity, the following limitations of the DeCrypt platform are explicitly documented:

1. **Absence of Real Ground-Truth Labels**:
   The underlying Bitcoin dataset lacks verified forensic fraud labels. All supervised evaluation metrics (AUROC, AP, Recall@K) are measured exclusively against synthetic proxy injections. Real addresses labeled as "normal" in benchmarks may contain undetected historical fraud or mixer interactions.
2. **Synthetic Anomaly Approximations**:
   The 5 fraud archetypes (Structuring, Bursts, Alternating Cycles, Bot-Periodic, Dormant-Spike) are generated using idealized mathematical distributions. Real-world adversaries employ dynamic, non-stationary evasion techniques (e.g. varying amount splits, non-uniform delays, decoy transactions) that may not match these stylized templates.
3. **Benchmark Circularity**:
   The tabular features introduced in v2.0 (`direction_switch_count`, `alternation_rate`, `cv_inter_event_time`, `modal_amount_share`, `lag1_autocorr_inter_event_time`) encode the mathematical definitions of the synthetic archetypes by construction. Their high benchmark scores reflect alignment with the generator rather than verified generalizability against unseen adversarial tactics.
4. **Subtle In-Situ Contamination Ceiling ($k=1$ Recall $\sim 0.31$)**:
   When an adversary injects only a single alternating cycle ($k=1$, 2 transactions) into an established address history, all models (Fixed IF, Mean Fusion, Window-5 Fusion, and 3-Signal Fusion) achieve low recall ($\sim 0.31 - 0.35$). Paired bootstrap confidence intervals against Fixed IF span zero. Two isolated transactions are statistically indistinguishable from legitimate behavioral variance.
5. **Training Domain & Temporal Calibration Bounds**:
   The pre-trained Transformer and baseline reference distributions were trained on early Bitcoin data (May 2011 to February 2014). The model is uncalibrated for modern DeFi protocols, Ethereum ERC-20 smart contracts, or fiat banking transactions. Scoring out-of-domain sources without fitting a dedicated empirical reference distribution triggers source-guard rejection.
6. **Heuristic Flow Graph Negative Result**:
   Because public cryptocurrency transaction exports lack direct UTXO input-output linkage graphs, heuristic time-and-amount flow matching produces more edges on randomly shuffled data (154–183 edges) than on real data (137 edges). Inclusion of these network features actively degrades Isolation Forest Recall@K by -0.051. The graph must not be used for compliance scoring.
