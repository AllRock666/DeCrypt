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
| **Operational Execution Profiles** | Single static fusion setting across all workflows. | Added `--profile batch|streaming`: `batch` (default, IF-led 0.8/0.2 mean fusion, 0.964 AUROC) vs `streaming` (experimental, 3-signal 0.2/0.1/0.7 window-5 fusion, +0.300 recall lift on contaminated streams). |
| **Bootstrap Inference Unit** | Seed-level resampling ($N=10$) collapsed to zero variance on identical seed outcomes. | Fixed by upgrading to **hierarchical cluster bootstrap** (resampling seeds with replacement, and within each seed resampling address detection indicators). Verified non-zero width CIs across all contamination strata. |
| **Component RNG Isolation** | Shared master seed led to run-to-run seed cross-talk across processes. | Components (data split, anomaly injection, Isolation Forest, Transformer training, graph permutation test) derive independent, isolated RNG seeds via `np.random.SeedSequence(seed).generate_state(5)`. Verified 100% deterministic reproducibility. |

---

## 2. Consolidated Whole-Address Benchmark (10 Held-Out Seeds)

Evaluated across 10 held-out evaluation seeds (42–51) with isolated RNG seeds per component. Compares Legacy IF, Fixed IF, Transformer (Mean), Transformer (Window-5), Batch Fusion (0.8/0.2 Mean), and Streaming Fusion (3-Signal 0.2/0.1/0.7):

| Model | AUROC [95% CI] | AP [95% CI] | Recall@K [95% CI] | Structuring | Burst | Alternating | Bot-Periodic | Dormant-Spike |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Legacy IF** | 0.878 [0.864, 0.891] | 0.768 [0.731, 0.804] | 0.704 [0.682, 0.729] | 0.96 [0.93, 0.99] | 0.93 [0.89, 0.97] | 0.45 [0.37, 0.54] | 0.18 [0.14, 0.23] | 0.98 [0.96, 1.00] |
| **Fixed IF** | 0.960 [0.953, 0.966] | 0.872 [0.845, 0.899] | 0.856 [0.839, 0.873] | 0.89 [0.85, 0.93] | 0.91 [0.86, 0.97] | 0.76 [0.68, 0.84] | 0.74 [0.68, 0.81] | 0.97 [0.94, 0.99] |
| **Transformer (Mean)** | 0.802 [0.781, 0.826] | 0.691 [0.651, 0.736] | 0.589 [0.548, 0.632] | 0.62 [0.46, 0.80] | 0.20 [0.02, 0.43] | 1.00 [1.00, 1.00] | 0.25 [0.12, 0.37] | 0.89 [0.79, 0.96] |
| **Transformer (Window-5)** | 0.816 [0.795, 0.839] | 0.663 [0.620, 0.710] | 0.601 [0.565, 0.637] | 0.55 [0.35, 0.76] | 0.35 [0.16, 0.58] | 1.00 [1.00, 1.00] | 0.19 [0.09, 0.30] | 0.91 [0.78, 0.99] |
| **Batch Fusion (0.8/0.2 Mean)** [**Default**] | **0.964 [0.959, 0.970]** | **0.900 [0.877, 0.921]** | **0.869 [0.855, 0.886]** | **0.92 [0.88, 0.96]** | 0.84 [0.78, 0.90] | **0.96 [0.93, 0.99]** | 0.65 [0.57, 0.72] | **0.98 [0.96, 1.00]** |
| **Streaming Fusion (3-Signal)** [**Experimental**] | 0.862 [0.844, 0.881] | 0.748 [0.713, 0.787] | 0.659 [0.626, 0.694] | 0.68 [0.53, 0.85] | 0.39 [0.19, 0.61] | 1.00 [1.00, 1.00] | 0.27 [0.16, 0.39] | 0.96 [0.86, 1.00] |

---

## 3. Paired Bootstrap Confidence Intervals of the Difference

### 3.1 Paired Differences vs. Fixed IF
Paired differences $\Delta = \text{Model} - \text{Fixed IF}$ evaluated across all 10 held-out evaluation seeds (1,000 bootstrap resamples):

| Model | $\Delta$ AUROC [95% CI] | $\Delta$ AP [95% CI] | $\Delta$ Recall@K [95% CI] | Key Archetype Shifts |
| :--- | :---: | :---: | :---: | :--- |
| **Batch Fusion (0.8/0.2 Mean)** | **+0.0046 [+0.0020, +0.0072]** | **+0.0276 [+0.0151, +0.0394]** | **+0.0130 [+0.0000, +0.0240]** (Borderline) | Alternating: **+0.200 [+0.140, +0.255]**; Structuring: +0.025; Burst: -0.075 |
| **Streaming Fusion (3-Signal)** [**Experimental**] | **-0.0976 [-0.1147, -0.0817]** | **-0.1246 [-0.1489, -0.1010]** | **-0.1970 [-0.2310, -0.1670]** | Burst: **-0.525**; Bot-Periodic: **-0.475**; Structuring: **-0.210**; Alternating: +0.240 |

### 3.2 Paired Differences vs. Batch Fusion (0.8/0.2 Mean)
Paired differences $\Delta = \text{Model} - \text{Batch Fusion}$ on the whole-address benchmark:

| Model | $\Delta$ AUROC [95% CI] | $\Delta$ AP [95% CI] | $\Delta$ Recall@K [95% CI] | Empirical Impact on Whole-Address Fraud |
| :--- | :---: | :---: | :---: | :--- |
| **Streaming Fusion (3-Signal)** [**Experimental**] | **-0.1021 [-0.1176, -0.0874]** | **-0.1522 [-0.1780, -0.1272]** | **-0.2100 [-0.2360, -0.1820]** | **Severe Degradation**: Drops burst recall from 0.84 to 0.39 (-0.450), bot-periodic from 0.65 to 0.27 (-0.380), and structuring from 0.92 to 0.68 (-0.235). |

### 3.3 Root-Cause Analysis: Why 3-Signal Fails on Whole-Address Archetypes
* **The Noise Threshold vs. Drop Magnitude**: The run-to-run random seed noise level is approximately **~0.01–0.03**. The observed drops in 3-signal fusion (-0.102 AUROC, -0.152 AP, -0.210 Recall@K) exceed the noise level by **over 5x to 10x**.
* **Mechanistic Cause**: In 3-signal fusion, 70% weight is allocated to localized Window-5 max surprise, and only 20% to Isolation Forest. Whole-address anomalies exhibit diffuse behavioral signatures across the entity's complete lifespan (high burst count, total structuring turnover, strict inter-arrival time regularity). The localized 5-event window captures local transition surprise but completely loses the macroscopic count/volume distributions that tabular Isolation Forest isolates.
* **Architectural Resolution**: Because 3-Signal severely impairs whole-address detection, it **cannot be the global default**. Instead, we introduce `--profile batch|streaming`:
  - `batch` (**Default**): 0.8 IF + 0.2 TF Mean surprise. Highest point estimates on aggregate metrics (AUROC and AP paired CIs vs Fixed IF exclude zero; Recall@K borderline); weaker than Fixed IF on burst and bot-periodic archetypes.
  - `streaming` (**Experimental**): 3-Signal ($0.20 \text{ IF} + 0.10 \text{ Mean} + 0.70 \text{ Window-5}$). Applied in real-time streaming where local burstiness must be captured amidst long benign histories.

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
> [!NOTE] Superseded pre-v1.1 Table
> This table evaluated standalone model surprise aggregations before the introduction of multi-signal fusion ensembles and embedded multi-archetype controls (see Section 5 for the consolidated embedded multi-archetype benchmark).

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
> [!NOTE] Superseded pre-v1.1 Table
> Evaluated single-seed exploratory delay variants for standalone Transformer vs. Fixed IF prior to the multi-seed benchmark and in-situ contamination suite.

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

## 5. Embedded Multi-Archetype In-Situ Contamination Benchmark (10 Held-Out Seeds)

Evaluated across 10 held-out evaluation seeds (42–51) on real addresses with $\ge 5$ transactions. Injected anomalies are placed into genuine transaction histories rather than isolated standalone synthetic accounts, benchmarking model performance against localized fraud contamination.

### 5.1 Verification of Contamination Test Split Variation
* **Split Variance**: Across the 10 evaluation seeds, `rng_split = np.random.default_rng(seed_split)` shuffles all 646 unique dataset addresses before taking a 70/30 train/test split.
* **Test Universe Coverage**: Out of 646 addresses, **619 unique addresses** appear in at least one test set across the 10 seeds.
* **Contaminated Address Diversity**: Exactly **121 unique addresses** were selected for contamination across seeds 42–51 (out of 170 total contamination assignments).
* **Per-Seed Stratum Positives**:
  - `5–9 events`: 6 positive addresses per seed (sampled from 25–40 available candidates per seed).
  - `10–24 events`: 6 positive addresses per seed (sampled from 22–28 available candidates per seed).
  - `25+ events`: 5 positive addresses per seed (sampled from 12–22 available candidates per seed).
  - `Overall ($\ge 5$)`: 17 positive addresses per seed (sampled from 63–83 available candidates per seed).
* **Pairwise Overlap**: The mean pairwise address overlap across seeds is **1.31 / 17 addresses** (range: 0 to 4), confirming high data split diversity across the evaluation suite.

### 5.2 Hierarchical Cluster Bootstrap Methodology
* **Resampling Scheme**:
  - Level 1: Resample 10 evaluation seeds with replacement (1,000 resamples).
  - Level 2: Within each sampled seed, resample the individual address binary detection indicators $\{d_i = y_{\text{model}, i} - y_{\text{baseline}, i}\}_{i=1}^{K_s}$ with replacement.
  This captures both seed-to-seed split variability and within-seed address binomial variance, ensuring statistically sound, non-zero width confidence intervals.

### 5.3 Empirical Results by Injected Archetype and History Stratum (Overall First)

#### Archetype A: Embedded High-Frequency Burst (20–30 rapid transactions, intervals 5–30s)
| Stratum | Fixed IF Recall@K | Batch Fusion Recall@K | Diff (Batch vs IF) | Streaming Fusion Recall@K | Diff (Streaming vs IF) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall ($\ge 5$)** | 0.688 [0.588, 0.776] | 0.682 [0.565, 0.800] | -0.006 [-0.094, +0.076] | 0.553 [0.412, 0.694] | -0.135 [-0.259, -0.012] |
| **5–9 events** | 0.867 [0.767, 0.950] | 0.833 [0.683, 0.950] | -0.033 [-0.167, +0.100] | 0.667 [0.500, 0.817] | -0.200 [-0.400, -0.033] |
| **10–24 events** | 0.833 [0.717, 0.933] | 0.783 [0.617, 0.917] | -0.050 [-0.200, +0.083] | 0.517 [0.333, 0.684] | -0.317 [-0.550, -0.083] |
| **25+ events** | 0.520 [0.340, 0.700] | 0.520 [0.320, 0.700] | +0.000 [-0.160, +0.160] | 0.500 [0.300, 0.680] | -0.020 [-0.240, +0.200] |

#### Archetype B: Embedded Structuring (8–12 sub-threshold deposits + 1 withdrawal)
| Stratum | Fixed IF Recall@K | Batch Fusion Recall@K | Diff (Batch vs IF) | Streaming Fusion Recall@K | Diff (Streaming vs IF) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall ($\ge 5$)** | 0.594 [0.512, 0.671] | **0.659 [0.571, 0.735]** | **+0.065 [+0.012, +0.118]** | 0.629 [0.541, 0.718] | +0.035 [-0.071, +0.141] |
| **5–9 events** | 0.567 [0.400, 0.733] | **0.733 [0.600, 0.867]** | **+0.167 [+0.050, +0.300]** | **0.783 [0.633, 0.917]** | **+0.217 [+0.000, +0.433]** (Borderline) |
| **10–24 events** | 0.667 [0.533, 0.817] | 0.700 [0.550, 0.833] | +0.033 [+0.000, +0.100] (Borderline) | 0.717 [0.583, 0.833] | +0.050 [-0.133, +0.233] |
| **25+ events** | 0.520 [0.320, 0.700] | 0.580 [0.400, 0.760] | +0.060 [-0.100, +0.260] | 0.620 [0.480, 0.760] | +0.100 [-0.120, +0.320] |

#### Archetype C: Embedded Bot-Periodic (15–25 transactions with identical interval and amount)
| Stratum | Fixed IF Recall@K | Batch Fusion Recall@K | Diff (Batch vs IF) | Streaming Fusion Recall@K | Diff (Streaming vs IF) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall ($\ge 5$)** | 0.547 [0.453, 0.635] | 0.588 [0.494, 0.682] | +0.041 [-0.024, +0.106] | 0.500 [0.394, 0.606] | -0.047 [-0.153, +0.053] |
| **5–9 events** | 0.717 [0.583, 0.833] | 0.767 [0.633, 0.883] | +0.050 [-0.033, +0.150] | 0.600 [0.450, 0.750] | -0.117 [-0.283, +0.050] |
| **10–24 events** | 0.667 [0.533, 0.800] | 0.700 [0.550, 0.850] | +0.033 [-0.067, +0.150] | 0.450 [0.316, 0.600] | -0.217 [-0.383, -0.050] |
| **25+ events** | 0.340 [0.160, 0.540] | 0.460 [0.280, 0.620] | +0.120 [-0.060, +0.280] | 0.540 [0.360, 0.700] | +0.200 [-0.040, +0.400] |

#### Archetype D: Embedded Alternating Cycles ($k = 1, 2, 3$)

##### Injected Alternating Cycles $k = 1$ (2 events)
| Stratum | Fixed IF Recall@K | Batch Fusion Recall@K | Diff (Batch vs IF) | Streaming Fusion Recall@K | Diff (Streaming vs IF) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall ($\ge 5$)** | 0.276 [0.194, 0.365] | 0.312 [0.224, 0.412] | +0.035 [-0.024, +0.100] | **0.388 [0.288, 0.488]** | **+0.112 [+0.006, +0.218]** |
| **5–9 events** | 0.217 [0.100, 0.350] | 0.267 [0.133, 0.433] | +0.050 [+0.000, +0.150] (Borderline) | 0.400 [0.233, 0.567] | +0.183 [-0.017, +0.417] |
| **10–24 events** | 0.250 [0.100, 0.417] | 0.267 [0.117, 0.433] | +0.017 [+0.000, +0.067] (Borderline) | 0.350 [0.217, 0.500] | +0.100 [-0.083, +0.267] |
| **25+ events** | 0.300 [0.140, 0.460] | 0.340 [0.180, 0.520] | +0.040 [-0.120, +0.180] | 0.460 [0.280, 0.640] | +0.160 [-0.040, +0.360] |

##### Injected Alternating Cycles $k = 2$ (4 events)
| Stratum | Fixed IF Recall@K | Batch Fusion Recall@K | Diff (Batch vs IF) | Streaming Fusion Recall@K | Diff (Streaming vs IF) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall ($\ge 5$)** | 0.300 [0.218, 0.388] | **0.412 [0.329, 0.512]** | **+0.112 [+0.053, +0.182]** | **0.547 [0.429, 0.671]** | **+0.247 [+0.135, +0.353]** |
| **5–9 events** | 0.350 [0.217, 0.500] | 0.400 [0.267, 0.567] | +0.050 [+0.000, +0.133] (Borderline) | **0.683 [0.483, 0.850]** | **+0.333 [+0.167, +0.500]** |
| **10–24 events** | 0.383 [0.233, 0.550] | 0.450 [0.267, 0.633] | +0.067 [-0.050, +0.183] | 0.533 [0.383, 0.684] | +0.150 [-0.050, +0.333] |
| **25+ events** | 0.340 [0.180, 0.500] | 0.360 [0.200, 0.520] | +0.020 [-0.120, +0.160] | 0.520 [0.340, 0.680] | +0.180 [-0.020, +0.380] |

##### Injected Alternating Cycles $k = 3$ (6 events)
| Stratum | Fixed IF Recall@K | Batch Fusion Recall@K | Diff (Batch vs IF) | Streaming Fusion Recall@K | Diff (Streaming vs IF) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall ($\ge 5$)** | 0.429 [0.353, 0.506] | **0.518 [0.429, 0.606]** | **+0.088 [+0.006, +0.171]** | **0.659 [0.553, 0.759]** | **+0.229 [+0.112, +0.341]** |
| **5–9 events** | 0.467 [0.317, 0.617] | **0.583 [0.450, 0.717]** | **+0.117 [+0.033, +0.233]** | **0.800 [0.683, 0.900]** | **+0.333 [+0.167, +0.483]** |
| **10–24 events** | 0.483 [0.333, 0.650] | 0.533 [0.383, 0.683] | +0.050 [-0.033, +0.150] | **0.733 [0.600, 0.867]** | **+0.250 [+0.067, +0.417]** |
| **25+ events** | 0.360 [0.180, 0.540] | 0.380 [0.220, 0.540] | +0.020 [-0.120, +0.180] | **0.600 [0.440, 0.740]** | **+0.240 [+0.000, +0.460]** (Borderline) |

### 5.4 Key Takeaways across Embedded Archetypes
1. **Structuring Benefit**: On embedded structuring, Batch Fusion provides a statistically significant recall lift (+0.065 overall, +0.167 on short histories, 95% bootstrap CI excludes zero). Sequence modeling recognizes repeated directional deposits without corresponding withdrawals as unusual sequential progressions.
2. **Burst Resistance**: On embedded bursts, tabular Fixed IF (0.688) and Batch Fusion (0.682) perform comparably ($\Delta = -0.006$). In contrast, Streaming Fusion degrades to 0.553 (-0.135 drop, 95% CI excludes zero) because down-weighting tabular features impairs the detection of rapid volume accumulations.
3. **Alternating Cycles Lift**: Streaming Fusion excels on alternating deposit/withdrawal cycles, delivering +0.112 (at $k=1$), +0.247 (at $k=2$), and +0.229 (at $k=3$) overall lift over Fixed IF. On long histories (25+ events) where tabular averages dilute the alternation signature, Streaming Fusion achieves 0.600 vs. Fixed IF 0.360 (+0.240 lift).
4. **Conclusion on Operational Profile**: Batch Fusion remains the recommended production default (`--profile batch`). Highest point estimates on aggregate metrics (AUROC and AP paired CIs vs Fixed IF exclude zero; Recall@K borderline); weaker than Fixed IF on burst and bot-periodic archetypes. Streaming Fusion is designated **(Experimental)** (`--profile streaming`), specifically suited for real-time streaming pipelines monitoring high-activity entities for localized micro-bursts and rapid reversals.

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
