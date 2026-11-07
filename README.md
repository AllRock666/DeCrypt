# DeCrypt: Multi-Source Financial & Cryptocurrency Anomaly Detection Platform

**DeCrypt** is an end-to-end, high-performance financial and cryptocurrency transaction anomaly detection platform. It combines **multi-source real-time streaming ingestion** (Plaid Sandbox banking, Alchemy EVM, Blockstream/Mempool.space Bitcoin, legacy CSVs) with **calibrated machine learning** (vectorized behavioral profiling, Isolation Forest, and a BERT-style Masked Event Transformer).

DeCrypt operates in two complementary modes:
1. **Continuous Real-Time Streaming Service**: A high-throughput FastAPI service with sliding-window rate limiting, HMAC webhook signature verification, SHA-256 idempotency, incremental 23-feature extraction, and leave-one-out sequence surprise scoring.
2. **Forensic Batch Engine**: Full offline dataset analysis generating multi-page PDF dossiers, cross-seed synthetic fraud benchmarks, clustering diagnostics, and CSV score exports.

---

## Architecture & System Overview

For detailed technical designs, refer to the dedicated documentation:
- 📖 [**Architecture Specification (`ARCHITECTURE.md`)**](file:///e:/projects/DeCrypt/DeCrypt/ARCHITECTURE.md)
- 🔌 [**REST API Reference (`API_DOCUMENTATION.md`)**](file:///e:/projects/DeCrypt/DeCrypt/API_DOCUMENTATION.md)
- 🚀 [**Installation & Setup Guide (`SETUP.md`)**](file:///e:/projects/DeCrypt/DeCrypt/SETUP.md)
- 🔍 [**Architecture & Security Audit (`ARCHITECTURE_AUDIT.md`)**](file:///e:/projects/DeCrypt/DeCrypt/ARCHITECTURE_AUDIT.md)

### Core Components

1. **Unified Domain & Canonical Schema (`decrypt.domain.schema`)**:
   - `UnifiedTransaction` schema enforcing arbitrary `Decimal` precision (eliminating IEEE 754 floating-point drift) and strict UTC normalization.
   - Deterministic SHA-256 idempotency key generation across heterogeneous providers.
   - Genuine wallet-level UTXO decomposition for Bitcoin transactions, distinguishing actual net spend/receipt from unrelated multi-input/output change flows.
   - Seamless bidirectional backward compatibility with legacy 5-column transaction CSVs.

2. **Multi-Source Ingestion Layer (`decrypt.providers`)**:
   - **Banking (Plaid Sandbox)**: `/transactions/sync` cursor-based synchronization, modification/removal handling, and webhook verification. Inverts Plaid debit amounts to standard outflows.
   - **EVM Blockchains (Alchemy)**: Native ETH and ERC-20 transfers across Ethereum, Arbitrum, Optimism, Polygon, and Base via `alchemy_getAssetTransfers`. Includes HMAC-SHA256 signature verification for address activity webhooks.
   - **Bitcoin (Esplora / Mempool.space)**: Confirmed on-chain transaction history, unconfirmed mempool tracking, and exact Satoshi-to-BTC fixed-base conversion.

3. **Incremental Real-Time ML Engine (`decrypt.pipeline`)**:
   - **State Management**: `EntityStateManager` maintains an in-memory sliding window of the last 128 events per entity.
   - **Incremental Behavioral Profiling**: `IncrementalFeatureExtractor` computes the exact 23 vectorized features on-the-fly without requiring full-dataset group-by table scans.
   - **Reference Distribution Calibration**: Eliminates $N=1$ rank distortion in real-time inference by scoring entities against frozen empirical CDF reference distributions (`scipy.stats.percentileofscore`).
   - **Sequence Surprise Inference**: Leave-one-out masking on `EventTransformerModel` evaluates negative log-likelihood (NLL) surprise of recent transactions without retraining.
   - **Automated Explainability**: Generates top anomalous behavioral features and identifies the top-3 most surprising sequence events.

4. **Vectorized Behavioral & Temporal Profiling (28 Tabular Features)**:
   - Computes true volume ($\sum |amount|$), net flow ($\sum amount$), pass-through ratios, and deposit/withdrawal distributions.
   - Activity span and normalized entity-relative temporal centroid `activity_centroid` (mean event offset from first transaction normalized by span).
   - Temporal dynamics via **Goh-Barabási burstiness** $B = \frac{\sigma - \mu}{\sigma + \mu}$, **Coefficient of Variation (CV)** of inter-event times, **lag-1 autocorrelation** of $\Delta t$, and Shannon amount entropy.
   - High-precision behavioral features: **modal-amount share** ($\frac{\text{count}(\text{mode})}{\text{total\_txs}}$) targeting automated/bot behavior.
   - Direction transition dynamics: **`direction_switch_count`** (consecutive event pairs with opposite direction), **`alternation_rate`** ($\frac{\text{switches}}{N-1}$), and **`max_alternating_run_length`** (longest run of alternating direction transitions).

5. **Heuristic Flow Graph (Negative Result: Excluded by Default)**:
   - Traces fund movements between withdrawals and deposits occurring within a configurable window (24h, 2% tolerance).
   - **Permutation Sanity Test & Negative Result**: Evaluated against null permutation models (20 timestamp-shuffled runs: mean 183.4 edges; 20 amount-shuffled runs: mean 154.4 edges vs. real 137 edges). Permuting timestamps and amounts yields *equal or more* edges than the real graph.
   - **Ablation Finding**: An ablation benchmark confirms that adding these network features degrades Isolation Forest performance ($\Delta \text{AUROC} = -0.0226$, $\Delta \text{AP} = -0.0672$, $\Delta \text{Recall@K} = -0.0510$, where 95% bootstrap CI excludes zero). Consequently, the heuristic graph is formally reclassified as a **negative result** and network features are excluded from scoring by default (retained behind `--include-network-features`).

6. **Masked Event Transformer (`decrypt.sequence`)**:
   - 2-layer, 4-head Transformer encoder ($d_{\text{model}} = 64$) trained on discrete/continuous transaction sequences via 15% masked event modeling.
   - Generates leave-one-out event **negative log-likelihood (NLL) surprise scores** with score aggregations: whole-sequence mean, top-$k$ mean, and rolling-window max (`window5` selected on tuning seeds 101–103).
   - Provides per-address explainability by reporting the top-3 most surprising transactions.
   - Extracts mean-pooled sequence embeddings for non-linear density clustering via **HDBSCAN** and KMeans on the subset of entities with $\ge 3$ events.
   - **Alternating Deposit/Withdraw Detection**: The Transformer detects rapid alternating deposit/withdraw cycles and unusual inter-arrival timings. *Note: the model evaluates discrete transition surprises and does not verify fund conservation.*

7. **Ensemble Risk Fusion & Operational Profiles (`--profile batch|streaming`)**:
   - Combines Isolation Forest deviance scores with Transformer surprise scores using percentile-rank calibration ($0.0 - 100.0$).
   - **Execution Profiles**:
     - `--profile batch` (**Default**): Optimized for holistic whole-address forensic audits. Uses $w_{\text{IF}} = 0.8, w_{\text{TF}} = 0.2$ with whole-sequence mean surprise. Achieves **0.964 AUROC, 0.900 AP, and 0.869 Recall@K**, providing dominant detection across all 5 fraud archetypes simultaneously (structuring 0.92, burst 0.84, alternating 0.96, bot-periodic 0.65, dormant-spike 0.98).
     - `--profile streaming`: Optimized for real-time transaction ingestion and sliding-window event streams. Uses **3-Signal Fusion** ($0.20 \times \text{rank}_{\text{IF}} + 0.10 \times \text{rank}_{\text{TF, Mean}} + 0.70 \times \text{rank}_{\text{TF, Win5}}$). Excels at detecting subtle in-situ injected cycles against long, established benign histories where tabular features are diluted (+0.300 recall lift at $k=3$ cycles over Fixed IF).
   - **Production Default Pipeline Config**:
     - `profile = 'batch'`: Production default mode.
     - `if_max_features = 1.0`: All 28 features considered per tree split.
     - `if_weight = 0.8`, `transformer_weight = 0.2`: Calibrated whole-address ensemble fusion.
   - **Plain-Language Trade-Off Analysis**:
     - *Global Tabular vs. Local Sequence Dynamics*: Tabular Isolation Forest aggregates behavioral metrics across an entity's entire lifespan, excelling on whole-address synthetics (0.860–0.870 Recall@K). However, in realistic streaming scenarios where an attacker contaminates an established account with a brief burst of fraud, tabular averages are heavily diluted by benign history (dropping to 0.320 recall on 25+ event accounts). The Sequence Transformer's sliding-window max surprise (`window5`) focuses strictly on localized event clusters, restoring detection power (+0.220 to +0.300 recall lift on long histories).
     - *Why `max_features=1.0` is the Default*: Setting `max_features=sqrt` (5 features per split) helps structuring recall (0.860 -> 0.935) by preventing alternation features from competing with amount variance splits, but it degrades burst recall (-3.5%) and bot-periodic recall (-4.5%), decreasing overall Recall@K from 0.861 to 0.848. Subsampling features adds tree variance; `max_features=1.0` is retained as the default to ensure balanced detection across all fraud archetypes simultaneously.
   - Multiple fusion strategies supported: **Mean-Rank** (0.5/0.5 rank average), **Max-Rank** ($\max(\text{rank}_{\text{IF}}, \text{rank}_{\text{TF}})$), **Tuned Weighted Fusion** ($0.8 \times \text{rank}_{\text{IF}} + 0.2 \times \text{rank}_{\text{TF}}$), and **3-Signal Fusion** ($0.20 \times \text{rank}_{\text{IF}} + 0.10 \times \text{rank}_{\text{TF, Mean}} + 0.70 \times \text{rank}_{\text{TF, Win5}}$).
   - Entities with fewer than `min_seq_len` (default 3) events (143 with 1 tx + 131 with 2 txs = 274 addresses) are tagged with `insufficient_history=True` and scored via Isolation Forest alone.

8. **Clustering Diagnostics & Subset Analysis**:
   - Tabular KMeans identifies $k=2$ clusters with extreme single-cluster dominance (Cluster 1: 634 addresses / 98.1%; Cluster 0: 12 volume outliers / 1.9%), triggering an automated `>90% single cluster` diagnostic flag.
   - HDBSCAN evaluated strictly on addresses with $\ge 3$ events ($N=372$) produces 2 clusters (silhouette 0.1365, sizes: Cluster 0 with 290 entities, Cluster 1 with 7 entities, and 75 noise points). KMeans on this subset yields balanced clusters ($k=2$, silhouette 0.3712, sizes: 180 and 192).

9. **Synthetic Anomaly Benchmark Suite (`decrypt.benchmark`, Gen v2.0)**:
   - Evaluates performance against 5 realistic fraud archetypes: Structuring, Bursts, Alternating Deposit/Withdraw (formerly pass-through, with delay variants: 15m, 1h, 6h, 24h), Bot-Periodic, and Dormant-Spike patterns.
   - Evaluates across 10 random seeds with 20 injections per archetype (100 synthetic anomalies/seed).
   - Computes 95% bootstrap confidence intervals for all metrics as well as **paired bootstrap CIs of the difference** ($\text{Tuned Fusion} - \text{Fixed IF}$), marking intervals where the 95% bootstrap CI excludes zero or touches zero (borderline).

10. **Multi-Page Reporting & API Services**:
    - Programmatically generates `risk_scores.csv`, a 6-panel visualization dashboard (`fraud_analysis_patterns.png`), and a detailed PDF risk dossier (`fraud_analysis_report.pdf`).
    - Exposes complete real-time REST API with `/transactions/ingest`, `/risk/{entity_id}`, `/metrics`, and `/health`.

---

## Installation

We recommend using [`uv`](https://github.com/astral-sh/uv) for fast and reliable dependency resolution:

```bash
# Clone the repository
git clone https://github.com/AllRock666/DeCrypt.git
cd DeCrypt

# Create virtual environment and install dependencies
uv venv venv
uv pip install -r requirements.txt
```

*(Alternatively: `python -m venv venv && pip install -r requirements.txt`)*

---

## Quickstart

### 1. Launch the Real-Time Ingestion Service

```bash
uvicorn decrypt.service.app:app --host 0.0.0.0 --port 8000 --reload
```

- **Health Check**: `GET http://localhost:8000/health`
- **Interactive OpenAPI UI**: `http://localhost:8000/docs`
- **Telemetry Metrics**: `GET http://localhost:8000/metrics`

### 2. Ingest a Transaction via REST

```bash
curl -X POST http://localhost:8000/transactions/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx_demo_001",
    "source": "crypto_ethereum",
    "network": "ethereum_mainnet",
    "entity_id": "0x71C...b847",
    "transaction_hash": "0xabc...123",
    "timestamp": "2026-11-07T10:00:00Z",
    "transaction_type": "deposit",
    "amount": "2.5",
    "asset": "ETH"
  }'
```

### 3. Evaluate Real-Time Entity Risk

```bash
curl http://localhost:8000/risk/0x71C...b847
```

### 4. Run Offline Forensic Batch Pipeline

Analyze a transaction CSV archive and output the complete risk dossier:

```bash
python -m decrypt.main --data data/bitcoin_transactions.csv --outdir output
```

### 5. Run Synthetic Anomaly Benchmark

Evaluate the models across 10 random seeds against 5 fraud archetypes:

```bash
python -m decrypt.main --data data/bitcoin_transactions.csv --outdir output --benchmark
```

### 3. CLI Arguments Reference

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--data` | `str` | *Required* | Path to input CSV file. |
| `--outdir` | `str` | `.` | Target directory for generated outputs and models. |
| `--benchmark` | `flag` | `False` | Run the 10-seed synthetic anomaly benchmark suite. |
| `--profile` | `str` | `batch` | Execution profile: `batch` (IF-led 0.8/0.2 mean fusion) or `streaming` (3-signal 0.2/0.1/0.7 window-5). |
| `--include-network-features` | `flag` | `False` | Include heuristic graph network features in scoring (Negative Result; defaults to False). |
| `--no-transformer` | `flag` | `False` | Disable Transformer sequence modeling (IF only). |
| `--legacy-features` | `flag` | `False` | Reproduce the legacy baseline feature pipeline. |
| `--epochs` | `int` | `20` | Training epochs for the sequence Transformer. |
| `--seed` | `int` | `42` | Random seed for reproducibility. |
| `--contamination` | `float` | `0.1` | Expected outlier proportion for Isolation Forest. |
| `--clusters` | `int` | `10` | Maximum cluster count for silhouette search. |
| `--link-window-hours` | `float` | `24.0` | Heuristic graph time window in hours. |
| `--link-tolerance` | `float` | `0.02` | Heuristic graph amount matching tolerance. |
| `--has-header` | `flag` | `False` | Specify if input CSV has a header row. |

---

## Generated Output Artifacts

Running the pipeline populates `--outdir` with:
- `risk_scores.csv`: Per-address risk breakdown (`if_risk_score`, `transformer_surprise`, `ensemble_risk_score`, `cluster_id`, `hdbscan_cluster_id`, `insufficient_history`).
- `fraud_analysis_report.pdf`: Executive PDF summary including methodology, Top 20 high-risk addresses with Top-3 surprising transaction events, cluster statistics, and benchmark table.
- `fraud_analysis_patterns.png`: Visual analytics dashboard (risk distributions, amounts, cluster sizes, heuristic flow subgraph, hourly activity).
- `transformer_model.pt` & `transformer_bins.json`: Trained PyTorch Transformer weights and fitted quantile bin cutoffs.
- `benchmark_results.json` & `benchmark_curves.png`: (When `--benchmark` is set) Cross-seed metrics, paired difference CIs, pass-through delay ablation, and ROC/PR curve comparisons.
- `validation_controls.json`: Empirical results for length-control truncation, amount-perturbation invariance, in-situ contamination, and real address turnaround distributions.

---

## Running Tests

Execute the automated test suite with `pytest`:

```bash
python -m pytest tests/
```

Test coverage includes feature correctness (e.g. deposit 100 / withdraw 100, direction alternation dynamics), heuristic graph linkages, sequence tensor shapes, and an end-to-end smoke test on a 50-address sample.

---

## Known Dataset Characteristics & Caveats

1. **Absence of Ground-Truth Labels**: The provided dataset (`data/bitcoin_transactions.csv`) lacks fraud labels. Benchmark numbers are computed against realistic synthetic proxy archetypes; real addresses may contain undetected organic anomalies.
2. **Heuristic Flow Graph (Negative Result)**: Raw Bitcoin transactions in this dataset consist of isolated address records rather than multi-input/multi-output transactions. Graph edges (137 edges) are heuristically inferred based on temporal proximity ($\le 24$h) and amount matching ($\le 2\%$). In permutation testing, shuffling timestamps produces a mean of 183.5 edges and shuffling amounts produces a mean of 154.5 edges. Because randomized null permutations produce *equal or more* edges than the real graph, and because an empirical ablation demonstrates that network features degrade Isolation Forest performance ($\Delta \text{AUROC} = -0.0208$, $\Delta \text{AP} = -0.0566$), the graph is formally classified as a **Negative Result** and excluded from scoring by default.
3. **Short Histories Audit**: Exactly 274 addresses (42.41% of all 646 addresses) have fewer than 3 transactions (143 addresses with 1 transaction, 131 addresses with 2 transactions). These are flagged as `insufficient_history=True` and evaluated reliably via Isolation Forest without forcing sequence inference.
4. **Empirical Temporal Patterns**: In the real dataset, **29 addresses (4.49%)** exhibit a deposit followed by a withdrawal within 15 minutes, and **61 addresses (9.44%)** have inter-event gaps under 60 seconds.
5. **Clustering Diagnostics**: Tabular KMeans produces a cluster containing 98.14% of addresses (634/646) and an outlier cluster of 12 entities, triggering an automated cluster skew warning. HDBSCAN evaluated on the valid sequence history subset ($N=372$) yields 2 clusters (silhouette 0.0984, sizes: 339 and 7, 26 noise points), whereas KMeans on this subset yields balanced clusters ($k=2$, silhouette 0.4180, sizes: 187 and 185).
6. **Alternating Deposit/Withdraw Dynamics & Fund Conservation**: The Sequence Transformer achieves 1.000 recall across all delay variants for alternating deposit/withdraw patterns. With the addition of tabular direction features (`direction_switch_count`, `alternation_rate`, `max_alternating_run_length`), Fixed IF recall improves from 0.50 to 0.830 [0.760, 0.890], but does not fully match the Transformer (1.000). The model operates purely on discrete transition sequences and delta-t timings; it does *not verify fund conservation*. In validation controls (`output/validation_controls.json`), truncating sequences to 3–6 events drops Fixed IF recall to 0.200 while Transformer retains 0.910 recall. In in-situ contamination of real addresses ($\ge 5$ txs), injecting 2–3 alternating cycles yields 0.400 recall on Transformer vs 0.300–0.310 on Fixed IF.
