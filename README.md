# DeCrypt: Multi-Source Financial & Cryptocurrency Anomaly Detection Platform

> [!NOTE]
> **Research Prototype Status**: DeCrypt is an experimental research prototype for multi-source financial and cryptocurrency transaction anomaly detection. Baseline models and empirical reference distributions are calibrated exclusively on historical Bitcoin on-chain transactions (May 2011 to February 2014). An automated **source guard** restricts scoring on non-Bitcoin sources without refitting, and a **temporal-drift warning** triggers for transactions dated outside the historical calibration window (e.g. 2026). External provider adapters (Plaid Sandbox banking, Alchemy EVM, Esplora Bitcoin) are verified on recorded test fixtures only, without live network API validation.

DeCrypt operates in two complementary modes:
1. **Forensic Batch Engine**: Full offline dataset analysis generating multi-page PDF dossiers, cross-seed synthetic fraud benchmarks, clustering diagnostics, and CSV risk score exports.
2. **Continuous Real-Time Streaming Service**: A FastAPI service featuring sliding-window rate limiting, HMAC webhook signature verification, SHA-256 idempotency, incremental 28-feature extraction, and leave-one-out sequence surprise scoring.

---

## Architecture & System Overview

For detailed technical designs, refer to the dedicated documentation:
- 📖 [**Architecture Specification (`ARCHITECTURE.md`)**](file:///e:/projects/DeCrypt/DeCrypt/ARCHITECTURE.md)
- 🔌 [**REST API Reference (`API_DOCUMENTATION.md`)**](file:///e:/projects/DeCrypt/DeCrypt/API_DOCUMENTATION.md)
- 🚀 [**Installation & Setup Guide (`SETUP.md`)**](file:///e:/projects/DeCrypt/DeCrypt/SETUP.md)
- 🔍 [**Architecture & Security Audit (`ARCHITECTURE_AUDIT.md`)**](file:///e:/projects/DeCrypt/DeCrypt/ARCHITECTURE_AUDIT.md)

### Core Components

1. **Unified Domain & Canonical Schema (`decrypt.domain.schema`)**:
   - `UnifiedTransaction` schema enforcing arbitrary `Decimal` precision and strict UTC normalization.
   - Deterministic SHA-256 idempotency key generation across heterogeneous providers.
   - Wallet-level UTXO decomposition for Bitcoin transactions, distinguishing actual net spend/receipt from unrelated multi-input/output change flows.
   - Seamless bidirectional backward compatibility with legacy 5-column transaction CSVs.

2. **Multi-Source Ingestion Layer (`decrypt.providers`)**:
   - **Banking (Plaid Sandbox)**: `/transactions/sync` cursor-based synchronization, modification/removal handling, and webhook verification. Inverts Plaid debit amounts to standard outflows.
   - **EVM Blockchains (Alchemy)**: Native ETH and ERC-20 transfers across Ethereum, Arbitrum, Optimism, Polygon, and Base via `alchemy_getAssetTransfers`. Includes HMAC-SHA256 signature verification for address activity webhooks.
   - **Bitcoin (Esplora / Mempool.space)**: Confirmed on-chain transaction history, unconfirmed mempool tracking, and exact Satoshi-to-BTC fixed-base conversion.

3. **Incremental Real-Time ML Engine (`decrypt.pipeline`)**:
   - **State Management**: `EntityStateManager` maintains an in-memory sliding window of the last 128 events per entity.
   - **Incremental Behavioral Profiling**: `IncrementalFeatureExtractor` computes the exact 28 vectorized features on-the-fly without requiring full-dataset group-by table scans.
   - **Reference Distribution Calibration**: Eliminates $N=1$ rank distortion in real-time inference by scoring entities against frozen empirical CDF reference distributions (`scipy.stats.percentileofscore`).
   - **Sequence Surprise Inference**: Leave-one-out masking on `EventTransformerModel` evaluates negative log-likelihood (NLL) surprise of recent transactions without retraining.
   - **Automated Explainability**: Generates top anomalous behavioral features and identifies the top-3 most surprising sequence events.

4. **Vectorized Behavioral & Temporal Profiling (28 Tabular Features)**:
   - Computes true volume ($\sum |\text{amount}|$), net flow ($\sum \text{amount}$), pass-through ratios, and deposit/withdrawal distributions.
   - Activity span and normalized entity-relative temporal centroid `activity_centroid` (mean event offset from first transaction normalized by span).
   - Temporal dynamics via **Goh-Barabási burstiness** $B = \frac{\sigma - \mu}{\sigma + \mu}$, **Coefficient of Variation (CV)** of inter-event times, **lag-1 autocorrelation** of $\Delta t$, and Shannon amount entropy.
   - High-precision behavioral features: **modal-amount share** ($\frac{\text{count}(\text{mode})}{\text{total txs}}$) targeting automated/bot behavior.
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
   - Combines Isolation Forest deviance scores with Transformer surprise scores using percentile-rank calibration ($[0.0, 100.0]$).
   - **Execution Profiles**:
     - `--profile batch` (**Default**): Optimized for holistic whole-address forensic audits. Uses $0.80 \times \text{rank}_{\text{IF}} + 0.20 \times \text{rank}_{\text{TF, mean}}$ ($w_{\text{IF}} = 0.8, w_{\text{TF}} = 0.2$) with whole-sequence mean surprise. Achieves **0.964 AUROC, 0.900 AP, and 0.869 Recall@K**. Highest point estimates on aggregate metrics (AUROC and AP paired CIs vs Fixed IF exclude zero; Recall@K borderline); weaker than Fixed IF on burst and bot-periodic archetypes (structuring 0.92, burst 0.84, alternating 0.96, bot-periodic 0.65, dormant-spike 0.98).
     - `--profile streaming` (**Experimental**): Optimized for real-time transaction ingestion and sliding-window event streams. Uses **3-Signal Fusion** ($0.20 \times \text{rank}_{\text{IF}} + 0.10 \times \text{rank}_{\text{TF, mean}} + 0.70 \times \text{rank}_{\text{TF, win5}}$). Detects subtle in-situ injected cycles against long, established benign histories where tabular features are diluted (+0.229 [+0.112, +0.341] Recall@K lift at $k=3$ cycles over Fixed IF, overall).
   - **Production Default Pipeline Config**:
     - `profile = 'batch'`: Production default mode.
     - `if_max_features = 1.0`: All 28 features considered per tree split.
     - `if_weight = 0.8`, `transformer_weight = 0.2`: Calibrated whole-address ensemble fusion.
   - **Plain-Language Trade-Off Analysis**:
     - *Global Tabular vs. Local Sequence Dynamics*: Tabular Isolation Forest aggregates behavioral metrics across an entity's entire lifespan, performing strongly on whole-address synthetics (0.856 [0.835, 0.875] Recall@K for Fixed IF, 0.869 [0.852, 0.887] for Batch Fusion). However, in realistic streaming scenarios where an attacker contaminates an established account with subtle fraud, tabular averages are diluted by benign history (dropping to 0.360 [0.180, 0.540] Recall@K on 25+ event accounts at $k=3$). The Sequence Transformer's sliding-window max surprise (`window5`) focuses on localized event clusters, restoring detection power (+0.229 [+0.112, +0.341] Recall@K lift at $k=3$ cycles over Fixed IF, overall; reaching 0.600 [0.440, 0.740] on 25+ event accounts).
     - *Why `max_features=1.0` is the Default*: Setting `max_features=sqrt` (5 features per split) helps structuring recall (0.860 -> 0.935) by preventing alternation features from competing with amount variance splits, but it degrades pass-through recall (-7.5%) and bot-periodic recall (-9.0%), decreasing overall Recall@K from 0.861 to 0.848. Subsampling features adds tree variance; `max_features=1.0` is retained as the default to ensure balanced detection across fraud archetypes simultaneously.
   - Multiple fusion strategies supported: **Mean-Rank** ($0.50 \times \text{rank}_{\text{IF}} + 0.50 \times \text{rank}_{\text{TF}}$), **Max-Rank** ($\max(\text{rank}_{\text{IF}}, \text{rank}_{\text{TF}})$), **Tuned Weighted Fusion** ($0.80 \times \text{rank}_{\text{IF}} + 0.20 \times \text{rank}_{\text{TF, mean}}$), and **3-Signal Fusion** ($0.20 \times \text{rank}_{\text{IF}} + 0.10 \times \text{rank}_{\text{TF, mean}} + 0.70 \times \text{rank}_{\text{TF, win5}}$).
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
    - Exposes real-time REST API with `/transactions/ingest`, `/risk/{entity_id}`, `/metrics`, and `/health`.

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

### 1. Run Offline Forensic Batch Pipeline

Analyze a transaction CSV archive and output the complete risk dossier:

```bash
python -m decrypt.main --data data/bitcoin_transactions.csv --outdir output
```

### 2. Run Synthetic Anomaly Benchmark

Evaluate the models across 10 random seeds against 5 fraud archetypes:

```bash
python -m decrypt.main --data data/bitcoin_transactions.csv --outdir output --benchmark
```

### 3. Launch the Real-Time Ingestion Service

```bash
uvicorn decrypt.service.app:app --host 0.0.0.0 --port 8000 --reload
```

- **Health Check**: `GET http://localhost:8000/health`
- **Interactive OpenAPI UI**: `http://localhost:8000/docs`
- **Telemetry Metrics**: `GET http://localhost:8000/metrics`

### 4. Ingest a Transaction via REST

```bash
curl -X POST http://localhost:8000/transactions/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test_key" \
  -d '{
    "transaction_id": "tx_demo_001",
    "source": "crypto_bitcoin",
    "network": "bitcoin_mainnet",
    "entity_id": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
    "transaction_hash": "0xabc...123",
    "timestamp": "2013-05-01T10:00:00Z",
    "transaction_type": "deposit",
    "amount": "2.5",
    "asset": "BTC"
  }'
```

### 5. Evaluate Real-Time Entity Risk

```bash
curl -H "X-API-Key: test_key" http://localhost:8000/risk/1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
```

### 6. CLI Arguments Reference

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--data` | `str` | *Required* | Path to input CSV file. |
| `--outdir` | `str` | `.` | Target directory for generated outputs and models. |
| `--benchmark` | `flag` | `False` | Run the 10-seed synthetic anomaly benchmark suite. |
| `--profile` | `str` | `batch` | Execution profile: `batch` (Default, IF-led 0.8/0.2 mean fusion) or `streaming` (Experimental, 3-signal 0.2/0.1/0.7 window-5). |
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

Test coverage includes feature correctness (e.g. deposit 100 / withdraw 100, direction alternation dynamics), feature parity between batch and incremental extractors across 20 real entities, heuristic graph linkages, sequence tensor shapes, and a complete smoke test on a 50-address sample.

---

## Known System Limitations & Dataset Characteristics

To maintain scientific and engineering integrity, the following platform limitations are explicitly documented (mirroring `CHANGES.md` Section 7):

1. **Absence of Ground-Truth Labels**: The underlying Bitcoin dataset (`data/bitcoin_transactions.csv`) lacks verified forensic fraud labels. All supervised evaluation metrics (AUROC, AP, Recall@K) are measured exclusively against synthetic proxy injections. Real addresses labeled as "normal" in benchmarks may contain undetected historical fraud or mixer interactions.
2. **Synthetic Anomaly Approximations**: The 5 fraud archetypes (Structuring, Bursts, Alternating Cycles, Bot-Periodic, Dormant-Spike) are generated using stylized mathematical distributions. Real-world adversaries employ dynamic evasion techniques (e.g. varying amount splits, non-uniform delays, decoy transactions) that may not match these templates.
3. **Benchmark Circularity**: The tabular features introduced in v2.0 (`direction_switch_count`, `alternation_rate`, `cv_inter_event_time`, `modal_amount_share`, `lag1_autocorr_inter_event_time`) encode the mathematical definitions of the synthetic archetypes by construction. Their high benchmark scores reflect alignment with the generator rather than verified generalizability against unseen adversarial tactics.
4. **Subtle In-Situ Contamination Ceiling ($k=1$ Recall $\sim 0.31$)**: When an adversary injects only a single alternating cycle ($k=1$, 2 transactions) into an established address history, all models achieve low recall ($\sim 0.31 - 0.35$). Paired bootstrap confidence intervals against Fixed IF span zero. Two isolated transactions are statistically indistinguishable from legitimate behavioral variance.
5. **Training Domain & Temporal Calibration Bounds**: The pre-trained Transformer and baseline reference distributions were trained on early Bitcoin data (May 2011 to February 2014). The model is uncalibrated for modern DeFi protocols, Ethereum ERC-20 smart contracts, or fiat banking transactions. Scoring out-of-domain sources without fitting a dedicated empirical reference distribution triggers source-guard rejection.
6. **Heuristic Flow Graph Negative Result**: Because public cryptocurrency transaction exports lack direct UTXO input-output linkage graphs, heuristic time-and-amount flow matching produces more edges on randomly shuffled data (154–183 edges) than on real data (137 edges). Inclusion of these network features actively degrades Isolation Forest Recall@K by -0.051. The graph is excluded from scoring by default.
7. **Short Histories Audit**: Exactly 274 addresses (42.41% of all 646 addresses) have fewer than 3 transactions (143 addresses with 1 transaction, 131 addresses with 2 transactions). These are flagged as `insufficient_history=True` and evaluated via Isolation Forest without forcing sequence inference.
8. **Empirical Temporal Patterns**: In the real dataset, **29 addresses (4.49%)** exhibit a deposit followed by a withdrawal within 15 minutes, and **61 addresses (9.44%)** have inter-event gaps under 60 seconds.
9. **Clustering Diagnostics**: Tabular KMeans produces a cluster containing 98.14% of addresses (634/646) and an outlier cluster of 12 entities, triggering an automated cluster skew warning. HDBSCAN evaluated on the valid sequence history subset ($N=372$) yields 2 clusters (silhouette 0.1365, sizes: 290 and 7, 75 noise points), whereas KMeans on this subset yields balanced clusters ($k=2$, silhouette 0.3712, sizes: 180 and 192).
10. **Alternating Deposit/Withdraw Dynamics & Fund Conservation**: The Sequence Transformer achieves 1.000 recall across all delay variants for alternating deposit/withdraw patterns. With the addition of tabular direction features (`direction_switch_count`, `alternation_rate`, `max_alternating_run_length`), Fixed IF recall improves from 0.50 to 0.830 [0.760, 0.890], but does not fully match the Transformer (1.000). The model operates purely on discrete transition sequences and delta-t timings; it does *not verify fund conservation*.
