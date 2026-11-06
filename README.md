# DeCrypt: Bitcoin & Financial Transaction Fraud Detection Engine

**DeCrypt** is a modular, high-performance Python framework for detecting anomalous and fraudulent entities in cryptocurrency and financial transaction datasets. It combines **vectorized behavioral engineering**, a **heuristic temporal flow graph**, a **BERT-style Masked Event Transformer**, **Isolation Forest**, and **HDBSCAN/KMeans clustering**.

---

## Architecture & Features

1. **Vectorized Behavioral & Temporal Profiling**:
   - Computes true volume ($\sum |amount|$), net flow ($\sum amount$), pass-through ratios, and deposit/withdrawal distributions.
   - Measures inactivity via `days_until_end` relative to dataset boundaries.
   - Captures event dynamics using the **Goh-Barabási burstiness parameter** $B = \frac{\sigma - \mu}{\sigma + \mu}$ and Shannon entropy of transaction amounts.
2. **Heuristic Flow Graph & Permutation Sanity Audit**:
   - Traces fund movements between withdrawals and deposits occurring within a configurable window (e.g. 24h) and matching amount tolerance (e.g. 2%).
   - Calculates directed in/out-degree centralities, weighted degrees, clustering coefficients, and PageRank.
   - Gracefully skips network features with an explicit logged warning if fewer than `min_edges` are formed.
   - **Permutation Sanity Test & Confidence Level**: Evaluated against null permutation baselines (20 timestamp-shuffled runs: mean 184.8 edges; 20 amount-shuffled runs: mean 158.2 edges). Because shuffled runs produce equal or more edges than the real graph (137 edges) due to uniform temporal mixing, the heuristic flow graph is classified and labeled as **Low-Confidence** and reflects temporal co-occurrence rather than direct cryptographic provenance.
3. **Masked Event Transformer (`decrypt.sequence`)**:
   - 2-layer, 4-head Transformer encoder ($d_{\text{model}} = 64$) trained on discrete/continuous transaction sequences via 15% masked event modeling.
   - Generates leave-one-out event **negative log-likelihood (NLL) surprise scores**.
   - Provides per-address explainability by reporting the top-3 most surprising transactions.
   - Extracts mean-pooled sequence embeddings for non-linear density clustering via **HDBSCAN** and KMeans.
4. **Ensemble Risk Fusion**:
   - Combines Isolation Forest deviance scores with Transformer surprise scores using percentile-rank calibration ($0.0 - 100.0$).
   - Multiple fusion strategies supported: **Mean-Rank** (0.5/0.5 rank average), **Max-Rank** ($\max(\text{rank}_{\text{IF}}, \text{rank}_{\text{TF}})$), and **Tuned Weighted Fusion** ($0.8 \times \text{rank}_{\text{IF}} + 0.2 \times \text{rank}_{\text{TF}}$, calibrated on independent tuning seeds).
   - Entities with fewer than `min_seq_len` (default 3) events (143 with 1 tx + 131 with 2 txs = 274 addresses) are tagged with `insufficient_history=True` and scored via Isolation Forest alone.
5. **Clustering & Topology Diagnostics**:
   - Tabular KMeans identifies $k=2$ clusters (silhouette 0.718) but exhibits extreme cluster dominance (Cluster 1: 634 addresses / 98.1%; Cluster 0: 12 addresses / 1.9% containing extreme volume outliers), triggering the `>90% single cluster` diagnostic flag.
   - HDBSCAN on Transformer sequence embeddings partitions the dataset into 2 clusters (silhouette 0.442, sizes: 372 / 57.6% and 274 / 42.4%), cleanly separating entities with rich temporal trajectories from single/double-transaction wallets.
6. **Synthetic Anomaly Benchmark Suite (`decrypt.benchmark`)**:
   - Evaluates performance against 5 realistic fraud archetypes: Structuring (Smurfing), Rapid Bursts, Pass-Through / Mixer Cycles, Bot-Periodic activity, and Dormant-then-Spike patterns.
   - Evaluates across 10 random seeds with 20 injections per archetype (100 synthetic anomalies/seed).
   - Computes 95% bootstrap confidence intervals for all AUROC, AP, Recall@K, and per-archetype recall metrics.
7. **Multi-Page Reporting**:
   - Programmatically generates `risk_scores.csv`, a 6-panel visualization dashboard (`fraud_analysis_patterns.png`), and a detailed PDF risk dossier (`fraud_analysis_report.pdf`).

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

*(Alternatively, you can use standard Python: `python -m venv venv && pip install -r requirements.txt`)*

---

## Usage

### 1. Standard End-to-End Analysis

Analyze a transaction dataset and output the complete risk dossier:

```bash
python -m decrypt.main --data data/bitcoin_transactions.csv --outdir output
```

### 2. Run with Synthetic Anomaly Benchmark

Evaluate the models against the 5-seed synthetic anomaly benchmark:

```bash
python -m decrypt.main --data data/bitcoin_transactions.csv --outdir output --benchmark
```

### 3. CLI Arguments Reference

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--data` | `str` | *Required* | Path to input CSV file. |
| `--outdir` | `str` | `.` | Target directory for generated outputs and models. |
| `--benchmark` | `flag` | `False` | Run the 5-seed synthetic anomaly benchmark suite. |
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
- `benchmark_results.json` & `benchmark_curves.png`: (When `--benchmark` is set) Cross-seed metrics and ROC/PR curve comparisons.

---

## Running Tests

Execute the automated test suite with `pytest`:

```bash
python -m pytest tests/
```

Test coverage includes feature correctness (e.g. deposit 100 / withdraw 100), heuristic graph linkages, sequence tensor shapes, and an end-to-end smoke test on a 50-address sample.

---

## Known Dataset Characteristics & Caveats

1. **Absence of Ground-Truth Labels**: The provided dataset (`data/bitcoin_transactions.csv`) lacks fraud labels. Benchmark numbers are computed against realistic synthetic proxy archetypes; real addresses may contain undetected organic anomalies.
2. **Heuristic Flow Graph (Low Confidence)**: Raw Bitcoin transactions in this dataset consist of isolated address records rather than multi-input/multi-output transactions. Graph edges (137 edges) are heuristically inferred based on temporal proximity ($\le 24$h) and amount matching ($\le 2\%$). In permutation testing, shuffling timestamps produces a mean of 184.8 edges and shuffling amounts produces a mean of 158.2 edges. Because randomized temporal mixing produces more coincidental matches than the empirical sequence, the graph is formally classified as **Low-Confidence**.
3. **Short Histories Audit**: Exactly 274 addresses (42.4% of all 646 addresses) have fewer than 3 transactions (143 addresses with 1 transaction, 131 addresses with 2 transactions). These are flagged as `insufficient_history=True` and evaluated reliably via Isolation Forest without forcing sequence inference.
4. **KMeans Skew vs HDBSCAN**: Tabular KMeans produces a cluster containing 98.1% of addresses (634/646) and an outlier cluster of 12 entities, triggering an automated cluster skew warning. In contrast, HDBSCAN on Transformer sequence embeddings partitions the dataset into 372 (57.6%) and 274 (42.4%) addresses, cleanly reflecting sequence history depth.
