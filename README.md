# DeCrypt: Bitcoin & Financial Transaction Fraud Detection Engine

**DeCrypt** is a modular, high-performance Python framework for detecting anomalous and fraudulent entities in cryptocurrency and financial transaction datasets. It combines **vectorized behavioral engineering**, a **heuristic temporal flow graph**, a **BERT-style Masked Event Transformer**, **Isolation Forest**, and **HDBSCAN/KMeans clustering**.

---

## Architecture & Features

1. **Vectorized Behavioral & Temporal Profiling**:
   - Computes true volume ($\sum |amount|$), net flow ($\sum amount$), pass-through ratios, and deposit/withdrawal distributions.
   - Measures inactivity via `days_until_end` relative to dataset boundaries.
   - Captures event dynamics using the **Goh-Barabási burstiness parameter** $B = \frac{\sigma - \mu}{\sigma + \mu}$ and Shannon entropy of transaction amounts.
2. **Heuristic Flow Graph**:
   - Traces fund movements between withdrawals and deposits occurring within a configurable window (e.g. 24h) and matching amount tolerance (e.g. 2%).
   - Calculates directed in/out-degree centralities, weighted degrees, clustering coefficients, and PageRank.
   - Gracefully skips network features with an explicit logged warning if fewer than `min_edges` are formed.
3. **Masked Event Transformer (`decrypt.sequence`)**:
   - 2-layer, 4-head Transformer encoder ($d_{\text{model}} = 64$) trained on discrete/continuous transaction sequences via 15% masked event modeling.
   - Generates leave-one-out event **negative log-likelihood (NLL) surprise scores**.
   - Provides per-address explainability by reporting the top-3 most surprising transactions.
   - Extracts mean-pooled sequence embeddings for non-linear density clustering via **HDBSCAN** and KMeans.
4. **Rank-Calibrated Ensemble**:
   - Combines Isolation Forest deviance scores with Transformer surprise scores using percentile-rank calibration ($0.0 - 100.0$).
   - Entities with fewer than `min_seq_len` (default 3) events are tagged with `insufficient_history=True` and scored via Isolation Forest alone.
5. **Synthetic Anomaly Benchmark Suite (`decrypt.benchmark`)**:
   - Evaluates performance against 5 realistic fraud archetypes: Structuring (Smurfing), Rapid Bursts, Pass-Through / Mixer Cycles, Bot-Periodic activity, and Dormant-then-Spike patterns.
   - Reports AUROC, Average Precision (AP), and Recall@K across multiple random seeds with ROC/PR curves.
6. **Multi-Page Reporting**:
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
2. **Heuristic Flow Linking**: Raw Bitcoin transactions in this dataset consist of isolated address records rather than multi-input/multi-output transactions. Graph edges are heuristically inferred based on temporal proximity and amount matching.
3. **Short Histories**: A portion of addresses have fewer than 3 transactions; these are appropriately flagged as `insufficient_history` and evaluated primarily via tabular anomaly detection.
