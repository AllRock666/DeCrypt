# DeCrypt: Architecture Audit & System Evolution Specification

**Document Version**: 2.1.0  
**Branch**: `feature/transformer-sequence-model`  
**Classification**: Engineering Audit & Target Specification  

---

## 1. Executive Summary

This document strictly separates:
1. **Part A: Original Codebase Findings**: Architectural, mathematical, and algorithmic deficiencies verified specifically within the original batch research codebase (`decrypt/data.py`, `decrypt/features.py`, `decrypt/models.py`, `decrypt/sequence.py`, `decrypt/reporting.py`).
2. **Part B: Production Design Requirements**: Forward-looking system requirements and specifications needed to evolve DeCrypt from an offline batch script into a continuous, real-time, multi-source financial and cryptocurrency fraud anomaly detection platform.

---

## 2. Part A: Original Codebase Findings (Deficiencies & Technical Flaws)

The original repository was developed as an offline research framework for forensic analysis on historical Bitcoin CSV dumps (`data/bitcoin_transactions.csv`). The following flaws and deficiencies were verified directly in the original source code:

| ID | Module(s) | Severity | Category | Verified Flaw in Original Code |
| :--- | :--- | :---: | :--- | :--- |
| **AUD-ORIG-01** | `decrypt/features.py` | **High** | Feature Engineering | **Net-Flow Volume Masking Bug**: Transaction volume was computed as `grouped['amount'].sum().abs()`. Because deposit amounts were positive and withdrawal amounts negative, an address depositing 100 BTC and withdrawing 100 BTC cancelled out to net zero (`0.0`), completely masking high-volume layering, pass-through, and structuring activity as zero volume. |
| **AUD-ORIG-02** | Entire Codebase | **High** | Model Validation | **Absent Evaluation Harness & Ground-Truth Labels**: The original codebase contained zero evaluation metrics, no synthetic anomaly benchmarks, no train/test splits, and no ground-truth validation labels. Anomaly detection performance was completely unmeasured. |
| **AUD-ORIG-03** | `decrypt/models.py` | **High** | Anomaly Calibration | **Arbitrary Risk-Score Formula**: Raw Isolation Forest decision function scores were converted to "risk scores" via `((1 - raw_scores) * 50).clip(0, 100)`. Because Isolation Forest `score_samples` produces raw values in $[-0.8, -0.4]$, this arbitrary linear shift compressed all samples into an uncalibrated 70–90 band without probabilistic foundation or percentile ranking. |
| **AUD-ORIG-04** | `decrypt/features.py` | **Medium** | Feature Engineering | **Flawed Global Recency Feature**: Recency features `days_since_first` and `days_since_last` were computed from the global dataset start timestamp (`first_tx - min_time`). This measured historical calendar position rather than relative entity recency or inactivity span, distorting temporal modeling across different observation windows. |
| **AUD-ORIG-05** | `decrypt/models.py` | **Medium** | Unsupervised Clustering | **Capped Elbow Heuristic without Silhouette Optimization**: KMeans clustering estimated $k$ using inertia differences capped arbitrarily at $k \le 10$ or $k \le 20$ (`optimal_k = elbow[-1] + 2`). It lacked silhouette score evaluation, frequently resulting in degenerate cluster assignments where a single cluster contained >98% of addresses. |
| **AUD-ORIG-06** | `decrypt/features.py` | **High** | Graph Modeling | **Heuristic Flow Graph as a Negative Result**: The original heuristic flow graph inferred transaction links based on 24-hour time windows and 2% amount matching. Empirical permutation testing proved that timestamp-shuffled null models produced equal or more edges (183.5 vs. real 137), and an ablation showed network features degraded model performance ($\Delta\text{AUROC} = -0.0143$, $\Delta\text{AP} = -0.0439$). Network features were excluded from default scoring. |
| **AUD-ORIG-07** | `decrypt/data.py` | **Medium** | Ingestion Architecture | **Rigid 5-Column CSV Schema Assumption**: Data loading was hardcoded to a five-column format (`address_id, hash, timestamp, type, amount`). The codebase lacked domain abstractions, entity modeling, or support for varying financial provider schemas. |
| **AUD-ORIG-08** | `decrypt/data.py`, `decrypt/features.py` | **Low** | Data Typing | **IEEE 754 Floating-Point Precision Loss**: Transaction amounts were loaded and stored as 64-bit IEEE 754 floats (`df['amount'].astype(float)`). *Note on satoshi precision*: In statistical anomaly detection and behavioral aggregations, float64 precision drift does not meaningfully alter model percentiles or rankings, although exact crypto accounting for 8-decimal Bitcoin satoshis and 18-decimal EVM wei requires string/Decimal preservation in storage. |
| **AUD-ORIG-09** | `decrypt/data.py` | **Low** | Data Integrity | **Naive Datetime Parsing**: Timestamps were parsed as offset-naive datetimes (`pd.to_datetime(format='%d-%m-%Y %H:%M')`). For single-source historical CSV batches with uniform timestamps, offset-naive datetimes functioned without runtime errors, though multi-source cross-timezone ingest required UTC standardization. |
| **AUD-ORIG-10** | `decrypt/sequence.py` | **Medium** | Model Architecture | **Hardcoded Transformer Hyperparameters & Masking Overhead**: The BERT-style Transformer encoder had hardcoded layer dimensions, heads, and sequence lengths embedded in the script rather than exposed via configuration. Additionally, leave-one-out masking evaluated all $L$ positions regardless of sequence recency. |
| **AUD-ORIG-11** | `decrypt/reporting.py` | **Medium** | Explainability | **Absence of Tabular Feature Attribution**: The original reporting generated scalar risk scores and Transformer surprise events, but provided no feature-level attribution explaining which specific behavioral features drove an Isolation Forest outlier classification. |

---

## 3. Part B: Production Design Requirements (Platform Extensions)

The following architectural capabilities were outside the scope of the original batch script. They represent target design specifications established for building the continuous, real-time, multi-source DeCrypt platform:

| Requirement ID | Domain | Priority | Design Specification & Target Architecture |
| :--- | :--- | :---: | :--- |
| **REQ-DES-01** | Algorithmic Scalability | **High** | **Stateless Full-Dataframe Groupby Scalability**: Offline batch recomputation of global Pandas groupbys must be replaced with an incremental sliding-window entity state manager (`EntityStateManager`, `IncrementalFeatureExtractor`) maintaining bounded in-memory histories (default 5,000 events) for real-time feature extraction. |
| **REQ-DES-02** | Inference Mathematics | **Critical** | **Streaming Percentile Calibration / $N=1$ Rank Distortion**: Batch ranking formula `rankdata(scores) / N * 100` collapses to 100.0% when scoring a single entity ($N=1$). Streaming inference requires empirical reference distributions (`if_reference_scores`, `tf_reference_surprises`) fitted during training to evaluate single-entity percentile ranks accurately. |
| **REQ-DES-03** | Banking Integration | **High** | **Plaid Sandbox Integration**: Implement a Plaid banking adapter utilizing the documented `/transactions/sync` cursor pagination protocol. Support `added`, `modified`, and `removed` sync streams. Invert Plaid debit amounts (positive in Plaid) into standard outflow `WITHDRAWAL` types. |
| **REQ-DES-04** | EVM Cryptocurrency | **High** | **Alchemy EVM Node Integration**: Implement an Alchemy JSON-RPC adapter querying `alchemy_getAssetTransfers` across Ethereum, Arbitrum, Optimism, Polygon, and Base. Accurately decode ERC-20 token contract decimals and distinguish native transfers from token transfers. |
| **REQ-DES-05** | Bitcoin Blockchain | **Critical** | **Wallet-Level UTXO Decomposition**: Query Blockstream Esplora and Mempool.space APIs. Decompose multi-input/multi-output UTXO transactions relative to the watched wallet address, recording net delta (`total_received - total_spent`) rather than mistaking multi-party change outputs for massive withdrawals. |
| **REQ-DES-06** | Ingestion Safety | **High** | **Deterministic Idempotency Keying**: Implement deterministic idempotency keys generated via $\text{SHA256}(\text{source} \parallel \text{network} \parallel \text{tx\_hash} \parallel \text{entity\_id} \parallel \text{type} \parallel \text{amount} \parallel \text{timestamp})$ to guarantee safe deduplication on retries. |
| **REQ-DES-07** | Service & API Layer | **Critical** | **Real-Time Asynchronous Ingestion Service**: Introduce a FastAPI service providing high-throughput endpoints (`/transactions/ingest`, `/transactions`, `/risk/{entity_id}`, `/risk`, `/health`, `/metrics`, and `/webhooks/*`). |
| **REQ-DES-08** | Persistence Layer | **High** | **Relational Storage & Schema**: Introduce SQLAlchemy 2.0 ORM with tables for `entities`, `transactions`, `risk_scores`, and `idempotency_keys`. Support PostgreSQL connection pooling for production and SQLite with WAL mode for local zero-dependency testing. |
| **REQ-DES-09** | Security & Cryptography | **High** | **Webhook Signature Verification & Secret Masking**: Verify incoming Alchemy webhooks using constant-time HMAC-SHA256 validation (`X-Alchemy-Signature`). Mask API keys, private keys, and bearer tokens in log output via a custom `MaskingFormatter`. Enforce fail-closed API Key authentication on protected ingest and risk endpoints. |
| **REQ-DES-10** | Operational Telemetry | **Medium** | **Service Health, Telemetry & Rate Limiting**: Expose Prometheus-compatible telemetry metrics and implement a sliding-window memory rate limiter to protect against denial-of-service spikes. |
| **REQ-DES-11** | Model Integrity | **High** | **Source Guard & Domain Calibration**: Record model training domain metadata (Bitcoin / BTC). Reject scoring for uncalibrated transaction sources with `"status": "not calibrated for this source"` unless a per-source reference distribution has been registered. |
