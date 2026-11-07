# DeCrypt: Architecture Audit & System Evolution Specification

**Document Version**: 2.0.0  
**Branch**: `feature/transformer-sequence-model`  
**Classification**: Engineering Audit & Target Specification  

---

## 1. Executive Summary

This document separates:
1. **Part A: Original Codebase Findings**: Architectural, mathematical, and implementation deficiencies identified specifically within the original batch research codebase (`decrypt/data.py`, `decrypt/features.py`, `decrypt/models.py`, `decrypt/sequence.py`, `decrypt/reporting.py`).
2. **Part B: Production Design Requirements**: Forward-looking system requirements and specifications needed to evolve DeCrypt from an offline batch script into a continuous, real-time, multi-source financial and cryptocurrency fraud anomaly detection platform.

---

## 2. Part A: Original Codebase Findings (Deficiencies & Technical Flaws)

The original repository was developed as an offline research framework for forensic analysis on historical Bitcoin CSV dumps (`data/bitcoin_transactions.csv`). The following deficiencies were verified directly in the original source code:

| ID | Module(s) | Severity | Category | Verified Flaw in Original Code |
| :--- | :--- | :---: | :--- | :--- |
| **AUD-ORIG-01** | `decrypt/data.py`, `decrypt/features.py` | **Critical** | Mathematical Precision | **IEEE 754 Floating-Point Precision Loss**: Transaction amounts were loaded and stored as 64-bit IEEE 754 floats (`df['amount'].astype(float)`). In crypto accounting, floating-point math induces cumulative precision drift on 8-decimal Bitcoin (Satoshis) and 18-decimal EVM tokens (Wei). |
| **AUD-ORIG-02** | `decrypt/data.py` | **High** | Data Integrity | **Naive Datetime Parsing**: Timestamps were parsed as offset-naive datetimes (`pd.to_datetime(format='%d-%m-%Y %H:%M')`). Combining data across international UTC sources without explicit timezone awareness caused timezone mismatch errors and invalid time-delta calculations. |
| **AUD-ORIG-03** | `decrypt/features.py` | **High** | Algorithmic Scalability | **Stateless Full-Dataframe Groupby Re-Extraction**: `extract_behavioral_features` relied on global Pandas group-by aggregations over the entire dataset. Computing features for a single entity required re-aggregating thousands of historical rows, prohibiting streaming or per-entity evaluation. |
| **AUD-ORIG-04** | `decrypt/features.py` | **High** | Model Methodology | **Heuristic Flow Graph as a Negative Result**: The original heuristic flow graph inferred transaction links based on 24-hour time windows and 2% amount matching. Empirical permutation testing proved that timestamp-shuffled null models produced equal or more edges (183.5 vs. real 137), and an ablation showed network features degraded model performance ($\Delta\text{AUROC} = -0.0143$). Network features needed to be excluded from default scoring. |
| **AUD-ORIG-05** | `decrypt/models.py`, `decrypt/sequence.py` | **Critical** | Inference Mathematics | **Batch-Only Percentile Calibration / $N=1$ Rank Distortion**: Percentile ranks were computed using `scipy.stats.rankdata(scores) / len(scores) * 100` over the current batch. When evaluating a single entity in a real-time context ($N=1$), this formula degenerates to $100.0\%$ (maximum risk), making real-time streaming rank fusion impossible without frozen empirical reference distributions. |
| **AUD-ORIG-06** | `decrypt/sequence.py` | **Medium** | Model Architecture | **Hardcoded Transformer Hyperparameters & Masking Overhead**: The BERT-style Transformer encoder had hardcoded layer dimensions, heads, and sequence lengths embedded in the script rather than exposed via configuration. Additionally, leave-one-out masking evaluated all $L$ positions regardless of sequence recency. |
| **AUD-ORIG-07** | `decrypt/reporting.py` | **Medium** | Explainability | **Absence of Tabular Feature Attribution**: The original reporting generated scalar risk scores and Transformer surprise events, but provided no feature-level attribution explaining which specific behavioral features drove an Isolation Forest outlier classification. |
| **AUD-ORIG-08** | `decrypt/data.py` | **Medium** | Ingestion Flexibility | **Rigid 5-Column CSV Schema Assumption**: Data loading was hardcoded to a five-column format (`address_id, hash, timestamp, type, amount`). The codebase lacked domain abstractions, entity modeling, or support for varying schemas. |

---

## 3. Part B: Production Design Requirements (Platform Extensions)

The following capabilities were outside the scope of the original batch script. They represent target design specifications established for building the real-time, multi-source DeCrypt platform:

| Requirement ID | Domain | Priority | Design Specification & Target Architecture |
| :--- | :--- | :---: | :--- |
| **REQ-DES-01** | Banking Integration | **High** | **Plaid Sandbox Integration**: Implement a Plaid banking adapter utilizing the documented `/transactions/sync` cursor pagination protocol. Support `added`, `modified`, and `removed` sync streams. Invert Plaid debit amounts (positive in Plaid) into standard outflow `WITHDRAWAL` types. |
| **REQ-DES-02** | EVM Cryptocurrency | **High** | **Alchemy EVM Node Integration**: Implement an Alchemy JSON-RPC adapter querying `alchemy_getAssetTransfers` across Ethereum, Arbitrum, Optimism, Polygon, and Base. Accurately decode ERC-20 token contract decimals and distinguish native transfers from token transfers. |
| **REQ-DES-03** | Bitcoin Blockchain | **Critical** | **Wallet-Level UTXO Decomposition**: Query Blockstream Esplora and Mempool.space APIs. Decompose multi-input/multi-output UTXO transactions relative to the watched wallet address, recording net delta (`total_received - total_spent`) rather than mistaking multi-party change outputs for massive withdrawals. |
| **REQ-DES-04** | Ingestion Safety | **High** | **Deterministic Idempotency Keying**: Implement deterministic idempotency keys generated via $\text{SHA256}(\text{source} \parallel \text{network} \parallel \text{tx\_hash} \parallel \text{entity\_id} \parallel \text{type} \parallel \text{amount} \parallel \text{timestamp})$ to guarantee safe deduplication on retries. |
| **REQ-DES-05** | Service & API Layer | **Critical** | **Real-Time Asynchronous Ingestion Service**: Introduce a FastAPI service providing high-throughput endpoints (`/transactions/ingest`, `/transactions`, `/risk/{entity_id}`, `/risk`, `/health`, `/metrics`, and `/webhooks/*`). |
| **REQ-DES-06** | Persistence Layer | **High** | **Relational Storage & Schema**: Introduce SQLAlchemy 2.0 ORM with tables for `entities`, `transactions`, `risk_scores`, and `idempotency_keys`. Support PostgreSQL connection pooling for production and SQLite with WAL mode for local zero-dependency testing. |
| **REQ-DES-07** | Security & Cryptography | **High** | **Webhook Signature Verification & Secret Masking**: Verify incoming Alchemy webhooks using constant-time HMAC-SHA256 validation (`X-Alchemy-Signature`). Mask API keys, private keys, and bearer tokens in log output via a custom `MaskingFormatter`. Enforce API Key authentication on protected ingest and risk endpoints. |
| **REQ-DES-08** | Operational Telemetry | **Medium** | **Service Health, Telemetry & Rate Limiting**: Expose Prometheus-compatible telemetry metrics and implement a sliding-window memory rate limiter to protect against denial-of-service spikes. |
| **REQ-DES-09** | Model Integrity | **High** | **Source Guard & Domain Calibration**: Record model training domain metadata (Bitcoin / BTC). Reject scoring for uncalibrated transaction sources with `"status": "not calibrated for this source"` unless a per-source reference distribution has been registered. |
