# DeCrypt: Comprehensive Architectural Audit

**Document Version**: 1.0.0  
**Date**: November 2026  
**Auditor**: Senior Systems Architect & Security Engineer  
**Repository**: `DeCrypt` (`feature/transformer-sequence-model`)

---

## 1. Executive Summary

DeCrypt was originally developed as an offline batch research framework for Bitcoin fraud and anomaly detection, pairing vectorized behavioral feature extraction with a 2-layer BERT-style Masked Event Transformer and an Isolation Forest ensemble. 

While the core statistical and sequence modeling foundations are scientifically sound—yielding strong validation benchmarks (**AUROC 0.969**, **AP 0.919**, **Recall@K 0.871** on synthetic archetypes)—the platform currently operates exclusively as an offline batch processor. 

To transform DeCrypt into a **production-grade, continuously operating, multi-source financial fraud detection engine**, significant architectural gaps must be addressed across:
1. Ingestion of multi-source banking (Plaid) and multi-chain cryptocurrency (Alchemy EVM, Blockstream/Mempool.space Bitcoin).
2. Canonical transaction schema, Decimal currency precision, and wallet-event level decomposition.
3. Real-time REST & webhook ingestion service (FastAPI) with idempotency, rate limiting, and signature verification.
4. Incremental stateful feature calculation, streaming Transformer surprise scoring, and empirical CDF calibration.
5. Persistent storage (PostgreSQL/SQLAlchemy) and operational observability (Prometheus/health metrics).

---

## 2. Detailed Findings & Vulnerability Matrix

| ID | Category | Severity | Affected Modules | Description & Verified Root Cause | Proposed Fix |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **AUD-01** | Data Ingestion | **High** | `decrypt/data.py` | **Static Schema & Missing Ingestion Providers**: Hardcoded for a 5-column Bitcoin CSV format (`address_id, hash, timestamp, type, amount`). No support for banking APIs (Plaid) or modern blockchain nodes (Alchemy EVM, Mempool.space). | Implement modular provider adapters (`PlaidAdapter`, `AlchemyAdapter`, `EsploraAdapter`) and an ingestion router. |
| **AUD-02** | Asset Precision | **Critical** | `decrypt/data.py`, `decrypt/features.py` | **Float Precision & Single-Asset Conflation**: Amounts are parsed into standard 64-bit IEEE 754 floats (`df['amount'].astype(float)`). This induces rounding drift on 18-decimal EVM tokens (Wei) and 8-decimal Bitcoin (Satoshi). Furthermore, amounts are treated as unitless, which would fatally conflate USD/INR and BTC/ETH if mixed. | Implement a canonical `UnifiedTransaction` schema with `Decimal` amount precision, asset isolation (`asset="BTC"`, `asset="ETH"`, `asset="USD"`), and separate scaling pipelines. |
| **AUD-03** | Data Ingestion | **High** | `decrypt/data.py` | **Lack of Idempotency & Duplicate Inflation**: Ingesting the same transactions twice (e.g. via overlapping batch loads or network retries) inflates transaction count, volume, and pass-through ratios without deduplication. | Enforce deterministic transaction IDs (`provider:network:txid:index`) and unique database constraints for idempotent upserts. |
| **AUD-04** | Blockchain Semantics | **Critical** | `decrypt/features.py`, `decrypt/data.py` | **Transaction-Level vs. Wallet-Level Conflation**: Raw Bitcoin transactions involve multiple inputs and outputs. Currently, the dataset assumes each record is already pre-aggregated per address. In real-world blockchain data, an address can appear in inputs, outputs, or both. Treating an entire UTXO transaction as a single withdrawal creates gross volume inflation. | Implement wallet-event level decomposition where incoming blockchain transactions are mapped relative to watched entity balances (`net_wallet_delta = total_received - total_spent`). |
| **AUD-05** | Streaming Inference | **Critical** | `decrypt/models.py`, `decrypt/sequence.py` | **Batch Refitting & Calibration Leakage**: `IsolationForest` scoring and percentile ranking (`rankdata(scores) / len(scores) * 100`) assume an entire batch is present simultaneously. For a single real-time streaming transaction ($N=1$), `rankdata` trivially outputs `100.0` (maximum risk)! | Calibrate scores against an empirical CDF fitted on historical baseline scores; persist scaler and IF estimators to disk via a `ModelRegistry`. |
| **AUD-06** | Real-Time Latency | **High** | `decrypt/sequence.py` | **Transformer Inference Redundancy**: In `score_addresses`, the Transformer computes leave-one-out NLL by masking all $L$ sequence positions. In real-time streaming, calculating all $L$ positions for every single incoming event introduces prohibitive latency ($O(L \cdot d_{\text{model}})$). | Implement streaming surprise scoring: evaluate cross-entropy surprise on the newest event conditional on the cached sequence history prefix. |
| **AUD-07** | State Management | **High** | `decrypt/features.py` | **Stateless Feature Re-Extraction**: `extract_behavioral_features` requires the entire historical transaction DataFrame to compute volume, delta-t stats, burstiness, and alternation rates. Running full groupby aggregations on 100k+ historical rows per transaction is unsustainable. | Introduce an `EntityStateManager` / incremental feature updater that maintains rolling counts, running means/variances, and sequence buffers. |
| **AUD-08** | Timezone Integrity | **Medium** | `decrypt/data.py` | **Naive Datetime Parsing**: Timestamps are parsed as naive local datetimes (`pd.to_datetime(format='%d-%m-%Y %H:%M')`). Combining data across Plaid (UTC), Mempool.space (UNIX epoch), and Alchemy (ISO 8601 UTC) without timezone awareness creates artificial temporal warping and bogus $\Delta t$ spikes. | Normalize all timestamps strictly to UTC timezone-aware `datetime` objects (`datetime.now(timezone.utc)`). |
| **AUD-09** | Architecture | **Critical** | `decrypt/main.py` | **Missing Real-Time Service & API Layer**: DeCrypt exists solely as a CLI batch script (`python -m decrypt.main`). There is no HTTP REST API, webhook listener, background processing queue, or async task runner. | Build a production-ready FastAPI application with async endpoints (`/transactions/ingest`, `/risk/{entity_id}`, `/webhooks/*`). |
| **AUD-10** | Security | **High** | None (Missing) | **Unauthenticated Webhook Exposure & Secrets**: No verification for Plaid or Alchemy webhook payloads; missing `.env` configuration; risks of replay attacks or forged transaction injections. | Implement webhook HMAC signature verification (`Alchemy-Signature`, Plaid verification tokens), environment-based settings, and secret redaction. |
| **AUD-11** | Explainability | **Medium** | `decrypt/reporting.py` | **Coarse Attribution & Absence of SHAP**: While Transformer top-3 surprising events are extracted, tabular Isolation Forest contributions are unranked; explanations lack feature-level attribution for high-risk entities. | Integrate tree feature contribution / TreeSHAP approximations to report the top-3 tabular features elevating an entity's risk score. |
| **AUD-12** | Database Layer | **High** | None (Missing) | **Lack of Durable Persistence**: Analysis results are saved to static flat files (`risk_scores.csv`, `transformer_model.pt`). In a production system, entities, transactions, risk scores, and audit logs must be durably stored and queryable via SQL. | Implement a comprehensive SQLAlchemy 2.0 ORM schema (PostgreSQL with SQLite fallback for local development/testing). |
| **AUD-13** | Network Features | **Low** | `decrypt/features.py` | **Residual Negative Result Footprint**: The heuristic flow graph was empirically confirmed to produce edges not exceeding shuffled null models (137 vs 154-183) and degrades AUROC by -0.0143. While disabled by default, network features still reside in the extraction codebase. | Retain network extraction strictly as an optional experimental plugin; keep default scoring pipelines purely tabular-behavioral and sequence-based. |
| **AUD-14** | Monitoring & Health | **Medium** | None (Missing) | **Zero Operational Visibility**: No telemetry for ingestion throughput, duplicate rates, inference latency, or webhook verification failures. | Expose a structured `/health` and `/metrics` endpoint tracking operational metrics and worker health. |

---

## 3. Architecture Target Blueprint

```
+-----------------------------------------------------------------------------------+
|                            DeCrypt Real-Time Platform                             |
+-----------------------------------------------------------------------------------+
                                         |
     +-----------------------------------+-----------------------------------+
     |                                   |                                   |
[Plaid Sandbox]                 [Alchemy EVM Node]               [Esplora / Mempool]
Traditional Banking              ETH + ERC-20 Tokens              Bitcoin UTXO
     |                                   |                                   |
     +-----------------------------------+-----------------------------------+
                                         |
                                         v
                      +-------------------------------------+
                      |      FastAPI Ingestion Gateway      |
                      |   - Webhook HMAC Signature Verify   |
                      |   - Rate Limiting & Auth Check      |
                      |   - Pydantic Input Validation       |
                      +-------------------------------------+
                                         |
                                         v
                      +-------------------------------------+
                      |    Unified Canonical Normalizer     |
                      |   - Decimal Asset Precision         |
                      |   - UTC Timestamps                  |
                      |   - Wallet-Level Event Resolution   |
                      |   - Deterministic Idempotency Key   |
                      +-------------------------------------+
                                         |
                     +-------------------+-------------------+
                     |                                       |
                     v                                       v
        +-------------------------+             +-------------------------+
        |   SQLAlchemy 2.0 DB     |             |  Real-Time ML Pipeline  |
        | - Entities              |             | - Incremental Features  |
        | - Transactions          |             | - Stream Transformer    |
        | - Risk Scores & Audits  |             | - 0.8 / 0.2 Fusion      |
        +-------------------------+             | - SHAP Attribution      |
                     |                          +-------------------------+
                     +-------------------+-------------------+
                                         |
                                         v
                      +-------------------------------------+
                      |    REST Query & Monitoring APIs     |
                      |   - GET /risk/{entity_id}           |
                      |   - GET /transactions               |
                      |   - GET /health & /metrics          |
                      +-------------------------------------+
```

---

## 4. Preservation Plan for Existing Research & Benchmarks

To maintain strict scientific and backwards compatibility:
1. **Existing Models Preserved**: The trained weights (`transformer_model.pt`), quantile bins (`transformer_bins.json`), and Isolation Forest estimators remain fully loadable.
2. **0.8 / 0.2 Fusion Preserved**: The proven ensemble ratio ($0.8 \times \text{rank}_{\text{IF}} + 0.2 \times \text{rank}_{\text{TF}}$) is retained as the baseline scoring mechanism.
3. **5-Column CSV Compatibility**: The ingestion layer transparently processes the historical `data/bitcoin_transactions.csv` dataset.
4. **Reproducible Synthetic Benchmarks**: [`decrypt/benchmark.py`](file:///e:/projects/DeCrypt/DeCrypt/decrypt/benchmark.py) and CLI flags (`--benchmark`, `--legacy-features`) remain fully intact and executable.
