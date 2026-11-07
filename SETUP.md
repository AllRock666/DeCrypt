# DeCrypt: Installation & Operational Setup Guide

This guide details how to install, configure, and operate the DeCrypt anomaly detection system for both local development and production environments.

---

## 1. Prerequisites

- **Python**: 3.10, 3.11, 3.12, 3.13, or 3.14
- **Package Manager**: [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`
- **Database**:
  - Development: SQLite with WAL mode (zero-configuration, built-in)
  - Production: PostgreSQL 14+ (optional, recommended for enterprise deployments)
- **External API Keys** (Optional for mock/testing; required for live integrations):
  - Plaid Sandbox account (`PLAID_CLIENT_ID`, `PLAID_SECRET`)
  - Alchemy EVM API Key (`ALCHEMY_API_KEY`)
  - Blockstream Esplora endpoint (public API works out-of-the-box)

---

## 2. Environment Setup

### 2.1 Clone and Create Virtual Environment

Using `uv` (recommended):
```bash
git clone https://github.com/AllRock666/DeCrypt.git
cd DeCrypt

# Create virtual environment with uv
uv venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

Using standard `pip`:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 3. Configuration & Environment Variables

Copy the template configuration file:
```bash
cp .env.example .env
```

### Key Configuration Directives in `.env`

```ini
# Storage & Database
# For SQLite (default development):
DATABASE_URL=sqlite:///./decrypt_storage.db
# For PostgreSQL (production):
# DATABASE_URL=postgresql://decrypt_user:password@localhost:5432/decrypt_db

# FastAPI Service
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000
RATE_LIMIT_PER_MINUTE=600
DECRYPT_API_KEY=decrypt-dev-key

# Banking (Plaid Sandbox)
PLAID_CLIENT_ID=your_sandbox_client_id_here
PLAID_SECRET=your_sandbox_secret_here
PLAID_ENV=sandbox

# Cryptocurrency (Alchemy EVM)
ALCHEMY_API_KEY=your_alchemy_api_key_here
ALCHEMY_NETWORK=eth-mainnet
ALCHEMY_WEBHOOK_SIGNING_KEY=your_webhook_signing_key_here

# Bitcoin Explorer
ESPLORA_API_URL=https://blockstream.info/api

# Real-Time ML Engine
MODEL_DIR=./models
ENSEMBLE_WEIGHT_IF=0.8
ENSEMBLE_WEIGHT_TF=0.2
MIN_SEQ_LEN=3
HIGH_RISK_THRESHOLD=75.0
CRITICAL_RISK_THRESHOLD=90.0
```

> **Note**: If Plaid or Alchemy credentials are not supplied, the adapters automatically operate in mock/testing mode, allowing full API testing without external accounts.

---

## 4. Running the Real-Time Ingestion Service

Start the FastAPI application with Uvicorn:

```bash
uvicorn decrypt.service.app:app --host 0.0.0.0 --port 8000 --reload
```

Verify service readiness:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "database": "connected",
  "models": {
    "isolation_forest": "ready",
    "transformer": "ready",
    "calibrator": "ready"
  }
}
```

Interactive OpenAPI documentation is available in your browser at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 5. API Authentication Reference

The service enforces API authentication across protected ingest and risk evaluation endpoints:

| Endpoint | Method | Authentication Required? | Mechanism / Header |
| :--- | :---: | :---: | :--- |
| `/transactions/ingest` | `POST` | **Yes** | `X-API-Key: <key>` or `Authorization: Bearer <key>` (`DECRYPT_API_KEY`) |
| `/risk/{entity_id}` | `GET` | **Yes** | `X-API-Key: <key>` or `Authorization: Bearer <key>` (`DECRYPT_API_KEY`) |
| `/risk` | `GET` | **Yes** | `X-API-Key: <key>` or `Authorization: Bearer <key>` (`DECRYPT_API_KEY`) |
| `/webhooks/alchemy` | `POST` | **Yes** | `X-Alchemy-Signature: <hex_hmac>` (`ALCHEMY_WEBHOOK_SIGNING_KEY`) |
| `/webhooks/plaid` | `POST` | **Yes** | `X-Plaid-Webhook-Secret` or `Plaid-Verification` |
| `/transactions` | `GET` | No | Public read query (rate-limited to 600 req/min) |
| `/transactions/{id}` | `GET` | No | Public read query (rate-limited to 600 req/min) |
| `/health` | `GET` | No | Unauthenticated health/readiness probe |
| `/metrics` | `GET` | No | Unauthenticated operational telemetry |

---

## 6. Provider Testing: Fixtures vs. Live Calls

To ensure fast, deterministic, and credential-free CI/CD execution, test suites are strictly partitioned:

| Test Module / Suite | Execution Mode | Fixtures vs. Live Calls | External Credentials Required? |
| :--- | :---: | :--- | :---: |
| `tests/test_domain_schema.py` | Unit | **Hermetic Fixtures**: Tests Decimal precision, UTC parsing, and legacy CSV mapping using local objects. | None |
| `tests/test_providers.py` | Unit | **Hermetic Fixtures**: Mocks Plaid JSON payloads (verifying deposit/withdrawal inversion), Alchemy EVM transfer events (verifying ERC-20 contract resolution), and Bitcoin Esplora UTXO structures (verifying net spent/received Satoshis). | None |
| `tests/test_service_api.py` | Integration | **TestClient In-Memory Fixtures**: Tests FastAPI endpoints, sliding-window rate limiting, and HMAC webhook verification using mock payloads. | None |
| `tests/test_feature_parity.py` | Parity | **Local Dataset Fixture**: Tests exact parity between batch and incremental feature extractors across 20 real entities using local `data/bitcoin_transactions.csv`. | None |
| `tests/test_source_guard.py` | Unit | **Hermetic Fixtures**: Tests model training domain restriction (Bitcoin/BTC) and uncalibrated source rejection using synthetic in-memory transaction lists. | None |
| **Live Provider Adapter Calls** | Optional Runtime | **Live External Calls**: Invoked only when executing live synchronization scripts or when `PLAID_CLIENT_ID`, `ALCHEMY_API_KEY`, or `ESPLORA_API_URL` are provided in `.env`. | Required for live network sync |

---

## 7. Running the Test Suite

Execute the comprehensive test suite spanning schema validation, provider adapters, incremental ML scoring, storage, and API endpoints:

```bash
python -m pytest tests/ -v
```

All 26 test suites should pass green.

---

## 8. Running Offline Batch Processing & Benchmarks

DeCrypt retains 100% backward compatibility with offline forensic batch datasets:

### Standard Analysis (PDF Dossier, CSV Scores, Visualizations)
```bash
python -m decrypt.main --data data/bitcoin_transactions.csv --outdir output
```

### Full Benchmark Evaluation (10 seeds, 5 fraud archetypes)
```bash
python -m decrypt.main --data data/bitcoin_transactions.csv --outdir output --benchmark
```

Outputs will be generated in `./output`:
- `risk_scores.csv`
- `fraud_analysis_report.pdf`
- `fraud_analysis_patterns.png`
- `benchmark_results.json`

---

## 9. Connecting External Providers

### 7.1 Plaid Sandbox
1. Register for free at [dashboard.plaid.com/signup](https://dashboard.plaid.com/signup).
2. Retrieve `client_id` and `sandbox` secret from the Keys section.
3. Add them to `.env`.
4. Trigger sandbox mock transactions using Plaid's sandbox public tokens (`user_good`, `custom_sandbox_user`).

### 7.2 Alchemy EVM
1. Register at [alchemy.com](https://alchemy.com) and create an Ethereum app.
2. Copy the API key and set `ALCHEMY_API_KEY` in `.env`.
3. In the Alchemy Notify dashboard, create an Address Activity webhook pointing to `https://your-domain.com/webhooks/alchemy`.
4. Copy the webhook signing key and set `ALCHEMY_WEBHOOK_SIGNING_KEY` in `.env`.

### 7.3 Bitcoin Esplora / Mempool.space
- By default, DeCrypt uses Blockstream's public Esplora endpoint (`https://blockstream.info/api`).
- For high-volume private setups, set `ESPLORA_API_URL` to a local Mempool instance (`http://localhost:8999/api`).
