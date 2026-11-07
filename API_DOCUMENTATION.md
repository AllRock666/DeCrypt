# DeCrypt: REST API Specification & Integration Reference

The DeCrypt real-time service is built with FastAPI. It provides high-throughput transaction ingestion, idempotency verification, real-time behavioral anomaly scoring, telemetry metrics, and secure provider webhook endpoints.

**Base URL**: `http://localhost:8000` (configurable via `SERVICE_HOST` and `SERVICE_PORT`)

---

## Table of Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/transactions/ingest` | Ingest a single transaction or batch of transactions with idempotency. |
| `GET` | `/transactions` | Retrieve paginated transactions filtered by entity, status, or asset. |
| `GET` | `/transactions/{transaction_id}` | Retrieve full details of a specific transaction by ID. |
| `GET` | `/risk/{entity_id}` | Retrieve or trigger real-time anomaly risk evaluation for an entity. |
| `GET` | `/risk` | Retrieve top high-risk entities ordered by risk score. |
| `GET` | `/health` | System health check (database connection, ML model status). |
| `GET` | `/metrics` | Real-time performance counters and telemetry metrics. |
| `POST` | `/webhooks/plaid` | Handle Plaid Sandbox transaction sync webhooks. |
| `POST` | `/webhooks/alchemy` | Handle Alchemy EVM address activity transfer webhooks. |

---

## Authentication & Headers

### Standard Request Headers
- `Content-Type: application/json`
- `X-Idempotency-Key`: *(Optional)* Custom idempotency key to prevent duplicate processing. If omitted, a deterministic SHA-256 hash is generated from transaction attributes.

### Rate Limiting Headers
All responses include rate limiting headers:
- `X-RateLimit-Limit`: Maximum allowed requests per window (default: 600 per minute).
- `X-RateLimit-Remaining`: Remaining requests in current window.

When exceeded, the API returns `429 Too Many Requests` with:
- `Retry-After: 60`

---

## 1. Transaction Ingestion

### `POST /transactions/ingest`
Ingests a single transaction or a list of transactions into the persistent store, validates schema constraints, updates entity state, and triggers asynchronous anomaly evaluation.

#### Request Body
```json
{
  "transaction_id": "tx_eth_001928",
  "source": "crypto_ethereum",
  "network": "ethereum_mainnet",
  "entity_id": "0x71C...b847",
  "transaction_hash": "0x9a8f...1b2c",
  "timestamp": "2026-11-07T10:15:30Z",
  "transaction_type": "deposit",
  "amount": "14.500000000000000000",
  "asset": "ETH",
  "counterparty": "0x123...456",
  "status": "confirmed",
  "block_number": 21102930,
  "metadata": {
    "gas_price_gwei": 22.4,
    "from_wallet_label": "Binance"
  }
}
```

#### Request Schema Fields
- `transaction_id` *(string, required)*: Unique transaction ID.
- `source` *(string, required)*: `banking_plaid`, `crypto_ethereum`, `crypto_bitcoin`, `csv_import`, or `internal`.
- `network` *(string, required)*: Network identifier (e.g., `ethereum_mainnet`, `bitcoin`, `fiat_banking`, `arbitrum`, `polygon`).
- `entity_id` *(string, required)*: Account or wallet address under observation.
- `transaction_hash` *(string, required)*: On-chain transaction hash or banking reference.
- `timestamp` *(string / datetime, required)*: UTC timestamp (ISO-8601). Naive datetimes are coerced to UTC.
- `transaction_type` *(string, required)*: `deposit`, `withdrawal`, `transfer`, or `fee`.
- `amount` *(string / number, required)*: Non-negative amount in native asset units. String format is recommended to prevent floating-point precision loss.
- `asset` *(string, required)*: Symbol (e.g., `ETH`, `BTC`, `USD`, `USDC`).
- `counterparty` *(string, optional)*: Associated sender or recipient entity.
- `status` *(string, optional)*: `pending`, `confirmed`, `posted`, `reversed`, `removed`. Defaults to `confirmed`.
- `block_number` *(integer, optional)*: Ledger block height.
- `metadata` *(object, optional)*: Arbitrary key-value metadata.

#### Response (200 OK)
```json
{
  "status": "success",
  "transaction_id": "tx_eth_001928",
  "entity_id": "0x71C...b847",
  "idempotency_key": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "is_duplicate": false,
  "message": "Transaction ingested successfully."
}
```

#### Duplicate Submission Response (200 OK)
When a transaction with matching idempotency key is resubmitted:
```json
{
  "status": "success",
  "transaction_id": "tx_eth_001928",
  "entity_id": "0x71C...b847",
  "idempotency_key": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "is_duplicate": true,
  "message": "Transaction already recorded."
}
```

---

## 2. Transactions Query

### `GET /transactions`
Retrieve stored transactions with optional filtering and pagination.

#### Query Parameters
- `entity_id` *(string, optional)*: Filter by entity ID.
- `asset` *(string, optional)*: Filter by asset symbol (e.g., `BTC`).
- `status` *(string, optional)*: Filter by status (e.g., `confirmed`).
- `limit` *(integer, optional, default: 50, max: 200)*: Maximum records to return.
- `offset` *(integer, optional, default: 0)*: Number of records to skip.

#### Response (200 OK)
```json
{
  "total": 1,
  "limit": 50,
  "offset": 0,
  "transactions": [
    {
      "transaction_id": "tx_eth_001928",
      "source": "crypto_ethereum",
      "network": "ethereum_mainnet",
      "entity_id": "0x71C...b847",
      "transaction_hash": "0x9a8f...1b2c",
      "timestamp": "2026-11-07T10:15:30Z",
      "transaction_type": "deposit",
      "amount": "14.500000000000000000",
      "asset": "ETH",
      "counterparty": "0x123...456",
      "status": "confirmed",
      "block_number": 21102930,
      "metadata": {
        "gas_price_gwei": 22.4
      }
    }
  ]
}
```

---

### `GET /transactions/{transaction_id}`
Retrieve a single transaction by its unique identifier.

#### Response (200 OK)
Returns the matching transaction object.

#### Response (404 Not Found)
```json
{
  "detail": "Transaction 'tx_nonexistent' not found."
}
```

---

## 3. Real-Time Risk & Anomaly Scoring

### `GET /risk/{entity_id}`
Evaluates or retrieves the latest calibrated anomaly risk score for an entity, including model sub-scores, behavioral feature attributions, and top sequence surprise events.

#### Query Parameters
- `refresh` *(boolean, optional, default: false)*: Force re-computation of risk score on latest transactions rather than returning cached evaluation.

#### Response (200 OK)
```json
{
  "entity_id": "0x71C...b847",
  "ensemble_risk_score": 87.4,
  "if_risk_score": 89.2,
  "transformer_surprise": 80.2,
  "risk_level": "HIGH",
  "insufficient_history": false,
  "cluster_id": 0,
  "event_count": 14,
  "top_features": [
    {
      "feature": "rapid_succession_60s",
      "value": 5.0,
      "direction": "elevated",
      "deviation_score": 2.85
    },
    {
      "feature": "alternation_rate",
      "value": 0.857,
      "direction": "elevated",
      "deviation_score": 2.41
    },
    {
      "feature": "burstiness",
      "value": 0.74,
      "direction": "elevated",
      "deviation_score": 2.12
    }
  ],
  "top_surprising_events": [
    {
      "timestamp": "2026-11-07T10:15:30Z",
      "transaction_type": "deposit",
      "amount": 14.5,
      "nll_surprise": 3.42,
      "event_index": 12
    },
    {
      "timestamp": "2026-11-07T10:15:45Z",
      "transaction_type": "withdrawal",
      "amount": 14.49,
      "nll_surprise": 3.18,
      "event_index": 13
    }
  ],
  "narrative": "Entity displays anomalous transaction behavior (calibrated ensemble risk score: 87.4/100). The entity shows marked deviation from population behavioral norms, driven by elevated rapid_succession_60s (5.000) and elevated alternation_rate (0.857). Sequence modeling flagged 2 events with high negative log-likelihood surprise. Note: Anomaly scores represent statistical deviation from baseline distributions, not confirmed legal fraud determinations.",
  "scored_at": "2026-11-07T10:16:00Z"
}
```

#### Risk Levels
- `CRITICAL`: $\ge 90.0$
- `HIGH`: $\ge 75.0$ and $< 90.0$
- `MEDIUM`: $\ge 50.0$ and $< 75.0$
- `LOW`: $< 50.0$

---

### `GET /risk`
List entities sorted by calibrated ensemble risk score.

#### Query Parameters
- `min_score` *(float, optional, default: 50.0)*: Minimum ensemble risk score filter.
- `limit` *(integer, optional, default: 50)*: Number of entities to return.
- `offset` *(integer, optional, default: 0)*: Pagination offset.

#### Response (200 OK)
```json
{
  "total": 12,
  "limit": 50,
  "offset": 0,
  "risk_records": [
    {
      "entity_id": "0x71C...b847",
      "ensemble_risk_score": 87.4,
      "if_risk_score": 89.2,
      "transformer_surprise": 80.2,
      "risk_level": "HIGH",
      "insufficient_history": false,
      "cluster_id": 0,
      "scored_at": "2026-11-07T10:16:00Z"
    }
  ]
}
```

---

## 4. Webhook Ingestion

### `POST /webhooks/alchemy`
Receives address activity events from Alchemy EVM webhooks.

#### Headers
- `X-Alchemy-Signature`: Hex-encoded HMAC-SHA256 signature of the raw request payload computed using your configured `ALCHEMY_WEBHOOK_SIGNING_KEY`.

#### Example Payload
```json
{
  "webhookId": "wh_12345",
  "id": "whevt_67890",
  "createdAt": "2026-11-07T10:15:30.000Z",
  "type": "ADDRESS_ACTIVITY",
  "event": {
    "network": "ETH_MAINNET",
    "activity": [
      {
        "fromAddress": "0x111...222",
        "toAddress": "0x333...444",
        "value": 2.5,
        "asset": "ETH",
        "hash": "0xabc...def",
        "blockNum": "0x1420f1a",
        "category": "external"
      }
    ]
  }
}
```

#### Response (200 OK)
```json
{
  "status": "success",
  "processed_events": 1,
  "message": "Alchemy webhook processed successfully."
}
```

#### Response (401 Unauthorized)
Returned if `X-Alchemy-Signature` is missing or fails HMAC-SHA256 signature verification.

---

### `POST /webhooks/plaid`
Receives transaction sync events from Plaid Sandbox.

#### Example Payload
```json
{
  "webhook_type": "TRANSACTIONS",
  "webhook_code": "SYNC_UPDATES_AVAILABLE",
  "item_id": "item_plaid_test_01",
  "initial_update_complete": true,
  "historical_update_complete": true
}
```

#### Response (200 OK)
```json
{
  "status": "success",
  "webhook_code": "SYNC_UPDATES_AVAILABLE",
  "message": "Plaid webhook processed."
}
```

---

## 5. Health & Monitoring

### `GET /health`
Verifies service liveness and component readiness.

#### Response (200 OK)
```json
{
  "status": "healthy",
  "database": "connected",
  "models": {
    "isolation_forest": "ready",
    "transformer": "ready",
    "calibrator": "ready"
  },
  "uptime_seconds": 3840.2
}
```

---

### `GET /metrics`
Returns internal telemetry counters in JSON format.

#### Response (200 OK)
```json
{
  "transactions_ingested_total": 1420,
  "duplicate_transactions_total": 18,
  "risk_evaluations_total": 412,
  "high_risk_alerts_total": 29,
  "webhook_requests_total": 85,
  "webhook_failures_total": 0,
  "rate_limit_rejections_total": 0,
  "mean_scoring_latency_ms": 14.2,
  "p95_scoring_latency_ms": 28.6
}
```
