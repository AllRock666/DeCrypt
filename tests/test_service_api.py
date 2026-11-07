import hmac
import hashlib
from decimal import Decimal
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from decrypt.service.app import app
from decrypt.domain.schema import (
    UnifiedTransaction,
    TransactionSource,
    NetworkType,
    TransactionType,
    TransactionStatus
)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_metrics_endpoints(client):
    """Test /health and /metrics endpoints."""
    r_health = client.get("/health")
    assert r_health.status_code == 200
    data = r_health.json()
    assert data["database_connected"] is True
    assert "status" in data

    r_metrics = client.get("/metrics")
    assert r_metrics.status_code == 200
    m_data = r_metrics.json()
    assert "transactions_received_total" in m_data
    assert "uptime_seconds" in m_data


import uuid

AUTH_HEADERS = {"X-API-Key": "decrypt-dev-key"}


def test_auth_enforcement_on_protected_endpoints(client):
    """Verify that /transactions/ingest, /risk/{id}, and /risk return 401 when API key is missing or invalid."""
    # 1. Ingest without auth
    r_no_auth = client.post("/transactions/ingest", json={"transaction": {}})
    assert r_no_auth.status_code == 401
    assert "Invalid or missing API key" in r_no_auth.json()["detail"]

    # 2. Ingest with invalid auth
    r_bad_auth = client.post("/transactions/ingest", json={"transaction": {}}, headers={"X-API-Key": "wrong-secret"})
    assert r_bad_auth.status_code == 401

    # 3. Risk query without auth
    r_risk_no_auth = client.get("/risk/user_123")
    assert r_risk_no_auth.status_code == 401

    # 4. High-risk list without auth
    r_list_no_auth = client.get("/risk")
    assert r_list_no_auth.status_code == 401


def test_transaction_ingest_and_query_flow(client):
    """Test full cycle: ingest transaction -> query transaction -> score entity."""
    test_uid = uuid.uuid4().hex[:8]
    test_tx_id = f"tx_api_test_{test_uid}"
    test_entity_id = f"user_api_{test_uid}"

    canonical_tx = {
        "transaction_id": test_tx_id,
        "source": "crypto_bitcoin",
        "network": "bitcoin",
        "entity_id": test_entity_id,
        "transaction_hash": f"bank_ref_{test_uid}",
        "timestamp": "2026-11-06T12:00:00Z",
        "transaction_type": "deposit",
        "amount": "1500.50",
        "asset": "BTC",
        "counterparty": "Employer Direct Deposit",
        "status": "confirmed",
        "metadata": {"test": True}
    }

    # 1. Ingest with valid auth
    r_ingest = client.post("/transactions/ingest", json={"transaction": canonical_tx}, headers=AUTH_HEADERS)
    assert r_ingest.status_code == 200
    ing_data = r_ingest.json()
    assert ing_data["transaction_id"] == test_tx_id
    assert ing_data["is_duplicate"] is False

    # 2. Duplicate detection
    r_dup = client.post("/transactions/ingest", json={"transaction": canonical_tx}, headers=AUTH_HEADERS)
    assert r_dup.status_code == 200
    assert r_dup.json()["is_duplicate"] is True

    # 3. Query transaction by ID (public query endpoint)
    r_get = client.get(f"/transactions/{test_tx_id}")
    assert r_get.status_code == 200
    tx_res = r_get.json()
    assert tx_res["entity_id"] == test_entity_id
    assert float(tx_res["amount"]) == 1500.50

    # 4. Query entity risk with valid auth
    r_risk = client.get(f"/risk/{test_entity_id}", headers=AUTH_HEADERS)
    assert r_risk.status_code == 200
    risk_data = r_risk.json()
    assert "ensemble_risk_score" in risk_data
    assert "explanation" in risk_data
    assert "disclaimer" in risk_data


def test_alchemy_webhook_endpoint(client):
    """Test /webhooks/alchemy with mock transfer payload."""
    payload = {
        "event": {
            "activity": [
                {
                    "hash": "0xabc123456789",
                    "fromAddress": "0xSender111",
                    "toAddress": "0xRecipient222",
                    "value": 1.75,
                    "asset": "ETH",
                    "category": "external",
                    "blockNum": "0x123",
                    "blockTimestamp": "2026-11-06T10:00:00.000Z"
                }
            ]
        }
    }
    r = client.post("/webhooks/alchemy", json=payload)
    assert r.status_code == 200
    assert r.json()["transfers_processed"] == 1


def test_plaid_webhook_endpoint(client):
    """Test /webhooks/plaid handling."""
    payload = {
        "webhook_type": "TRANSACTIONS",
        "webhook_code": "SYNC_UPDATES_AVAILABLE",
        "item_id": "item_sandbox_test_id",
        "initial_update_complete": True,
        "historical_update_complete": True
    }
    r = client.post("/webhooks/plaid", json=payload)
    assert r.status_code == 200
    assert r.json()["webhook_code"] == "SYNC_UPDATES_AVAILABLE"
