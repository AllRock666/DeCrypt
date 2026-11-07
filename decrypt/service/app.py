"""
FastAPI Ingestion Gateway, Real-Time Scoring, and Webhook Service for DeCrypt.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from decimal import Decimal
import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Header, Request, status, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from decrypt.domain.schema import (
    UnifiedTransaction,
    TransactionSource,
    NetworkType,
    TransactionType,
    TransactionStatus
)
from decrypt.storage.database import init_db, get_db
from decrypt.storage.repository import TransactionRepository
from decrypt.pipeline.incremental import EntityStateManager
from decrypt.pipeline.realtime_scorer import RealTimeScorer
from decrypt.providers.plaid_adapter import PlaidAdapter
from decrypt.providers.alchemy_adapter import AlchemyAdapter
from decrypt.security.verification import verify_alchemy_signature, verify_plaid_webhook, verify_api_key
from decrypt.security.redaction import redact_sensitive_dict
from .metrics import metrics

logger = logging.getLogger("decrypt.service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Global in-memory managers
state_manager = EntityStateManager(max_history_per_entity=None)
scorer: Optional[RealTimeScorer] = None
plaid_adapter = PlaidAdapter(
    client_id=os.getenv("PLAID_CLIENT_ID"),
    secret=os.getenv("PLAID_SECRET"),
    environment=os.getenv("PLAID_ENV", "sandbox"),
    webhook_verification_token=os.getenv("PLAID_WEBHOOK_SECRET")
)
alchemy_adapter = AlchemyAdapter(
    api_key=os.getenv("ALCHEMY_API_KEY"),
    webhook_signing_key=os.getenv("ALCHEMY_WEBHOOK_SIGNING_KEY")
)


# Rate Limiting (In-Memory Sliding Window)
RATE_LIMIT_REQUESTS = int(os.getenv("DECRYPT_RATE_LIMIT_RPM", "600"))
request_history: Dict[str, List[float]] = {}


def check_rate_limit(request: Request):
    client_ip = request.client.host if request.client else "unknown_client"
    now = time.time()
    history = request_history.setdefault(client_ip, [])
    # Keep timestamps within the last 60 seconds
    request_history[client_ip] = [t for t in history if now - t < 60.0]

    if len(request_history[client_ip]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({RATE_LIMIT_REQUESTS} req/min). Please back off."
        )
    request_history[client_ip].append(now)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scorer
    logger.info("Initializing DeCrypt Database and Real-Time Pipeline...")
    init_db()
    scorer = RealTimeScorer()
    logger.info("DeCrypt Service initialized successfully.")
    yield
    logger.info("DeCrypt Service shutting down gracefully.")


app = FastAPI(
    title="DeCrypt Anomaly Detection Engine",
    description="Multi-source, real-time transaction anomaly detection and risk intelligence platform.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request and Response Models
class IngestRequest(BaseModel):
    transaction: Optional[UnifiedTransaction] = None
    raw_payload: Optional[Dict[str, Any]] = None
    provider: Optional[str] = Field(None, description="plaid, alchemy, bitcoin, or canonical")
    entity_id: Optional[str] = Field(None, description="Optional entity override if raw_payload is supplied")


class IngestResponse(BaseModel):
    status: str
    transaction_id: str
    entity_id: str
    is_duplicate: bool
    current_risk_score: Optional[float]
    processing_time_ms: float
    disclaimer: str


# Helper background task for async scoring
def process_entity_scoring(entity_id: str, db: Session):
    repo = TransactionRepository(db)
    history = state_manager.get_history(entity_id)
    if not history:
        history = repo.get_entity_transactions(entity_id, limit=128)
        state_manager.load_entity_history(entity_id, history)

    if scorer:
        t0 = time.time()
        score_res = scorer.score_entity(entity_id, history)
        latency_ms = (time.time() - t0) * 1000.0
        metrics.record_inference_latency(latency_ms)

        ens_score = score_res.get("ensemble_risk_score")
        if ens_score is not None and ens_score >= 90.0:
            metrics.record_high_risk_alert()

        if ens_score is not None:
            repo.save_risk_score(
                entity_id=entity_id,
                if_score=score_res["if_risk_score"],
                tf_surprise=score_res["transformer_surprise"],
                ensemble_risk_score=score_res["ensemble_risk_score"],
                percentile_rank=score_res["percentile_rank"],
                insufficient_history=score_res["insufficient_history"],
                top_features=score_res.get("top_features"),
                top_events=score_res.get("top_events"),
                explanation=score_res.get("explanation")
            )


# =====================================================================
# Endpoints
# =====================================================================

@app.post("/transactions/ingest", response_model=IngestResponse, dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
async def ingest_transaction(
    payload: IngestRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Ingests a single transaction either as canonical UnifiedTransaction or via raw provider payload.
    """
    t_start = time.time()
    repo = TransactionRepository(db)

    # 1. Resolve to UnifiedTransaction
    if payload.transaction:
        tx = payload.transaction
    elif payload.raw_payload and payload.provider:
        prov = payload.provider.lower()
        if prov == "plaid":
            tx = plaid_adapter.normalize_transaction(payload.raw_payload, payload.entity_id)
        elif prov == "alchemy":
            tx = alchemy_adapter.normalize_transaction(payload.raw_payload, payload.entity_id or "0xDefault")
        elif prov == "bitcoin":
            from decrypt.providers.bitcoin_adapter import BitcoinAdapter
            btc_adapter = BitcoinAdapter()
            tx = btc_adapter.normalize_transaction(payload.raw_payload, payload.entity_id or "1Default")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {payload.provider}")
    else:
        raise HTTPException(status_code=400, detail="Must provide either 'transaction' or ('raw_payload' and 'provider')")

    # 2. Idempotent Storage
    is_new = repo.save_transaction(tx)
    metrics.record_transaction(newly_inserted=is_new)

    # 3. Update in-memory state
    state_manager.add_transaction(tx)

    # 4. Score immediately or in background
    history = state_manager.get_history(tx.entity_id)
    current_score = None
    if scorer:
        score_res = scorer.score_entity(tx.entity_id, history)
        current_score = score_res["ensemble_risk_score"]
        if score_res["ensemble_risk_score"] >= 90.0:
            metrics.record_high_risk_alert()

        # Save to DB in background
        background_tasks.add_task(process_entity_scoring, tx.entity_id, db)

    elapsed_ms = (time.time() - t_start) * 1000.0

    return IngestResponse(
        status="ingested" if is_new else "duplicate_ignored",
        transaction_id=tx.transaction_id,
        entity_id=tx.entity_id,
        is_duplicate=not is_new,
        current_risk_score=current_score,
        processing_time_ms=round(elapsed_ms, 2),
        disclaimer="Anomaly scores reflect statistical deviance and do not confirm fraudulent intent."
    )


@app.get("/transactions", dependencies=[Depends(check_rate_limit)])
def list_transactions(
    entity_id: Optional[str] = None,
    asset: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Retrieves paginated transactions filtered by entity, asset, or provider source."""
    repo = TransactionRepository(db)
    return repo.list_transactions(entity_id=entity_id, asset=asset, source=source, limit=limit, offset=offset)


@app.get("/transactions/{transaction_id}", dependencies=[Depends(check_rate_limit)])
def get_transaction(transaction_id: str, db: Session = Depends(get_db)):
    """Retrieves a specific transaction by its canonical transaction_id."""
    repo = TransactionRepository(db)
    tx = repo.get_transaction(transaction_id)
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found.")
    return tx


@app.get("/risk/{entity_id}", dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
def get_entity_risk(entity_id: str, db: Session = Depends(get_db)):
    """
    Computes or retrieves real-time multi-view anomaly score and explainability narrative.
    """
    repo = TransactionRepository(db)
    history = state_manager.get_history(entity_id)
    if not history:
        history = repo.get_entity_transactions(entity_id)
        state_manager.load_entity_history(entity_id, history)

    if not history:
        raise HTTPException(status_code=404, detail=f"No transaction history found for entity {entity_id}.")

    if not scorer:
        raise HTTPException(status_code=503, detail="RealTimeScorer is initializing.")

    return scorer.score_entity(entity_id, history)


@app.get("/risk", dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
def list_high_risk_entities(
    threshold: float = 90.0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Lists entities exceeding the specified anomaly risk threshold."""
    repo = TransactionRepository(db)
    return repo.list_high_risk_entities(threshold=threshold, limit=limit)


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Comprehensive service health inspection."""
    db_ok = True
    try:
        db.execute(sqlalchemy.text("SELECT 1"))
    except Exception:
        db_ok = False

    return {
        "status": "healthy" if db_ok and scorer is not None else "degraded",
        "service": "DeCrypt Financial Anomaly Detection Engine",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database_connected": db_ok,
        "models_loaded": scorer is not None,
        "model_version": "v2.0-ensemble"
    }


import sqlalchemy


@app.get("/metrics")
def get_operational_metrics():
    """Returns telemetry on ingestion rates, latencies, duplicates, and alerts."""
    return metrics.get_metrics_snapshot()


@app.post("/webhooks/plaid")
async def handle_plaid_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Processes Plaid webhook notifications (e.g. SYNC_UPDATES_AVAILABLE)."""
    body_bytes = await request.body()
    headers = dict(request.headers)

    verified = plaid_adapter.verify_webhook(headers, body_bytes)
    metrics.record_webhook(verified=verified)
    if not verified:
        logger.warning("Plaid webhook signature verification failed.")
        raise HTTPException(status_code=401, detail="Unauthorized webhook.")

    try:
        body = await request.json()
    except Exception:
        body = {}

    webhook_code = body.get("webhook_code")
    logger.info(f"Received verified Plaid webhook: {webhook_code}")

    if webhook_code == "SYNC_UPDATES_AVAILABLE":
        item_id = body.get("item_id")
        logger.info(f"Triggering sync for Plaid Item: {item_id}")
        # In production, lookup access token for item_id and sync
        access_token = os.getenv("PLAID_SAMPLE_ACCESS_TOKEN", "mock-token")
        txs, _, _ = plaid_adapter.sync_transactions(access_token=access_token)
        repo = TransactionRepository(db)
        for tx in txs:
            if repo.save_transaction(tx):
                state_manager.add_transaction(tx)
                background_tasks.add_task(process_entity_scoring, tx.entity_id, db)

    return {"status": "received", "webhook_code": webhook_code}


@app.post("/webhooks/alchemy")
async def handle_alchemy_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Processes Alchemy address activity webhook notifications with HMAC-SHA256 verification."""
    body_bytes = await request.body()
    sig = request.headers.get("x-alchemy-signature")

    signing_key = os.getenv("ALCHEMY_WEBHOOK_SIGNING_KEY")
    if signing_key:
        verified = verify_alchemy_signature(body_bytes, sig, signing_key)
    else:
        verified = True  # Sandbox / testing mode

    metrics.record_webhook(verified=verified)
    if not verified:
        logger.warning("Alchemy webhook HMAC signature mismatch.")
        raise HTTPException(status_code=401, detail="Invalid Alchemy signature.")

    try:
        body = await request.json()
    except Exception:
        body = {}

    txs = alchemy_adapter.parse_webhook_payload(body)
    logger.info(f"Alchemy webhook verified. Processing {len(txs)} asset transfers.")

    repo = TransactionRepository(db)
    for tx in txs:
        is_new = repo.save_transaction(tx)
        metrics.record_transaction(newly_inserted=is_new)
        state_manager.add_transaction(tx)
        background_tasks.add_task(process_entity_scoring, tx.entity_id, db)

    return {"status": "received", "transfers_processed": len(txs)}
