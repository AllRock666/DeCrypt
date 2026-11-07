"""
Repository Pattern for Database Operations in DeCrypt Storage.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from decimal import Decimal
import json
import logging
from sqlalchemy.orm import Session
from sqlalchemy import desc

from decrypt.domain.schema import (
    UnifiedTransaction,
    TransactionSource,
    NetworkType,
    TransactionType,
    TransactionStatus
)
from .models import TransactionRecord, EntityRecord, RiskScoreRecord, IdempotencyRecord

logger = logging.getLogger(__name__)


def to_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class TransactionRepository:
    def __init__(self, session: Session):
        self.session = session

    def save_transaction(self, tx: UnifiedTransaction) -> bool:
        """
        Idempotently stores a UnifiedTransaction and updates entity activity.
        Returns True if newly inserted, False if already present.
        """
        existing = self.session.query(TransactionRecord).filter_by(transaction_id=tx.transaction_id).first()
        if existing:
            # Check for status transition (e.g. pending -> confirmed)
            if existing.status != tx.status.value:
                existing.status = tx.status.value
                self.session.commit()
            return False

        record = TransactionRecord(
            transaction_id=tx.transaction_id,
            source=tx.source.value,
            network=tx.network.value,
            entity_id=tx.entity_id,
            transaction_hash=tx.transaction_hash,
            timestamp_utc=to_utc(tx.timestamp),
            transaction_type=tx.transaction_type.value,
            amount=tx.amount,
            asset=tx.asset,
            counterparty=tx.counterparty,
            status=tx.status.value,
            block_number=tx.block_number,
            metadata_json=json.dumps(tx.metadata, default=str)
        )
        self.session.add(record)

        # Update or create entity
        entity = self.session.query(EntityRecord).filter_by(entity_id=tx.entity_id).first()
        tx_ts = to_utc(tx.timestamp)
        if entity:
            current_last = to_utc(entity.last_seen) or tx_ts
            entity.last_seen = max(current_last, tx_ts)
            entity.total_txs += 1
        else:
            entity = EntityRecord(
                entity_id=tx.entity_id,
                source=tx.source.value,
                network=tx.network.value,
                asset=tx.asset,
                first_seen=tx_ts,
                last_seen=tx_ts,
                total_txs=1
            )
            self.session.add(entity)

        self.session.commit()
        return True

    def get_transaction(self, transaction_id: str) -> Optional[UnifiedTransaction]:
        rec = self.session.query(TransactionRecord).filter_by(transaction_id=transaction_id).first()
        if not rec:
            return None
        return self._to_unified(rec)

    def get_entity_transactions(self, entity_id: str, limit: int = 128) -> List[UnifiedTransaction]:
        records = (
            self.session.query(TransactionRecord)
            .filter_by(entity_id=entity_id)
            .order_by(TransactionRecord.timestamp_utc.asc())
            .all()
        )
        # Keep most recent limit
        if len(records) > limit:
            records = records[-limit:]
        return [self._to_unified(r) for r in records]

    def list_transactions(
        self,
        entity_id: Optional[str] = None,
        asset: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[UnifiedTransaction]:
        q = self.session.query(TransactionRecord)
        if entity_id:
            q = q.filter(TransactionRecord.entity_id == entity_id)
        if asset:
            q = q.filter(TransactionRecord.asset == asset.upper())
        if source:
            q = q.filter(TransactionRecord.source == source)
        records = q.order_by(desc(TransactionRecord.timestamp_utc)).offset(offset).limit(limit).all()
        return [self._to_unified(r) for r in records]

    def save_risk_score(
        self,
        entity_id: str,
        if_score: float,
        tf_surprise: float,
        ensemble_risk_score: float,
        percentile_rank: float,
        insufficient_history: bool,
        top_features: Optional[List[Dict[str, Any]]] = None,
        top_events: Optional[List[Dict[str, Any]]] = None,
        explanation: Optional[str] = None,
        model_version: str = "v2.0-ensemble"
    ) -> RiskScoreRecord:
        record = RiskScoreRecord(
            entity_id=entity_id,
            if_score=float(if_score),
            tf_surprise=float(tf_surprise),
            ensemble_risk_score=float(ensemble_risk_score),
            percentile_rank=float(percentile_rank),
            insufficient_history=insufficient_history,
            model_version=model_version,
            top_features_json=json.dumps(top_features or []),
            top_events_json=json.dumps(top_events or []),
            explanation=explanation
        )
        self.session.add(record)

        # Update entity current_risk_score
        entity = self.session.query(EntityRecord).filter_by(entity_id=entity_id).first()
        if entity:
            entity.current_risk_score = float(ensemble_risk_score)

        self.session.commit()
        return record

    def get_latest_risk_score(self, entity_id: str) -> Optional[Dict[str, Any]]:
        record = (
            self.session.query(RiskScoreRecord)
            .filter_by(entity_id=entity_id)
            .order_by(desc(RiskScoreRecord.timestamp_utc))
            .first()
        )
        if not record:
            return None
        return {
            "entity_id": record.entity_id,
            "timestamp": record.timestamp_utc.isoformat(),
            "if_risk_score": record.if_score,
            "transformer_surprise": record.tf_surprise,
            "ensemble_risk_score": record.ensemble_risk_score,
            "percentile_rank": record.percentile_rank,
            "insufficient_history": record.insufficient_history,
            "model_version": record.model_version,
            "top_features": json.loads(record.top_features_json or "[]"),
            "top_events": json.loads(record.top_events_json or "[]"),
            "explanation": record.explanation
        }

    def list_high_risk_entities(self, threshold: float = 90.0, limit: int = 50) -> List[Dict[str, Any]]:
        entities = (
            self.session.query(EntityRecord)
            .filter(EntityRecord.current_risk_score >= threshold)
            .order_by(desc(EntityRecord.current_risk_score))
            .limit(limit)
            .all()
        )
        return [
            {
                "entity_id": e.entity_id,
                "source": e.source,
                "network": e.network,
                "asset": e.asset,
                "total_txs": e.total_txs,
                "current_risk_score": e.current_risk_score,
                "last_seen": e.last_seen.isoformat() if e.last_seen else None
            }
            for e in entities
        ]

    def _to_unified(self, r: TransactionRecord) -> UnifiedTransaction:
        meta = json.loads(r.metadata_json or "{}")
        return UnifiedTransaction(
            transaction_id=r.transaction_id,
            source=TransactionSource(r.source),
            network=NetworkType(r.network),
            entity_id=r.entity_id,
            transaction_hash=r.transaction_hash,
            timestamp=r.timestamp_utc,
            transaction_type=TransactionType(r.transaction_type),
            amount=Decimal(str(r.amount)),
            asset=r.asset,
            counterparty=r.counterparty,
            status=TransactionStatus(r.status),
            block_number=r.block_number,
            metadata=meta
        )
