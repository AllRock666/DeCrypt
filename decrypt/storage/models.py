"""
SQLAlchemy ORM Models for Durable Storage in DeCrypt.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Numeric,
    Boolean,
    Integer,
    Text,
    Float,
    Index
)
from .database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EntityRecord(Base):
    __tablename__ = "entities"

    entity_id = Column(String(128), primary_key=True)
    source = Column(String(64), nullable=False)
    network = Column(String(64), nullable=False)
    asset = Column(String(32), nullable=False, default="BTC")
    first_seen = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    last_seen = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    total_txs = Column(Integer, default=0, nullable=False)
    current_risk_score = Column(Float, nullable=True)
    metadata_json = Column(Text, nullable=True)


class TransactionRecord(Base):
    __tablename__ = "transactions"

    transaction_id = Column(String(128), primary_key=True)
    source = Column(String(64), nullable=False, index=True)
    network = Column(String(64), nullable=False, index=True)
    entity_id = Column(String(128), nullable=False, index=True)
    transaction_hash = Column(String(128), nullable=False, index=True)
    timestamp_utc = Column(DateTime(timezone=True), nullable=False, index=True)
    transaction_type = Column(String(32), nullable=False)
    amount = Column(Numeric(precision=38, scale=18), nullable=False)
    asset = Column(String(32), nullable=False, index=True)
    counterparty = Column(String(128), nullable=True)
    status = Column(String(32), nullable=False, default="confirmed")
    block_number = Column(Integer, nullable=True)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    __table_args__ = (
        Index("idx_entity_timestamp", "entity_id", "timestamp_utc"),
        Index("idx_hash_entity", "transaction_hash", "entity_id"),
    )


class RiskScoreRecord(Base):
    __tablename__ = "risk_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(String(128), nullable=False, index=True)
    timestamp_utc = Column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)
    if_score = Column(Float, nullable=False)
    tf_surprise = Column(Float, nullable=False)
    ensemble_risk_score = Column(Float, nullable=False, index=True)
    percentile_rank = Column(Float, nullable=False)
    insufficient_history = Column(Boolean, default=False, nullable=False)
    model_version = Column(String(64), nullable=False, default="v2.0-ensemble")
    top_features_json = Column(Text, nullable=True)
    top_events_json = Column(Text, nullable=True)
    explanation = Column(Text, nullable=True)


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_keys"

    idempotency_key = Column(String(128), primary_key=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    status = Column(String(32), default="processed", nullable=False)
