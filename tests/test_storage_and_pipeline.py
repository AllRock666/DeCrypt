from decimal import Decimal
from datetime import datetime, timezone, timedelta
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from decrypt.domain.schema import (
    UnifiedTransaction,
    TransactionSource,
    NetworkType,
    TransactionType,
    TransactionStatus
)
from decrypt.storage.database import Base
from decrypt.storage.repository import TransactionRepository
from decrypt.pipeline.incremental import EntityStateManager, IncrementalFeatureExtractor
from decrypt.pipeline.realtime_scorer import RealTimeScorer


@pytest.fixture
def in_memory_repo():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield TransactionRepository(session)
    session.close()


def test_idempotent_transaction_saving(in_memory_repo):
    """Verify that saving the exact same transaction twice returns False and does not duplicate."""
    tx = UnifiedTransaction(
        transaction_id="tx_idemp_01",
        source=TransactionSource.BANKING_PLAID,
        network=NetworkType.FIAT_BANKING,
        entity_id="user_alpha",
        transaction_hash="hash_alpha",
        timestamp=datetime.now(timezone.utc),
        transaction_type=TransactionType.DEPOSIT,
        amount=Decimal("500.00"),
        asset="USD"
    )
    # First save: newly inserted
    assert in_memory_repo.save_transaction(tx) is True
    # Second save: duplicate detected
    assert in_memory_repo.save_transaction(tx) is False

    txs = in_memory_repo.get_entity_transactions("user_alpha")
    assert len(txs) == 1
    assert txs[0].amount == Decimal("500.00")


def test_incremental_feature_extraction():
    """Verify that incremental feature calculation produces the expected 23 behavioral features."""
    now = datetime.now(timezone.utc)
    history = [
        UnifiedTransaction(
            transaction_id=f"tx_{i}",
            source=TransactionSource.CRYPTO_BITCOIN,
            network=NetworkType.BITCOIN,
            entity_id="addr_test",
            transaction_hash=f"h_{i}",
            timestamp=now + timedelta(hours=i),
            transaction_type=TransactionType.DEPOSIT if i % 2 == 0 else TransactionType.WITHDRAWAL,
            amount=Decimal("10.0"),
            asset="BTC"
        )
        for i in range(4)
    ]
    extractor = IncrementalFeatureExtractor()
    feats = extractor.compute_features(history)

    assert feats["total_transactions"] == 4.0
    assert feats["total_volume"] == 40.0
    assert feats["net_flow"] == 0.0  # +10 -10 +10 -10
    assert feats["pass_through_ratio"] == 1.0  # min(dep, wd)/max(dep, wd)
    assert feats["direction_switch_count"] == 3.0
    assert feats["alternation_rate"] == 1.0
    assert feats["max_alternating_run_length"] == 4.0


def test_realtime_scorer_short_history_vs_full():
    """Verify scoring logic for entities with <3 events vs >=3 events."""
    scorer = RealTimeScorer()
    now = datetime.now(timezone.utc)

    # 1. Short history (<3 events)
    short_history = [
        UnifiedTransaction(
            transaction_id="tx_s1",
            source=TransactionSource.CRYPTO_BITCOIN,
            network=NetworkType.BITCOIN,
            entity_id="addr_short",
            transaction_hash="h1",
            timestamp=now,
            transaction_type=TransactionType.DEPOSIT,
            amount=Decimal("1.5"),
            asset="BTC"
        )
    ]
    res_short = scorer.score_entity("addr_short", short_history)
    assert res_short["insufficient_history"] is True
    assert res_short["transformer_surprise"] == 0.0
    assert "fewer than 3 transactions" in res_short["explanation"]

    # 2. History with >=3 events
    full_history = short_history + [
        UnifiedTransaction(
            transaction_id=f"tx_s{i}",
            source=TransactionSource.CRYPTO_BITCOIN,
            network=NetworkType.BITCOIN,
            entity_id="addr_short",
            transaction_hash=f"h{i}",
            timestamp=now + timedelta(minutes=i*10),
            transaction_type=TransactionType.WITHDRAWAL if i == 2 else TransactionType.DEPOSIT,
            amount=Decimal("1.45"),
            asset="BTC"
        )
        for i in [2, 3]
    ]
    res_full = scorer.score_entity("addr_short", full_history)
    assert res_full["insufficient_history"] is False
    assert 0.0 <= res_full["ensemble_risk_score"] <= 100.0
    assert len(res_full["top_features"]) > 0
