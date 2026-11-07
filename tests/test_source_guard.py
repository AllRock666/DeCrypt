"""
Unit tests for RealTimeScorer source guard and domain calibration.
"""

from decimal import Decimal
from datetime import datetime, timezone, timedelta
import numpy as np
import pytest

from decrypt.domain.schema import (
    UnifiedTransaction,
    TransactionSource,
    NetworkType,
    TransactionType
)
from decrypt.pipeline.realtime_scorer import RealTimeScorer


def test_source_guard_bitcoin_vs_uncalibrated_source():
    """
    Verify that Bitcoin transactions are scored successfully, while uncalibrated
    sources return 'not calibrated for this source' status until calibrated.
    """
    scorer = RealTimeScorer()
    now = datetime.now(timezone.utc)

    # 1. Bitcoin transactions (trained domain)
    btc_history = [
        UnifiedTransaction(
            transaction_id=f"tx_btc_{i}",
            source=TransactionSource.CRYPTO_BITCOIN,
            network=NetworkType.BITCOIN,
            entity_id="addr_btc_1",
            transaction_hash=f"h_btc_{i}",
            timestamp=now + timedelta(minutes=i*15),
            transaction_type=TransactionType.DEPOSIT if i % 2 == 0 else TransactionType.WITHDRAWAL,
            amount=Decimal("1.5"),
            asset="BTC"
        )
        for i in range(4)
    ]
    res_btc = scorer.score_entity("addr_btc_1", btc_history)
    assert res_btc["status"] == "success"
    assert res_btc["ensemble_risk_score"] is not None
    assert 0.0 <= res_btc["ensemble_risk_score"] <= 100.0

    # 2. Banking Plaid transactions (uncalibrated source)
    plaid_history = [
        UnifiedTransaction(
            transaction_id=f"tx_plaid_{i}",
            source=TransactionSource.BANKING_PLAID,
            network=NetworkType.FIAT_BANKING,
            entity_id="acc_plaid_1",
            transaction_hash=f"h_plaid_{i}",
            timestamp=now + timedelta(hours=i),
            transaction_type=TransactionType.DEPOSIT if i % 2 == 0 else TransactionType.WITHDRAWAL,
            amount=Decimal("250.00"),
            asset="USD"
        )
        for i in range(4)
    ]
    res_plaid = scorer.score_entity("acc_plaid_1", plaid_history)
    assert res_plaid["status"] == "not calibrated for this source"
    assert res_plaid["ensemble_risk_score"] is None
    assert res_plaid["training_domain"] == "crypto_bitcoin"
    assert "require a fitted per-source reference distribution" in res_plaid["explanation"]

    # 3. Register per-source reference distribution for Plaid
    mock_if_ref = np.linspace(10.0, 90.0, 50)
    mock_tf_ref = np.linspace(0.5, 4.0, 50)
    scorer.register_source_calibration("banking_plaid", mock_if_ref, mock_tf_ref)

    assert scorer.is_source_calibrated("banking_plaid") is True

    # 4. Now scoring Plaid succeeds
    res_plaid_calibrated = scorer.score_entity("acc_plaid_1", plaid_history)
    assert res_plaid_calibrated["status"] == "success"
    assert res_plaid_calibrated["ensemble_risk_score"] is not None
    assert 0.0 <= res_plaid_calibrated["ensemble_risk_score"] <= 100.0


def test_2026_dated_transaction_bounded_features_and_temporal_drift_warning():
    """
    Verify that 2026-dated transactions do not produce unbounded or out-of-range features
    (due to removal of global-calendar recency features in favor of entity-relative activity_centroid),
    and verify that scoring a 2026 transaction triggers a temporal drift warning.
    """
    from decrypt.pipeline.incremental import IncrementalFeatureExtractor

    extractor = IncrementalFeatureExtractor()
    t0 = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 6, 1, 10, 30, tzinfo=timezone.utc)
    t2 = datetime(2026, 6, 1, 11, 15, tzinfo=timezone.utc)

    history_2026 = [
        UnifiedTransaction(
            transaction_id="tx_2026_1",
            source=TransactionSource.CRYPTO_BITCOIN,
            network=NetworkType.BITCOIN,
            entity_id="addr_2026",
            transaction_hash="h_2026_1",
            timestamp=t0,
            transaction_type=TransactionType.DEPOSIT,
            amount=Decimal("5.0"),
            asset="BTC"
        ),
        UnifiedTransaction(
            transaction_id="tx_2026_2",
            source=TransactionSource.CRYPTO_BITCOIN,
            network=NetworkType.BITCOIN,
            entity_id="addr_2026",
            transaction_hash="h_2026_2",
            timestamp=t1,
            transaction_type=TransactionType.WITHDRAWAL,
            amount=Decimal("4.8"),
            asset="BTC"
        ),
        UnifiedTransaction(
            transaction_id="tx_2026_3",
            source=TransactionSource.CRYPTO_BITCOIN,
            network=NetworkType.BITCOIN,
            entity_id="addr_2026",
            transaction_hash="h_2026_3",
            timestamp=t2,
            transaction_type=TransactionType.DEPOSIT,
            amount=Decimal("2.0"),
            asset="BTC"
        )
    ]

    # 1. Feature extraction check: bounded entity-relative features
    feats = extractor.compute_features(history_2026)
    assert "activity_centroid" in feats
    assert 0.0 <= feats["activity_centroid"] <= 1.0
    assert "days_since_first" not in feats
    assert "days_until_end" not in feats
    for k, v in feats.items():
        assert not np.isnan(v), f"Feature {k} is NaN"
        assert not np.isinf(v), f"Feature {k} is Inf"

    # 2. Scorer check: temporal drift warning
    scorer = RealTimeScorer()
    res = scorer.score_entity("addr_2026", history_2026)
    assert res["status"] == "success"
    assert res["temporal_drift_warning"] is True
    assert 0.0 <= res["ensemble_risk_score"] <= 100.0
