from decimal import Decimal
from datetime import datetime, timezone
import pytest
from decrypt.domain.schema import (
    UnifiedTransaction,
    TransactionSource,
    NetworkType,
    TransactionType,
    TransactionStatus
)


def test_unified_transaction_decimal_precision():
    """Verify that Decimal amounts preserve precision without IEEE 754 drift."""
    tx = UnifiedTransaction(
        transaction_id="tx_test_001",
        source=TransactionSource.CRYPTO_BITCOIN,
        network=NetworkType.BITCOIN,
        entity_id="1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
        transaction_hash="hash_001",
        timestamp=datetime.now(timezone.utc),
        transaction_type=TransactionType.DEPOSIT,
        amount=Decimal("0.00000001"),  # 1 Satoshi
        asset="BTC",
        status=TransactionStatus.CONFIRMED
    )
    assert tx.amount == Decimal("0.00000001")
    assert format(tx.amount, 'f') == "0.00000001"


def test_unified_transaction_utc_normalization():
    """Verify naive datetimes and ISO strings are converted strictly to UTC."""
    # From ISO string with Z
    tx1 = UnifiedTransaction(
        transaction_id="tx_test_002",
        source=TransactionSource.BANKING_PLAID,
        network=NetworkType.FIAT_BANKING,
        entity_id="acc_123",
        transaction_hash="h1",
        timestamp="2026-10-31T15:30:00Z",
        transaction_type=TransactionType.WITHDRAWAL,
        amount=Decimal("150.00"),
        asset="USD"
    )
    assert tx1.timestamp.tzinfo == timezone.utc
    assert tx1.timestamp.hour == 15

    # From legacy DD-MM-YYYY HH:MM string
    tx2 = UnifiedTransaction(
        transaction_id="tx_test_003",
        source=TransactionSource.CSV_LEGACY,
        network=NetworkType.BITCOIN,
        entity_id="addr_1",
        transaction_hash="h2",
        timestamp="15-05-2023 10:00",
        transaction_type=TransactionType.DEPOSIT,
        amount=Decimal("10.0"),
        asset="BTC"
    )
    assert tx2.timestamp.tzinfo == timezone.utc
    assert tx2.timestamp.day == 15
    assert tx2.timestamp.month == 5


def test_blockchain_wallet_event_decomposition():
    """
    Verify that multi-input/output transactions are decomposed relative
    to the watched wallet's net balance change.
    """
    watched_address = "1TargetWallet"

    # Case A: Net deposit (received 10 BTC, spent 2 BTC -> Net +8 BTC deposit)
    tx_dep = UnifiedTransaction.from_blockchain_event(
        entity_address=watched_address,
        tx_hash="tx_btc_01",
        network=NetworkType.BITCOIN,
        source=TransactionSource.CRYPTO_BITCOIN,
        timestamp=datetime.now(timezone.utc),
        total_received=Decimal("10.0"),
        total_spent=Decimal("2.0"),
        asset="BTC"
    )
    assert tx_dep.transaction_type == TransactionType.DEPOSIT
    assert tx_dep.amount == Decimal("8.0")

    # Case B: Net withdrawal (spent 5 BTC, received 0.5 BTC change -> Net -4.5 BTC withdrawal)
    tx_wd = UnifiedTransaction.from_blockchain_event(
        entity_address=watched_address,
        tx_hash="tx_btc_02",
        network=NetworkType.BITCOIN,
        source=TransactionSource.CRYPTO_BITCOIN,
        timestamp=datetime.now(timezone.utc),
        total_received=Decimal("0.5"),
        total_spent=Decimal("5.0"),
        asset="BTC"
    )
    assert tx_wd.transaction_type == TransactionType.WITHDRAWAL
    assert tx_wd.amount == Decimal("4.5")


def test_legacy_csv_row_conversion():
    """Verify bidirectional conversion with legacy 5-column representation."""
    tx = UnifiedTransaction.from_legacy_csv_row(
        address_id="addr_legacy",
        tx_hash="hash_legacy",
        timestamp_str="01-01-2023 12:00",
        tx_type_raw="deposit",
        amount_raw=50.25,
        asset="BTC"
    )
    assert tx.transaction_type == TransactionType.DEPOSIT
    assert tx.amount == Decimal("50.25")

    leg_dict = tx.to_legacy_dict()
    assert leg_dict["address_id"] == "addr_legacy"
    assert leg_dict["type"] == 1
    assert leg_dict["amount"] == 50.25
