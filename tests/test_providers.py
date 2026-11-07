import hmac
import hashlib
from decimal import Decimal
import pytest

from decrypt.domain.schema import TransactionType, TransactionStatus, NetworkType
from decrypt.providers.plaid_adapter import PlaidAdapter
from decrypt.providers.alchemy_adapter import AlchemyAdapter
from decrypt.providers.bitcoin_adapter import BitcoinAdapter


def test_plaid_direction_normalization():
    """
    Verify Plaid's counter-intuitive sign convention:
    Positive amount in Plaid = Withdrawal (expense)
    Negative amount in Plaid = Deposit (income)
    """
    adapter = PlaidAdapter()

    # Positive Plaid amount -> Withdrawal
    raw_exp = {
        "transaction_id": "plaid_01",
        "account_id": "acc_01",
        "amount": 75.50,
        "iso_currency_code": "USD",
        "date": "2026-11-01"
    }
    tx_exp = adapter.normalize_transaction(raw_exp)
    assert tx_exp.transaction_type == TransactionType.WITHDRAWAL
    assert tx_exp.amount == Decimal("75.50")
    assert tx_exp.asset == "USD"

    # Negative Plaid amount -> Deposit
    raw_inc = {
        "transaction_id": "plaid_02",
        "account_id": "acc_01",
        "amount": -1200.00,
        "iso_currency_code": "USD",
        "date": "2026-11-01"
    }
    tx_inc = adapter.normalize_transaction(raw_inc)
    assert tx_inc.transaction_type == TransactionType.DEPOSIT
    assert tx_inc.amount == Decimal("1200.00")


def test_alchemy_adapter_normalization_and_webhook():
    """Verify Alchemy EVM normalization and HMAC signature verification."""
    signing_key = "test_signing_key_secret_123"
    adapter = AlchemyAdapter(webhook_signing_key=signing_key)

    # 1. Normalization
    entity = "0xMyWalletAddress"
    raw_transfer = {
        "hash": "0x123abc",
        "from": "0xSenderAddress",
        "to": entity,
        "value": 2.5,
        "asset": "ETH",
        "category": "external",
        "blockNum": "0x100"
    }
    tx = adapter.normalize_transaction(raw_transfer, entity)
    assert tx.transaction_type == TransactionType.DEPOSIT
    assert tx.amount == Decimal("2.5")
    assert tx.counterparty == "0xsenderaddress"

    # 2. Webhook Signature Verification
    body = b'{"event": {"activity": []}}'
    valid_sig = hmac.new(signing_key.encode("utf-8"), body, hashlib.sha256).hexdigest()
    assert adapter.verify_webhook({"x-alchemy-signature": valid_sig}, body) is True
    assert adapter.verify_webhook({"x-alchemy-signature": "invalid_sig"}, body) is False


def test_bitcoin_adapter_utxo_decomposition():
    """Verify Bitcoin Esplora UTXO calculation and Satoshi conversion."""
    adapter = BitcoinAdapter()
    watched_addr = "1MyBitcoinColdStorage"

    raw_tx = {
        "txid": "btc_txid_999",
        "vin": [
            {
                "prevout": {
                    "scriptpubkey_address": watched_addr,
                    "value": 100000000  # 1.0 BTC spent
                }
            }
        ],
        "vout": [
            {
                "scriptpubkey_address": "1ExchangeDeposit",
                "value": 90000000  # 0.9 BTC sent out
            },
            {
                "scriptpubkey_address": watched_addr,
                "value": 9990000  # 0.0999 BTC change back
            }
        ],
        "status": {
            "confirmed": True,
            "block_height": 800000,
            "block_time": 1700000000
        }
    }
    # Net: spent 100,000,000 sats, received 9,990,000 sats -> Net -90,010,000 sats (0.9001 BTC withdrawal)
    tx = adapter.normalize_transaction(raw_tx, watched_addr)
    assert tx.transaction_type == TransactionType.WITHDRAWAL
    assert tx.amount == Decimal("0.9001")
    assert tx.status == TransactionStatus.CONFIRMED
    assert tx.asset == "BTC"
