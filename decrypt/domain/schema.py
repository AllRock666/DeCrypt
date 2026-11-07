"""
Canonical Unified Transaction Schema for DeCrypt.

Provides a unified, provider-agnostic representation for both traditional banking
and multi-chain cryptocurrency transactions. Enforces UTC normalization, Decimal
precision, asset isolation, and wallet-level blockchain event decomposition.
"""

from __future__ import annotations
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import hashlib
import json
from pydantic import BaseModel, Field, field_validator, model_validator


class TransactionSource(str, Enum):
    BANKING_PLAID = "banking_plaid"
    CRYPTO_EVM = "crypto_evm"
    CRYPTO_BITCOIN = "crypto_bitcoin"
    CSV_LEGACY = "csv_legacy"
    API_DIRECT = "api_direct"


class NetworkType(str, Enum):
    BITCOIN = "bitcoin"
    BITCOIN_TESTNET = "bitcoin_testnet"
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    BASE = "base"
    SEPOLIA = "sepolia"
    FIAT_BANKING = "fiat_banking"
    SIMULATED = "simulated"


class TransactionType(str, Enum):
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"


class TransactionStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    POSTED = "posted"
    REVERSED = "reversed"
    REMOVED = "removed"
    FAILED = "failed"


class UnifiedTransaction(BaseModel):
    """
    Canonical representation of a financial or blockchain transaction.
    """
    transaction_id: str = Field(
        ...,
        description="Deterministic globally unique transaction identifier (provider:network:hash:index)"
    )
    source: TransactionSource = Field(..., description="Provider source of transaction data")
    network: NetworkType = Field(..., description="Underlying settlement network or rail")
    entity_id: str = Field(..., description="Subject wallet address or account identifier")
    transaction_hash: str = Field(..., description="On-chain hash or banking transaction ID")
    timestamp: datetime = Field(..., description="Transaction execution timestamp strictly in UTC")
    transaction_type: TransactionType = Field(..., description="Normalized direction of funds")
    amount: Decimal = Field(..., description="Non-negative transaction amount with exact precision")
    asset: str = Field(..., description="Asset or currency symbol (e.g. BTC, ETH, USDC, USD, INR)")
    counterparty: Optional[str] = Field(None, description="Counterparty address, merchant, or institution")
    status: TransactionStatus = Field(default=TransactionStatus.CONFIRMED, description="Settlement state")
    block_number: Optional[int] = Field(None, description="Blockchain block number or bank ledger index")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Raw provider payload and audit attributes")

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc_timestamp(cls, v: Any) -> datetime:
        if isinstance(v, str):
            # Parse ISO formats or timestamps
            try:
                dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                # Fallback for DD-MM-YYYY HH:MM
                dt = datetime.strptime(v, "%d-%m-%Y %H:%M")
        elif isinstance(v, (int, float)):
            dt = datetime.fromtimestamp(v, tz=timezone.utc)
        elif isinstance(v, datetime):
            dt = v
        else:
            raise ValueError(f"Unsupported timestamp format: {v}")

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt

    @field_validator("amount", mode="before")
    @classmethod
    def ensure_decimal_amount(cls, v: Any) -> Decimal:
        if isinstance(v, Decimal):
            d = v
        else:
            d = Decimal(str(v))
        if d < 0:
            d = abs(d)
        return d

    @field_validator("asset")
    @classmethod
    def normalize_asset(cls, v: str) -> str:
        return v.upper().strip()

    @classmethod
    def generate_id(cls, source: str, network: str, tx_hash: str, entity_id: str, index: int = 0) -> str:
        """
        Generates a deterministic unique ID for deduplication and idempotency.
        """
        raw = f"{source}:{network}:{tx_hash}:{entity_id}:{index}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"tx_{digest}"

    @classmethod
    def from_blockchain_event(
        cls,
        entity_address: str,
        tx_hash: str,
        network: NetworkType,
        source: TransactionSource,
        timestamp: datetime | int | str,
        total_received: Decimal | float | str,
        total_spent: Decimal | float | str,
        asset: str = "BTC",
        counterparty: Optional[str] = None,
        status: TransactionStatus = TransactionStatus.CONFIRMED,
        block_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        index: int = 0
    ) -> Optional["UnifiedTransaction"]:
        """
        Decomposes a multi-input/multi-output blockchain transaction relative to
        a specific watched wallet address. Prevents entire multi-input transactions
        from being erroneously treated as single massive withdrawals.
        """
        rec = Decimal(str(total_received))
        spent = Decimal(str(total_spent))
        net_delta = rec - spent

        if net_delta == 0:
            # Self-transfer or 0-balance change
            tx_type = TransactionType.TRANSFER
            amt = rec if rec > 0 else spent
        elif net_delta > 0:
            tx_type = TransactionType.DEPOSIT
            amt = net_delta
        else:
            tx_type = TransactionType.WITHDRAWAL
            amt = abs(net_delta)

        tx_id = cls.generate_id(source.value, network.value, tx_hash, entity_address, index)

        return cls(
            transaction_id=tx_id,
            source=source,
            network=network,
            entity_id=entity_address,
            transaction_hash=tx_hash,
            timestamp=timestamp,  # validator ensures UTC
            transaction_type=tx_type,
            amount=amt,
            asset=asset,
            counterparty=counterparty,
            status=status,
            block_number=block_number,
            metadata=metadata or {}
        )

    @classmethod
    def from_legacy_csv_row(
        cls,
        address_id: str,
        tx_hash: str,
        timestamp_str: str,
        tx_type_raw: int | str,
        amount_raw: float | str,
        asset: str = "BTC"
    ) -> "UnifiedTransaction":
        """
        Backward-compatibility loader for legacy 5-column CSV rows.
        """
        # Determine direction
        if isinstance(tx_type_raw, str):
            t_str = tx_type_raw.lower().strip()
            tx_type = TransactionType.DEPOSIT if t_str == "deposit" or t_str == "1" else TransactionType.WITHDRAWAL
        else:
            tx_type = TransactionType.DEPOSIT if int(tx_type_raw) == 1 else TransactionType.WITHDRAWAL

        amt = abs(Decimal(str(amount_raw)))
        tx_id = cls.generate_id("csv_legacy", "bitcoin", str(tx_hash), str(address_id), 0)

        return cls(
            transaction_id=tx_id,
            source=TransactionSource.CSV_LEGACY,
            network=NetworkType.BITCOIN,
            entity_id=str(address_id),
            transaction_hash=str(tx_hash),
            timestamp=timestamp_str,
            transaction_type=tx_type,
            amount=amt,
            asset=asset,
            status=TransactionStatus.CONFIRMED,
            metadata={"legacy_source": True}
        )

    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Converts canonical transaction back to legacy 5-column representation for models.
        """
        type_int = 1 if self.transaction_type == TransactionType.DEPOSIT else -1
        # In legacy format, amount is signed: deposits positive, withdrawals negative
        signed_amount = float(self.amount) if type_int == 1 else -float(self.amount)
        return {
            "address_id": self.entity_id,
            "hash": self.transaction_hash,
            "timestamp": self.timestamp.strftime("%d-%m-%Y %H:%M"),
            "type": type_int,
            "amount": signed_amount
        }
