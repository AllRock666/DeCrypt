"""
Base Provider Adapter Interface for Multi-Source Ingestion in DeCrypt.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from decrypt.domain.schema import UnifiedTransaction


class BaseProviderAdapter(ABC):
    """
    Abstract interface for provider ingestion adapters (Plaid, Alchemy, Esplora, etc.).
    """

    def __init__(self, api_key: Optional[str] = None, webhook_secret: Optional[str] = None):
        self.api_key = api_key
        self.webhook_secret = webhook_secret

    @abstractmethod
    def normalize_transaction(self, raw_data: Dict[str, Any], entity_id: str) -> UnifiedTransaction:
        """
        Transforms raw provider payload into a canonical UnifiedTransaction.
        """
        pass

    @abstractmethod
    def verify_webhook(self, headers: Dict[str, str], body_bytes: bytes) -> bool:
        """
        Validates cryptographic authenticity of an incoming provider webhook.
        """
        pass

    @abstractmethod
    def fetch_transactions(
        self,
        entity_id: str,
        limit: int = 100,
        cursor: Optional[str] = None,
        **kwargs
    ) -> List[UnifiedTransaction]:
        """
        Fetches historical transactions for an account or wallet.
        """
        pass
