"""
Sensitive Information Redaction and Sanitization for DeCrypt.
"""

import re
from typing import Dict, Any

SENSITIVE_KEYS = {
    "authorization",
    "password",
    "secret",
    "api_key",
    "x-alchemy-signature",
    "plaid-verification",
    "private_key",
    "seed_phrase",
    "access_token",
    "webhook_secret"
}


def redact_sensitive_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively masks sensitive values in dictionaries before logging or JSON serialization.
    """
    redacted = {}
    for k, v in data.items():
        k_lower = str(k).lower().replace("-", "_")
        if any(sens in k_lower for sens in SENSITIVE_KEYS):
            redacted[k] = "********[REDACTED]"
        elif isinstance(v, dict):
            redacted[k] = redact_sensitive_dict(v)
        elif isinstance(v, list):
            redacted[k] = [redact_sensitive_dict(item) if isinstance(item, dict) else item for item in v]
        else:
            redacted[k] = v
    return redacted


def sanitize_log_message(msg: str) -> str:
    """
    Masks API keys and private tokens in string log lines.
    """
    # Mask hex private keys (64 characters)
    msg = re.sub(r'\b(0x)?[a-fA-F0-9]{64}\b', '[REDACTED_HEX_KEY]', msg)
    # Mask common JWT / Bearer patterns
    msg = re.sub(r'Bearer\s+[A-Za-z0-9\-._~+/]+=*', 'Bearer [REDACTED_TOKEN]', msg)
    return msg
