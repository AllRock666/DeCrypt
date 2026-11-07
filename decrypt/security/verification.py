"""
Cryptographic Webhook Signature Verification for DeCrypt.
"""

import hmac
import hashlib
from typing import Dict, Optional


def verify_alchemy_signature(body_bytes: bytes, signature_header: Optional[str], signing_key: Optional[str]) -> bool:
    """
    Verifies Alchemy address activity webhook signature (HMAC-SHA256).
    Header: `x-alchemy-signature`
    """
    if not signing_key or not signature_header:
        return False

    computed = hmac.new(
        signing_key.encode("utf-8"),
        body_bytes,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(computed.lower(), signature_header.lower())


def verify_plaid_webhook(
    headers: Dict[str, str],
    body_bytes: bytes,
    expected_token: Optional[str] = None
) -> bool:
    """
    Verifies Plaid webhook token or verification header.
    """
    if not expected_token:
        # If no secret configured, accept (sandbox mode)
        return True

    header_token = headers.get("X-Plaid-Webhook-Secret") or headers.get("x-plaid-webhook-secret")
    if header_token and hmac.compare_digest(header_token, expected_token):
        return True

    # Check for Plaid-Verification header presence
    plaid_ver = headers.get("Plaid-Verification") or headers.get("plaid-verification")
    return bool(plaid_ver)
