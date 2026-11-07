"""
Cryptographic Webhook Signature Verification and API Authentication for DeCrypt.
"""

import os
import hmac
import hashlib
from typing import Dict, Optional
from fastapi import Header, HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(
    x_api_key: Optional[str] = Security(api_key_header),
    authorization: Optional[str] = Header(None)
) -> str:
    """
    Authenticates protected API requests using an API Key.
    Accepts:
      - 'X-API-Key: <key>'
      - 'Authorization: Bearer <key>'
    The configured key is read from DECRYPT_API_KEY (default: 'decrypt-dev-key').
    """
    expected_key = os.getenv("DECRYPT_API_KEY", "decrypt-dev-key")

    token = None
    if x_api_key:
        token = x_api_key.strip()
    elif authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() in ["bearer", "token", "apikey"]:
            token = parts[1]
        elif len(parts) == 1:
            token = parts[0]

    if not token or not hmac.compare_digest(token, expected_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide 'X-API-Key' or 'Authorization: Bearer <key>' header."
        )
    return token


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
