"""Small encryption helpers for server-side OAuth token storage."""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet

from app.config import settings


def _fernet() -> Fernet:
    key = settings.token_encryption_key
    if not key:
        raise RuntimeError("TOKEN_ENCRYPTION_KEY is required for TikTok token storage.")
    try:
        return Fernet(key.encode("utf-8"))
    except ValueError:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_token(token: str | None) -> str | None:
    if not token:
        return None
    return _fernet().encrypt(token.encode("utf-8")).decode("utf-8")


def decrypt_token(token: str | None) -> str | None:
    if not token:
        return None
    return _fernet().decrypt(token.encode("utf-8")).decode("utf-8")
