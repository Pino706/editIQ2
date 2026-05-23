"""Runtime configuration for external integrations."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    tiktok_client_key: str = _env("TIKTOK_CLIENT_KEY")
    tiktok_client_secret: str = _env("TIKTOK_CLIENT_SECRET")
    tiktok_redirect_uri: str = _env(
        "TIKTOK_REDIRECT_URI",
        "http://127.0.0.1:8000/api/tiktok/callback",
    )
    tiktok_scopes: str = _env(
        "TIKTOK_SCOPES",
        "user.info.basic,user.info.profile,user.info.stats,video.list",
    )
    token_encryption_key: str = _env("TOKEN_ENCRYPTION_KEY")
    ollama_base_url: str = _env("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = _env("OLLAMA_MODEL", "llama3:latest")

    @property
    def tiktok_configured(self) -> bool:
        return bool(self.tiktok_client_key and self.tiktok_client_secret)


settings = Settings()
