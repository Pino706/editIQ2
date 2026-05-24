"""TikTok OAuth and Display API client."""

from __future__ import annotations

import time
import base64
import hashlib
import hmac
import json
import secrets
from typing import Any
from urllib.parse import urlencode

import httpx

from app import database
from app.config import settings
from app.security import decrypt_token, encrypt_token

AUTHORIZE_URL = "https://www.tiktok.com/v2/auth/authorize/"
TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
USER_INFO_URL = "https://open.tiktokapis.com/v2/user/info/"
VIDEO_LIST_URL = "https://open.tiktokapis.com/v2/video/list/"

USER_FIELDS = ",".join(
    [
        "open_id",
        "union_id",
        "avatar_url",
        "display_name",
        "profile_deep_link",
        "bio_description",
        "is_verified",
        "follower_count",
        "following_count",
        "likes_count",
        "video_count",
    ]
)

VIDEO_FIELDS = ",".join(
    [
        "id",
        "create_time",
        "cover_image_url",
        "share_url",
        "video_description",
        "duration",
        "height",
        "width",
        "title",
        "like_count",
        "comment_count",
        "share_count",
        "view_count",
    ]
)


class TikTokConfigError(RuntimeError):
    pass


def create_oauth_state() -> str:
    payload = {
        "nonce": secrets.token_urlsafe(24),
        "iat": int(time.time()),
    }
    body = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = _b64url(_state_signature(body))
    return f"{body}.{sig}"


def validate_oauth_state(state: str, max_age_seconds: int = 900) -> None:
    try:
        body, sig = state.split(".", 1)
        expected = _b64url(_state_signature(body))
        if not hmac.compare_digest(sig, expected):
            raise ValueError("signature mismatch")
        payload = json.loads(_b64url_decode(body))
        issued_at = int(payload["iat"])
    except Exception as exc:
        raise TikTokConfigError("Invalid OAuth state.") from exc
    if issued_at < int(time.time()) - max_age_seconds:
        raise TikTokConfigError("Expired OAuth state.")


def build_login_url(state: str | None = None) -> str:
    if not settings.tiktok_configured:
        raise TikTokConfigError("TikTok OAuth is not configured.")
    state = state or create_oauth_state()
    database.create_oauth_state(state)
    params = {
        "client_key": settings.tiktok_client_key,
        "scope": settings.tiktok_scopes,
        "response_type": "code",
        "redirect_uri": settings.tiktok_redirect_uri,
        "state": state,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


async def complete_oauth(code: str, state: str) -> dict[str, Any]:
    validate_oauth_state(state)
    if not database.consume_oauth_state(state):
        raise TikTokConfigError("OAuth state was already used or is unknown.")
    token = await _request_token(
        {
            "client_key": settings.tiktok_client_key,
            "client_secret": settings.tiktok_client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": settings.tiktok_redirect_uri,
        }
    )
    profile = await fetch_profile(token["access_token"])
    _store_account(profile, token)
    videos = await fetch_recent_videos(token["access_token"])
    database.upsert_tiktok_videos(profile["open_id"], videos)
    return {"profile": profile, "videos": videos}


def _state_signature(body: str) -> bytes:
    if not settings.token_encryption_key:
        raise TikTokConfigError("TOKEN_ENCRYPTION_KEY is required before connecting TikTok.")
    return hmac.new(
        settings.token_encryption_key.encode("utf-8"),
        body.encode("utf-8"),
        hashlib.sha256,
    ).digest()


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> str:
    padded = data + ("=" * (-len(data) % 4))
    return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")


async def refresh_connected_account() -> dict[str, Any]:
    account = database.get_connected_tiktok_account()
    if not account:
        return {"connected": False}
    access_token = await get_valid_access_token(account)
    profile = await fetch_profile(access_token)
    token_payload = {
        "scope": account.get("scopes"),
        "access_expires_at": account.get("access_expires_at"),
        "refresh_expires_at": account.get("refresh_expires_at"),
    }
    _store_account(profile, token_payload, access_token=access_token)
    videos = await fetch_recent_videos(access_token)
    database.upsert_tiktok_videos(profile["open_id"], videos)
    return {"connected": True, "profile": profile, "videos": videos}


async def get_valid_access_token(account: dict[str, Any]) -> str:
    access_expires_at = int(account.get("access_expires_at") or 0)
    access_token = decrypt_token(account.get("access_token"))
    if access_token and access_expires_at > int(time.time()) + 120:
        return access_token

    refresh_token = decrypt_token(account.get("refresh_token"))
    if not refresh_token:
        raise TikTokConfigError("TikTok refresh token is missing.")
    token = await _request_token(
        {
            "client_key": settings.tiktok_client_key,
            "client_secret": settings.tiktok_client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
    )
    open_id = account["open_id"]
    database.update_tiktok_tokens(
        open_id=open_id,
        encrypted_access_token=encrypt_token(token.get("access_token")) or "",
        encrypted_refresh_token=encrypt_token(token.get("refresh_token")),
        access_expires_at=token.get("access_expires_at"),
        refresh_expires_at=token.get("refresh_expires_at"),
        scopes=token.get("scope"),
    )
    return token["access_token"]


async def fetch_profile(access_token: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        res = await client.get(
            USER_INFO_URL,
            params={"fields": USER_FIELDS},
            headers={"Authorization": f"Bearer {access_token}"},
        )
    data = _raise_for_tiktok(res)
    user = data.get("data", {}).get("user", {})
    return _normalize_profile(user)


async def fetch_recent_videos(access_token: str, max_count: int = 20) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=20.0) as client:
        res = await client.post(
            VIDEO_LIST_URL,
            params={"fields": VIDEO_FIELDS},
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={"max_count": max_count},
        )
    data = _raise_for_tiktok(res)
    videos = data.get("data", {}).get("videos", [])
    return [_normalize_video(v) for v in videos]


async def _request_token(payload: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        res = await client.post(
            TOKEN_URL,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    data = _raise_for_tiktok(res)
    now = int(time.time())
    if data.get("expires_in"):
        data["access_expires_at"] = now + int(data["expires_in"])
    if data.get("refresh_expires_in"):
        data["refresh_expires_at"] = now + int(data["refresh_expires_in"])
    return data


def _store_account(
    profile: dict[str, Any],
    token_payload: dict[str, Any],
    access_token: str | None = None,
) -> None:
    encrypted_access = encrypt_token(access_token or token_payload.get("access_token"))
    if not encrypted_access:
        raise TikTokConfigError("TikTok access token is missing.")
    database.upsert_tiktok_account(
        profile=profile,
        token_payload=token_payload,
        encrypted_access_token=encrypted_access,
        encrypted_refresh_token=encrypt_token(token_payload.get("refresh_token")),
    )


def _raise_for_tiktok(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise TikTokConfigError(f"TikTok returned non-JSON response ({response.status_code}).") from exc
    if response.status_code >= 400:
        detail = data.get("error_description") or data.get("message") or str(data)
        raise TikTokConfigError(f"TikTok API error: {detail}")
    error = data.get("error")
    if isinstance(error, dict) and error.get("code") not in (None, "ok"):
        raise TikTokConfigError(f"TikTok API error: {error.get('message') or error}")
    if isinstance(error, str) and error:
        raise TikTokConfigError(f"TikTok API error: {error}")
    return data


def _normalize_profile(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "open_id": user.get("open_id"),
        "display_name": user.get("display_name"),
        "avatar_url": user.get("avatar_url"),
        "profile_deep_link": user.get("profile_deep_link"),
        "bio_description": user.get("bio_description"),
        "is_verified": user.get("is_verified"),
        "follower_count": user.get("follower_count"),
        "following_count": user.get("following_count"),
        "likes_count": user.get("likes_count"),
        "video_count": user.get("video_count"),
    }


def _normalize_video(video: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(video.get("id") or ""),
        "create_time": video.get("create_time"),
        "cover_image_url": video.get("cover_image_url"),
        "share_url": video.get("share_url"),
        "title": video.get("title"),
        "video_description": video.get("video_description"),
        "duration": video.get("duration"),
        "view_count": video.get("view_count"),
        "like_count": video.get("like_count"),
        "comment_count": video.get("comment_count"),
        "share_count": video.get("share_count"),
    }
