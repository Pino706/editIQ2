"""Account-context scoring built from connected TikTok data."""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any

from app import database


def get_account_dashboard() -> dict[str, Any]:
    account = database.get_connected_tiktok_account()
    if not account:
        return {
            "connected": False,
            "profile": None,
            "videos": [],
            "summary": None,
            "message": "Connect TikTok to unlock account intelligence.",
        }
    videos = database.list_tiktok_videos(account["open_id"], limit=50)
    return {
        "connected": True,
        "profile": _public_profile(account),
        "videos": [_public_video(v) for v in videos],
        "summary": summarize_account(account, videos),
    }


def summarize_account(account: dict[str, Any], videos: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [_public_video(v) for v in videos]
    view_values = [int(v["view_count"] or 0) for v in clean if v.get("view_count") is not None]
    like_values = [int(v["like_count"] or 0) for v in clean if v.get("like_count") is not None]
    comment_values = [int(v["comment_count"] or 0) for v in clean if v.get("comment_count") is not None]
    share_values = [int(v["share_count"] or 0) for v in clean if v.get("share_count") is not None]

    best = max(clean, key=lambda v: v.get("view_count") or 0, default=None)
    worst = min(clean, key=lambda v: v.get("view_count") or 0, default=None)
    trend = _growth_trend(clean)
    frequency = _posting_frequency(clean)
    avg_views = int(statistics.mean(view_values)) if view_values else None
    avg_likes = int(statistics.mean(like_values)) if like_values else None
    avg_comments = int(statistics.mean(comment_values)) if comment_values else None
    avg_shares = int(statistics.mean(share_values)) if share_values else None
    momentum = _momentum_score(view_values, trend, frequency)

    return {
        "recent_uploads": len(clean),
        "followers": account.get("follower_count"),
        "following": account.get("following_count"),
        "likes": account.get("likes_count"),
        "average_views": avg_views,
        "average_likes": avg_likes,
        "average_comments": avg_comments,
        "average_shares": avg_shares,
        "posting_frequency_per_week": frequency,
        "recent_growth_trend": trend,
        "best_recent_video": best,
        "worst_recent_video": worst,
        "account_momentum_score": momentum,
    }


def score_account_fit(features: dict[str, Any], account_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not account_summary:
        return {
            "account_fit_score": None,
            "trend_fit_score": None,
            "account_adjustment_factor": 1.0,
            "account_reason": "No connected TikTok account context.",
        }

    momentum = float(account_summary.get("account_momentum_score") or 50)
    avg_views = account_summary.get("average_views")
    pacing = min(100.0, float(features.get("cuts_per_second", 0) or 0) * 14)
    retention = float(features.get("retention_proxy", 0) or 0) * 100
    sync = float(features.get("av_sync_score", 0) or 0) * 100
    account_fit = round(0.35 * momentum + 0.25 * pacing + 0.25 * retention + 0.15 * sync, 1)
    trend = account_summary.get("recent_growth_trend") or 0
    trend_fit = round(max(0.0, min(100.0, 50 + trend * 0.65 + (momentum - 50) * 0.35)), 1)
    adjustment = 1.0 + ((account_fit - 50) / 100) * 0.24 + ((trend_fit - 50) / 100) * 0.18
    adjustment = round(max(0.65, min(1.45, adjustment)), 3)
    reason = (
        f"Account momentum is {momentum:.0f}/100"
        + (f" with recent average around {_format_views(avg_views)} views." if avg_views else ".")
    )
    return {
        "account_fit_score": account_fit,
        "trend_fit_score": trend_fit,
        "account_adjustment_factor": adjustment,
        "account_reason": reason,
    }


def _public_profile(account: dict[str, Any]) -> dict[str, Any]:
    return {
        "open_id": account.get("open_id"),
        "display_name": account.get("display_name"),
        "avatar_url": account.get("avatar_url"),
        "profile_deep_link": account.get("profile_deep_link"),
        "bio_description": account.get("bio_description"),
        "is_verified": bool(account.get("is_verified")) if account.get("is_verified") is not None else None,
        "follower_count": account.get("follower_count"),
        "following_count": account.get("following_count"),
        "likes_count": account.get("likes_count"),
        "video_count": account.get("video_count"),
        "updated_at": account.get("updated_at"),
    }


def _public_video(video: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": video.get("id"),
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


def _growth_trend(videos: list[dict[str, Any]]) -> float | None:
    ordered = sorted(
        [v for v in videos if v.get("create_time") and v.get("view_count") is not None],
        key=lambda v: int(v["create_time"]),
    )
    if len(ordered) < 4:
        return None
    half = max(2, len(ordered) // 2)
    older = [int(v["view_count"] or 0) for v in ordered[:half]]
    newer = [int(v["view_count"] or 0) for v in ordered[-half:]]
    old_avg = statistics.mean(older) if older else 0
    new_avg = statistics.mean(newer) if newer else 0
    if old_avg <= 0:
        return 0.0
    return round(((new_avg - old_avg) / old_avg) * 100, 1)


def _posting_frequency(videos: list[dict[str, Any]]) -> float | None:
    times = sorted([int(v["create_time"]) for v in videos if v.get("create_time")])
    if len(times) < 2:
        return None
    days = max(1.0, (times[-1] - times[0]) / 86400)
    return round((len(times) / days) * 7, 2)


def _momentum_score(view_values: list[int], trend: float | None, frequency: float | None) -> int | None:
    if not view_values:
        return None
    median_views = max(1, statistics.median(view_values))
    latest = statistics.mean(view_values[: min(5, len(view_values))])
    performance = max(0, min(100, (latest / median_views) * 50))
    trend_component = 50 if trend is None else max(0, min(100, 50 + trend))
    frequency_component = 50 if frequency is None else max(0, min(100, min(frequency, 14) / 14 * 100))
    return int(round(0.5 * performance + 0.3 * trend_component + 0.2 * frequency_component))


def _format_views(n: Any) -> str:
    if n is None:
        return "-"
    v = float(n)
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v / 1_000:.1f}K"
    return str(int(v))
