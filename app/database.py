"""SQLite persistence for video analyses."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "editiq.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    duration REAL NOT NULL,
    views INTEGER,
    likes INTEGER,
    shares INTEGER,
    saves INTEGER,
    retention_pct REAL,
    cuts_count INTEGER,
    cuts_per_second REAL,
    avg_scene_duration REAL,
    motion_intensity REAL,
    visual_change_rate REAL,
    visual_stability REAL,
    hook_speed REAL,
    audio_energy REAL,
    audio_spikes_count INTEGER,
    audio_pacing REAL,
    av_sync_score REAL,
    viral_score REAL,
    hook_score REAL,
    pacing_score REAL,
    retention_risk REAL,
    feedback TEXT,
    is_ml_predicted INTEGER DEFAULT 0,
    timeline_json TEXT
);

CREATE TABLE IF NOT EXISTS tiktok_accounts (
    open_id TEXT PRIMARY KEY,
    display_name TEXT,
    avatar_url TEXT,
    profile_deep_link TEXT,
    bio_description TEXT,
    is_verified INTEGER,
    follower_count INTEGER,
    following_count INTEGER,
    likes_count INTEGER,
    video_count INTEGER,
    scopes TEXT,
    access_token TEXT NOT NULL,
    refresh_token TEXT,
    access_expires_at INTEGER,
    refresh_expires_at INTEGER,
    connected_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tiktok_videos (
    id TEXT PRIMARY KEY,
    open_id TEXT NOT NULL,
    create_time INTEGER,
    cover_image_url TEXT,
    share_url TEXT,
    title TEXT,
    video_description TEXT,
    duration REAL,
    view_count INTEGER,
    like_count INTEGER,
    comment_count INTEGER,
    share_count INTEGER,
    raw_json TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(open_id) REFERENCES tiktok_accounts(open_id)
);

CREATE TABLE IF NOT EXISTS oauth_states (
    state TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);
"""


DATASET_GOAL = 200
MIN_TRAIN_SAMPLES = 5
RECOMMENDED_TRAIN_SAMPLES = 50


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.commit()


def _migrate(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(videos)")}
    if "predicted_views" not in cols:
        conn.execute("ALTER TABLE videos ADD COLUMN predicted_views INTEGER")
    if "features_extra_json" not in cols:
        conn.execute("ALTER TABLE videos ADD COLUMN features_extra_json TEXT")
    acct_cols = {row[1] for row in conn.execute("PRAGMA table_info(tiktok_accounts)")}
    optional_account_cols = {
        "is_verified": "INTEGER",
        "follower_count": "INTEGER",
        "following_count": "INTEGER",
        "likes_count": "INTEGER",
        "video_count": "INTEGER",
        "scopes": "TEXT",
    }
    for col, typ in optional_account_cols.items():
        if acct_cols and col not in acct_cols:
            conn.execute(f"ALTER TABLE tiktok_accounts ADD COLUMN {col} {typ}")


def merge_features(row: dict[str, Any]) -> dict[str, float]:
    """Merge base columns + extended JSON into full feature vector for ML."""
    from app.views_model import FEATURE_COLUMNS

    merged: dict[str, float] = {}
    extra: dict[str, Any] = {}
    raw = row.get("features_extra_json")
    if raw:
        try:
            extra = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            extra = {}
    for key in FEATURE_COLUMNS:
        if key in row and row[key] is not None:
            merged[key] = float(row[key])
        elif key in extra:
            merged[key] = float(extra[key])
        else:
            merged[key] = 0.0
    return merged


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def insert_analysis(record: dict[str, Any]) -> int:
    cols = [
        "filename",
        "timestamp",
        "duration",
        "views",
        "likes",
        "shares",
        "saves",
        "retention_pct",
        "cuts_count",
        "cuts_per_second",
        "avg_scene_duration",
        "motion_intensity",
        "visual_change_rate",
        "visual_stability",
        "hook_speed",
        "audio_energy",
        "audio_spikes_count",
        "audio_pacing",
        "av_sync_score",
        "viral_score",
        "hook_score",
        "pacing_score",
        "retention_risk",
        "feedback",
        "is_ml_predicted",
        "timeline_json",
        "predicted_views",
        "features_extra_json",
    ]
    placeholders = ", ".join("?" for _ in cols)
    names = ", ".join(cols)
    values = [record.get(c) for c in cols]
    if isinstance(values[cols.index("feedback")], (dict, list)):
        values[cols.index("feedback")] = json.dumps(values[cols.index("feedback")])
    extra_idx = cols.index("features_extra_json")
    if isinstance(values[extra_idx], dict):
        values[extra_idx] = json.dumps(values[extra_idx])
    with get_connection() as conn:
        cur = conn.execute(
            f"INSERT INTO videos ({names}) VALUES ({placeholders})",
            values,
        )
        conn.commit()
        return int(cur.lastrowid)


def list_history(limit: int = 100) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, filename, timestamp, duration, viral_score, hook_score,
                   pacing_score, retention_risk, cuts_per_second, motion_intensity,
                   views, likes, is_ml_predicted, predicted_views
            FROM videos
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_analysis(video_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def get_all_for_training() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM videos ORDER BY id").fetchall()
    return [_row_to_dict(r) for r in rows]


def count_labeled() -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM videos WHERE views IS NOT NULL AND views > 0"
        ).fetchone()
    return int(row["n"])


def count_unlabeled() -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM videos WHERE views IS NULL OR views <= 0"
        ).fetchone()
    return int(row["n"])


def dataset_stats() -> dict[str, Any]:
    labeled = count_labeled()
    unlabeled = count_unlabeled()
    return {
        "labeled": labeled,
        "unlabeled": unlabeled,
        "total": labeled + unlabeled,
        "goal": DATASET_GOAL,
        "min_train": MIN_TRAIN_SAMPLES,
        "recommended_train": RECOMMENDED_TRAIN_SAMPLES,
        "can_train": labeled >= MIN_TRAIN_SAMPLES,
        "model_quality": (
            "strong" if labeled >= 100 else ("good" if labeled >= 50 else "building")
        ),
    }


def list_labeled_for_training() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM videos
            WHERE views IS NOT NULL AND views > 0
            ORDER BY views DESC
            """
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def list_labeled_summary() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, filename, views, duration, viral_score, hook_score,
                   pacing_score, cuts_per_second, motion_intensity, predicted_views,
                   timestamp
            FROM videos
            WHERE views IS NOT NULL AND views > 0
            ORDER BY views DESC
            """
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def update_views(video_id: int, views: int) -> bool:
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE videos SET views = ? WHERE id = ?",
            (views, video_id),
        )
        conn.commit()
        return cur.rowcount > 0


def update_predicted_views(video_id: int, predicted_views: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE videos SET predicted_views = ? WHERE id = ?",
            (predicted_views, video_id),
        )
        conn.commit()


def create_oauth_state(state: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO oauth_states (state, created_at) VALUES (?, ?)",
            (state, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def consume_oauth_state(state: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT state FROM oauth_states WHERE state = ?",
            (state,),
        ).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM oauth_states WHERE state = ?", (state,))
        conn.commit()
        return True


def upsert_tiktok_account(
    profile: dict[str, Any],
    token_payload: dict[str, Any],
    encrypted_access_token: str,
    encrypted_refresh_token: str | None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    open_id = profile.get("open_id")
    if not open_id:
        raise ValueError("TikTok profile missing open_id")
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO tiktok_accounts (
                open_id, display_name, avatar_url, profile_deep_link,
                bio_description, is_verified, follower_count, following_count,
                likes_count, video_count, scopes, access_token, refresh_token,
                access_expires_at, refresh_expires_at, connected_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(open_id) DO UPDATE SET
                display_name = excluded.display_name,
                avatar_url = excluded.avatar_url,
                profile_deep_link = excluded.profile_deep_link,
                bio_description = excluded.bio_description,
                is_verified = excluded.is_verified,
                follower_count = excluded.follower_count,
                following_count = excluded.following_count,
                likes_count = excluded.likes_count,
                video_count = excluded.video_count,
                scopes = excluded.scopes,
                access_token = excluded.access_token,
                refresh_token = excluded.refresh_token,
                access_expires_at = excluded.access_expires_at,
                refresh_expires_at = excluded.refresh_expires_at,
                updated_at = excluded.updated_at
            """,
            (
                open_id,
                profile.get("display_name"),
                profile.get("avatar_url"),
                profile.get("profile_deep_link"),
                profile.get("bio_description"),
                int(bool(profile.get("is_verified"))) if profile.get("is_verified") is not None else None,
                profile.get("follower_count"),
                profile.get("following_count"),
                profile.get("likes_count"),
                profile.get("video_count"),
                token_payload.get("scope"),
                encrypted_access_token,
                encrypted_refresh_token,
                token_payload.get("access_expires_at"),
                token_payload.get("refresh_expires_at"),
                now,
                now,
            ),
        )
        conn.commit()


def get_connected_tiktok_account() -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM tiktok_accounts ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
    return _row_to_dict(row) if row else None


def update_tiktok_tokens(
    open_id: str,
    encrypted_access_token: str,
    encrypted_refresh_token: str | None,
    access_expires_at: int | None,
    refresh_expires_at: int | None,
    scopes: str | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE tiktok_accounts
            SET access_token = ?, refresh_token = COALESCE(?, refresh_token),
                access_expires_at = ?, refresh_expires_at = ?,
                scopes = COALESCE(?, scopes), updated_at = ?
            WHERE open_id = ?
            """,
            (
                encrypted_access_token,
                encrypted_refresh_token,
                access_expires_at,
                refresh_expires_at,
                scopes,
                datetime.now(timezone.utc).isoformat(),
                open_id,
            ),
        )
        conn.commit()


def upsert_tiktok_videos(open_id: str, videos: list[dict[str, Any]]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        for video in videos:
            video_id = str(video.get("id") or "")
            if not video_id:
                continue
            conn.execute(
                """
                INSERT INTO tiktok_videos (
                    id, open_id, create_time, cover_image_url, share_url, title,
                    video_description, duration, view_count, like_count,
                    comment_count, share_count, raw_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    create_time = excluded.create_time,
                    cover_image_url = excluded.cover_image_url,
                    share_url = excluded.share_url,
                    title = excluded.title,
                    video_description = excluded.video_description,
                    duration = excluded.duration,
                    view_count = excluded.view_count,
                    like_count = excluded.like_count,
                    comment_count = excluded.comment_count,
                    share_count = excluded.share_count,
                    raw_json = excluded.raw_json,
                    updated_at = excluded.updated_at
                """,
                (
                    video_id,
                    open_id,
                    video.get("create_time"),
                    video.get("cover_image_url"),
                    video.get("share_url"),
                    video.get("title"),
                    video.get("video_description"),
                    video.get("duration"),
                    video.get("view_count"),
                    video.get("like_count"),
                    video.get("comment_count"),
                    video.get("share_count"),
                    json.dumps(video),
                    now,
                ),
            )
        conn.commit()


def list_tiktok_videos(open_id: str, limit: int = 50) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM tiktok_videos
            WHERE open_id = ?
            ORDER BY COALESCE(create_time, 0) DESC
            LIMIT ?
            """,
            (open_id, limit),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def disconnect_tiktok_account(open_id: str | None = None) -> None:
    with get_connection() as conn:
        if open_id:
            conn.execute("DELETE FROM tiktok_videos WHERE open_id = ?", (open_id,))
            conn.execute("DELETE FROM tiktok_accounts WHERE open_id = ?", (open_id,))
        else:
            conn.execute("DELETE FROM tiktok_videos")
            conn.execute("DELETE FROM tiktok_accounts")
        conn.commit()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    if d.get("feedback"):
        try:
            d["feedback"] = json.loads(d["feedback"])
        except (json.JSONDecodeError, TypeError):
            pass
    if d.get("timeline_json"):
        try:
            d["timeline"] = json.loads(d["timeline_json"])
        except (json.JSONDecodeError, TypeError):
            d["timeline"] = None
    if d.get("features_extra_json"):
        try:
            d["features_extra"] = json.loads(d["features_extra_json"])
        except (json.JSONDecodeError, TypeError):
            d["features_extra"] = {}
    return d


_BASE_DB_FEATURES = {
    "cuts_count",
    "cuts_per_second",
    "avg_scene_duration",
    "motion_intensity",
    "visual_change_rate",
    "visual_stability",
    "hook_speed",
    "audio_energy",
    "audio_spikes_count",
    "audio_pacing",
    "av_sync_score",
}


def build_record(
    filename: str,
    duration: float,
    features: dict[str, Any],
    scores: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    is_ml: bool = False,
) -> dict[str, Any]:
    meta = metadata or {}
    timeline = features.get("timeline", {})
    feat_cols = {k: v for k, v in features.items() if k not in ("timeline", "duration")}
    extra = {k: v for k, v in feat_cols.items() if k not in _BASE_DB_FEATURES}
    base_feats = {k: v for k, v in feat_cols.items() if k in _BASE_DB_FEATURES}
    return {
        "filename": filename,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration": duration,
        "views": meta.get("views"),
        "likes": meta.get("likes"),
        "shares": meta.get("shares"),
        "saves": meta.get("saves"),
        "retention_pct": meta.get("retention_pct"),
        **base_feats,
        **scores,
        "is_ml_predicted": 1 if is_ml else 0,
        "timeline_json": json.dumps(timeline),
        "predicted_views": meta.get("predicted_views"),
        "features_extra_json": json.dumps(extra) if extra else None,
    }
