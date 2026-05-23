"""TikTok algorithm notes from tiktokalgorithm.txt — used for edit feedback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

KNOWLEDGE_PATH = Path(__file__).resolve().parent.parent / "tiktokalgorithm.txt"

_cached_text: str | None = None


def load_algorithm_doc() -> str:
    global _cached_text
    if _cached_text is not None:
        return _cached_text
    if KNOWLEDGE_PATH.exists():
        _cached_text = KNOWLEDGE_PATH.read_text(encoding="utf-8")
    else:
        _cached_text = ""
    return _cached_text


def doc_loaded() -> bool:
    return bool(load_algorithm_doc().strip())


def algorithm_tips(features: dict[str, Any]) -> list[str]:
    """Rule-based tips aligned with tiktokalgorithm.txt (retention, hook, rewatch)."""
    if not doc_loaded():
        return []

    tips: list[str] = []
    hook_speed = float(features.get("hook_speed", 3.0))
    cps = float(features.get("cuts_per_second", 0.0))
    motion = float(features.get("motion_intensity", 0.0))
    av_sync = float(features.get("av_sync_score", 0.5))
    duration = float(features.get("duration", 0.0) or 15.0)

    if hook_speed > 2.0:
        tips.append(
            "Algorithm: the first 1–2 seconds are a ranking gate — add motion or a cut "
            "immediately (avoid empty intros)."
        )
    elif hook_speed <= 1.0:
        tips.append(
            "Algorithm: strong early hook — you match the “stop scrolling” signal TikTok tests first."
        )

    if cps < 0.8:
        tips.append(
            "Algorithm: low visual rhythm — retention/completion matter more than likes; "
            "faster pacing often helps edits stay above the test-phase threshold."
        )

    if motion < 0.05:
        tips.append(
            "Algorithm: retention is the dominant signal — add movement, punch-ins, or "
            "visual density to reduce early swipe-away."
        )

    if av_sync < 0.5:
        tips.append(
            "Algorithm: audio–visual sync helps classification (edit clusters) and emotional pacing — "
            "snap cuts to beat drops."
        )

    if duration <= 12 and cps >= 1.0:
        tips.append(
            "Algorithm: short edits that loop can boost rewatch rate — one of the strongest signals for edits."
        )

    return tips[:3]
