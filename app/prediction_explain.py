"""Human-readable justification for predicted view counts."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from app import database
from app.feature_insights import FEATURE_LABELS, build_feature_insights
from app.tiktok_knowledge import algorithm_tips
from app.view_tiers import VIEW_TIERS, views_to_tier_index
from app.views_model import FEATURE_COLUMNS, META_PATH


def _load_meta() -> dict[str, Any]:
    if not META_PATH.exists():
        return {}
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def _dataset_stats() -> dict[str, dict[str, float]]:
    meta = _load_meta()
    if meta.get("dataset_feature_stats"):
        return meta["dataset_feature_stats"]
    rows = database.list_labeled_for_training()
    if len(rows) < 3:
        return {}
    merged = [database.merge_features(r) for r in rows]
    stats: dict[str, dict[str, float]] = {}
    for col in FEATURE_COLUMNS:
        vals = np.array([float(m.get(col, 0) or 0) for m in merged])
        if len(vals) == 0:
            continue
        stats[col] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "p25": float(np.percentile(vals, 25)),
            "p75": float(np.percentile(vals, 75)),
        }
    return stats


def _format_views(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _feature_direction(key: str, val: float, ds: dict[str, float]) -> str:
    med = ds.get("median", val)
    if abs(val - med) < 1e-6:
        return "typical"
    higher_is_better = key in {
        "motion_intensity",
        "hook_motion_3s",
        "motion_first_2s",
        "cuts_first_3s",
        "cuts_per_second",
        "av_sync_score",
        "audio_pacing",
        "early_hold_score",
        "retention_proxy",
        "loop_potential_score",
        "visual_change_rate",
        "audio_dynamic_range",
    }
    lower_is_better = {
        "hook_speed",
        "longest_scene_gap",
        "silence_ratio",
        "avg_scene_duration",
    }
    if key in lower_is_better:
        if val < med:
            return "better_than_usual"
        if val > med:
            return "worse_than_usual"
    elif higher_is_better or key not in lower_is_better:
        if val > med:
            return "better_than_usual"
        if val < med:
            return "worse_than_usual"
    return "typical"


def _contribution_score(
    key: str, val: float, importance: float, ds: dict[str, float]
) -> float:
    if not ds or importance < 0.01:
        return 0.0
    med = ds.get("median", 0)
    spread = max(ds.get("p75", med) - ds.get("p25", med), 1e-6)
    z = (val - med) / spread
    direction = _feature_direction(key, val, ds)
    sign = 1.0 if direction == "better_than_usual" else (-1.0 if direction == "worse_than_usual" else 0.0)
    return float(importance * sign * min(2.0, abs(z)))


def _find_similar_videos(features: dict[str, Any], k: int = 3) -> list[dict[str, Any]]:
    rows = database.list_labeled_for_training()
    if len(rows) < 2:
        return []
    target = np.array([float(features.get(c, 0) or 0) for c in FEATURE_COLUMNS])
    scored: list[tuple[float, dict]] = []
    for r in rows:
        m = database.merge_features(r)
        vec = np.array([float(m.get(c, 0) or 0) for c in FEATURE_COLUMNS])
        dist = float(np.linalg.norm(target - vec))
        scored.append((dist, r))
    scored.sort(key=lambda x: x[0])
    out = []
    for dist, r in scored[:k]:
        out.append(
            {
                "filename": r.get("filename"),
                "views": int(r.get("views", 0)),
                "distance": round(dist, 3),
            }
        )
    return out


def build_views_justification(
    features: dict[str, Any],
    predicted_views: int | None,
    feature_insights: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if predicted_views is None or predicted_views <= 0:
        return {
            "summary": "Train the views model with labeled videos to get a justified prediction.",
            "drivers": [],
            "risks": [],
            "similar_videos": [],
            "predicted_tier": None,
        }

    meta = _load_meta()
    importances = meta.get("feature_importances", {})
    ds_stats = _dataset_stats()
    insights = feature_insights or build_feature_insights(features)

    contributions: list[dict[str, Any]] = []
    for ins in insights:
        key = ins["key"]
        val = float(features.get(key, 0) or 0)
        imp = float(importances.get(key, ins.get("importance", 0)))
        ds = ds_stats.get(key, {})
        contrib = _contribution_score(key, val, imp, ds)
        if abs(contrib) < 0.005 and imp < 0.03:
            continue
        direction = _feature_direction(key, val, ds) if ds else "typical"
        contributions.append(
            {
                "key": key,
                "label": ins["label"],
                "value": ins["value"],
                "importance": imp,
                "contribution": round(contrib, 4),
                "direction": direction,
                "vs_dataset": _vs_dataset_line(key, val, ds),
                "algorithm_note": ins.get("algorithm_note", ""),
            }
        )

    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    drivers = [c for c in contributions if c["contribution"] > 0][:5]
    risks = [c for c in contributions if c["contribution"] < 0][:4]

    tier_idx = views_to_tier_index(predicted_views)
    tier = VIEW_TIERS[tier_idx]

    summary_parts = [
        f"This edit is predicted at approximately **{_format_views(predicted_views)} views**, "
        f"landing in the **{tier['label']}** band based on your trained model "
        f"({meta.get('samples', '?')} labeled videos, tiktokalgorithm.txt retention rules)."
    ]
    if drivers:
        top = drivers[0]
        summary_parts.append(
            f" Main upward driver: **{top['label']}** ({top['vs_dataset']})."
        )
    if risks:
        summary_parts.append(
            f" Main drag: **{risks[0]['label']}** ({risks[0]['vs_dataset']})."
        )

    algo_tips = algorithm_tips(features)
    narrative_bullets = []
    for d in drivers[:3]:
        line = f"**{d['label']}** ({d['value']}): pushes views up — {d['vs_dataset']}."
        if d.get("algorithm_note"):
            line += f" TikTok: {d['algorithm_note']}"
        narrative_bullets.append(line)
    for r in risks[:2]:
        narrative_bullets.append(
            f"**{r['label']}** ({r['value']}): likely limits reach — {r['vs_dataset']}."
        )

    return {
        "summary": " ".join(summary_parts),
        "predicted_views": predicted_views,
        "predicted_tier": tier["label"],
        "drivers": drivers,
        "risks": risks,
        "narrative_bullets": narrative_bullets,
        "algorithm_tips": algo_tips,
        "similar_videos": _find_similar_videos(features),
        "model_samples": meta.get("samples"),
        "model_r2": meta.get("test_r2"),
    }


def _vs_dataset_line(key: str, val: float, ds: dict[str, float]) -> str:
    if not ds:
        return "no labeled baseline yet"
    med = ds["median"]
    direction = _feature_direction(key, val, ds)
    if direction == "typical":
        return "close to your dataset median"

    baseline = abs(med)
    spread = max(abs(ds.get("p75", med) - ds.get("p25", med)), 1e-6)
    if baseline < 1e-6:
        if abs(val) < 1e-6:
            return "close to your dataset median"
        multiple = abs(val) / spread
        if multiple >= 1:
            return f"{multiple:.1f}x beyond your usual baseline range"
        return "above your usual zero baseline"

    pct = abs(val - med) / baseline * 100
    if pct > 500:
        if direction == "better_than_usual":
            return "far better than your typical edit"
        return "far weaker than your typical edit"
    if direction == "better_than_usual":
        return f"~{pct:.0f}% better than your typical edit"
    return f"~{pct:.0f}% weaker than your typical edit"
