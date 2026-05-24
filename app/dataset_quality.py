"""Dataset diagnostics tuned for a creator's own editing style."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any

import numpy as np

from app import database
from app.views_model import FEATURE_COLUMNS


STYLE_FEATURES = [
    "cuts_per_second",
    "motion_intensity",
    "visual_change_rate",
    "hook_speed",
    "audio_pacing",
    "av_sync_score",
    "retention_proxy",
    "loop_potential_score",
]


def build_dataset_quality() -> dict[str, Any]:
    rows = database.list_labeled_for_training()
    n = len(rows)
    if n == 0:
        return {
            "score": 0,
            "label": "No labeled data",
            "summary": "Add videos with real views to build a style-specific predictor.",
            "checks": [],
            "style_clusters": [],
            "recommendations": [
                "Start with 10+ edits that match your usual style, then add low, medium, and high performers."
            ],
        }

    merged = [database.merge_features(r) for r in rows]
    views = [int(r.get("views") or 0) for r in rows]
    style_profiles = [str(r.get("style_profile") or "Unlabeled style").strip() for r in rows]
    style_counts = Counter(style_profiles)
    dominant_style, dominant_count = style_counts.most_common(1)[0]
    tagged_count = sum(1 for s in style_profiles if s != "Unlabeled style")

    view_spread = _log_view_spread(views)
    similarity = _style_similarity_score(merged)
    duplicates = _duplicate_risk(merged)
    dominant_ratio = dominant_count / max(1, tagged_count) if dominant_style != "Unlabeled style" and tagged_count else 0.0
    tag_coverage = tagged_count / max(1, n)
    label_coverage = min(100.0, (n / 50) * 100)
    style_focus = round(100 * (0.45 * similarity + 0.35 * dominant_ratio + 0.2 * tag_coverage), 1)
    view_variety = round(min(100.0, view_spread * 34), 1)

    score = round(
        0.28 * label_coverage
        + 0.27 * style_focus
        + 0.25 * view_variety
        + 0.2 * (100 - duplicates),
        1,
    )
    label = "Strong" if score >= 78 else ("Good" if score >= 62 else ("Building" if score >= 40 else "Weak"))

    checks = [
        _check("Labeled videos", label_coverage, f"{n}/50 labeled videos collected."),
        _check(
            "Style focus",
            style_focus,
            f"{tagged_count}/{n} videos have an explicit style tag; dominant style: {dominant_style}.",
        ),
        _check("View spread", view_variety, f"Views range from {_fmt(min(views))} to {_fmt(max(views))}."),
        _check("Duplicate risk", 100 - duplicates, "Feature fingerprints look clean." if duplicates < 20 else "Some videos look too similar for training."),
    ]

    recommendations: list[str] = []
    if n < 10:
        recommendations.append("Reach 10 labeled videos before trusting predictions beyond rough direction.")
    if n < 50:
        recommendations.append("Aim for 50 labeled edits from your own style for a useful personal model.")
    if view_variety < 45:
        recommendations.append("Add examples with very low, normal, and breakout views. Similar style is good, identical results are not.")
    if style_focus < 55:
        recommendations.append("Tag each upload with the edit style and keep the first model focused on your main style.")
    if duplicates > 35:
        recommendations.append("Avoid training on near-duplicate edits unless their real performance was meaningfully different.")
    if not recommendations:
        recommendations.append("Dataset shape is healthy. Keep adding recent posts every 10-15 uploads and retrain.")

    return {
        "score": score,
        "label": label,
        "summary": (
            f"{label} dataset: {n} labeled videos, {style_focus:.0f}/100 style focus, "
            f"{view_variety:.0f}/100 view variety."
        ),
        "checks": checks,
        "style_clusters": [
            {"style": style, "count": count, "pct": round(count / n * 100, 1)}
            for style, count in style_counts.most_common(6)
        ],
        "recommendations": recommendations,
    }


def _check(name: str, score: float, detail: str) -> dict[str, Any]:
    return {
        "name": name,
        "score": round(max(0.0, min(100.0, score)), 1),
        "status": "ok" if score >= 70 else ("watch" if score >= 45 else "risk"),
        "detail": detail,
    }


def _log_view_spread(views: list[int]) -> float:
    if len(views) < 2:
        return 0.0
    logs = [math.log1p(max(0, v)) for v in views]
    return float(np.percentile(logs, 90) - np.percentile(logs, 10))


def _style_similarity_score(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2:
        return 1.0
    scores = []
    for key in STYLE_FEATURES:
        vals = np.array([float(r.get(key, 0) or 0) for r in rows], dtype=float)
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if abs(mean) < 1e-6:
            scores.append(1.0 if std < 1e-6 else 0.45)
            continue
        cv = abs(std / mean)
        scores.append(max(0.0, min(1.0, 1.0 - cv / 1.6)))
    return float(statistics.mean(scores)) if scores else 0.0


def _duplicate_risk(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 3:
        return 0.0
    matrix = np.array(
        [[float(r.get(c, 0) or 0) for c in FEATURE_COLUMNS] for r in rows],
        dtype=float,
    )
    spread = np.std(matrix, axis=0)
    spread[spread < 1e-6] = 1.0
    norm = (matrix - np.mean(matrix, axis=0)) / spread
    near = 0
    comparisons = 0
    for i in range(len(norm)):
        for j in range(i + 1, len(norm)):
            comparisons += 1
            if float(np.linalg.norm(norm[i] - norm[j])) < 0.85:
                near += 1
    return round((near / max(1, comparisons)) * 100, 1)


def _fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
