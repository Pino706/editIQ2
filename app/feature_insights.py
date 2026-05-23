"""Per-feature explanations: ML learnings + tiktokalgorithm.txt alignment."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from app import database
from app.tiktok_knowledge import doc_loaded, load_algorithm_doc
from app.views_model import FEATURE_COLUMNS, META_PATH

FEATURE_LABELS: dict[str, str] = {
    "cuts_count": "Total cuts",
    "cuts_per_second": "Cuts per second",
    "avg_scene_duration": "Avg scene length",
    "motion_intensity": "Motion intensity",
    "visual_change_rate": "Visual change rate",
    "visual_stability": "Visual stability",
    "hook_speed": "Hook speed (first cut)",
    "audio_energy": "Audio energy",
    "audio_spikes_count": "Audio beat spikes",
    "audio_pacing": "Audio pacing",
    "av_sync_score": "Audio–video sync",
    "hook_motion_3s": "Motion in first 3s",
    "motion_first_2s": "Motion in first 2s",
    "cuts_first_3s": "Cuts in first 3s",
    "longest_scene_gap": "Longest gap between cuts",
    "shake_intensity": "Shake / micro-movement",
    "color_variance": "Color variance",
    "loop_potential_score": "Loop / rewatch potential",
    "audio_dynamic_range": "Audio dynamic range",
    "silence_ratio": "Silence ratio",
    "early_hold_score": "Early hold score (1–2s)",
    "retention_proxy": "Retention proxy (composite)",
}

ALGORITHM_BY_FEATURE: dict[str, str] = {
    "hook_speed": "First 1–2 seconds are a ranking gate; slow hooks increase swipe-away.",
    "hook_motion_3s": "Early motion helps the 1s and 3s hold rates TikTok tests in the test phase.",
    "motion_first_2s": "Stop-scrolling signal: immediate visual density in the opening window.",
    "cuts_first_3s": "Fast early pacing supports completion and pattern matching for edit clusters.",
    "motion_intensity": "Retention is the dominant signal — dynamic edits hold attention longer.",
    "cuts_per_second": "Visual rhythm affects completion rate more than raw like count.",
    "av_sync_score": "Beat-synced cuts align with emotional pacing and edit classification.",
    "audio_pacing": "Audio spikes map to beat drops; strong sync boosts perceived quality.",
    "loop_potential_score": "Rewatch loops are one of the strongest positive signals for short edits.",
    "longest_scene_gap": "Long static gaps raise retention risk and slow test-phase expansion.",
    "early_hold_score": "1-second and 3-second hold rates decide if the video gets a second wave.",
    "retention_proxy": "Combined early-retention proxy aligned with FYP test-phase thresholds.",
    "silence_ratio": "Dead air in the opening hurts the stop-scroll signal.",
    "visual_stability": "Too much chaos vs too little motion both hurt; balance visual density.",
}


def _load_importances() -> dict[str, float]:
    if not META_PATH.exists():
        return {}
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return meta.get("feature_importances", {})


def _dataset_correlations() -> dict[str, float]:
    rows = database.list_labeled_for_training()
    if len(rows) < 5:
        return {}
    merged = [database.merge_features(r) for r in rows]
    views = np.array([float(r["views"]) for r in rows])
    log_views = np.log1p(views)
    corrs: dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        vals = np.array([float(m.get(col, 0) or 0) for m in merged])
        if np.std(vals) < 1e-9:
            continue
        c = float(np.corrcoef(vals, log_views)[0, 1])
        if not np.isnan(c):
            corrs[col] = round(c, 3)
    return corrs


def _ml_learning_line(name: str, importance: float, corr: float | None) -> str:
    imp_pct = round(importance * 100, 1) if importance else 0
    parts = []
    if importance > 0.02:
        parts.append(f"Your trained model weights this {imp_pct}% for predicting views.")
    if corr is not None:
        if corr > 0.15:
            parts.append(f"In your dataset, higher values correlate with more views (r={corr:+.2f}).")
        elif corr < -0.15:
            parts.append(f"In your dataset, higher values correlate with fewer views (r={corr:+.2f}).")
        else:
            parts.append(f"Weak correlation with views in your data (r={corr:+.2f}).")
    if not parts:
        parts.append("Not enough labeled data yet to learn a strong pattern for this signal.")
    return " ".join(parts)


def build_feature_insights(features: dict[str, Any]) -> list[dict[str, Any]]:
    importances = _load_importances()
    corrs = _dataset_correlations()
    algo_loaded = doc_loaded()
    insights: list[dict[str, Any]] = []

    for key in FEATURE_COLUMNS:
        val = features.get(key)
        if val is None:
            continue
        imp = float(importances.get(key, 0))
        corr = corrs.get(key)
        algo = ALGORITHM_BY_FEATURE.get(key, "")
        if algo_loaded and not algo:
            algo = "Supports overall retention and test-phase performance per tiktokalgorithm.txt."

        insights.append(
            {
                "key": key,
                "label": FEATURE_LABELS.get(key, key.replace("_", " ").title()),
                "value": _format_value(key, val),
                "ml_learning": _ml_learning_line(key, imp, corr),
                "algorithm_note": algo,
                "importance": imp,
                "correlation": corr,
            }
        )

    insights.sort(key=lambda x: x.get("importance", 0), reverse=True)
    return insights


def _format_value(key: str, val: Any) -> str:
    if isinstance(val, float):
        if key.endswith("_score") or "ratio" in key or "intensity" in key:
            return f"{val:.3f}"
        if "speed" in key or "gap" in key or "duration" in key:
            return f"{val:.2f}s"
        return f"{val:.4f}"
    return str(val)
