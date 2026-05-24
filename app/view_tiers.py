"""View-count tier definitions and reach-probability estimation."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

from app import database
from app.views_model import FEATURE_COLUMNS, MODEL_DIR

TIER_MODEL_PATH = MODEL_DIR / "tier_model.pkl"

# User-facing view brackets (inclusive lower bound, exclusive upper except last)
VIEW_TIERS: list[dict[str, Any]] = [
    {"id": "0-1k", "label": "0 – 1K", "min": 0, "max": 1_000},
    {"id": "1k-5k", "label": "1K – 5K", "min": 1_000, "max": 5_000},
    {"id": "5k-15k", "label": "5K – 15K", "min": 5_000, "max": 15_000},
    {"id": "15k-50k", "label": "15K – 50K", "min": 15_000, "max": 50_000},
    {"id": "50k-100k", "label": "50K – 100K", "min": 50_000, "max": 100_000},
    {"id": "100k-200k", "label": "100K – 200K", "min": 100_000, "max": 200_000},
    {"id": "200k+", "label": "200K+", "min": 200_000, "max": None},
]


def views_to_tier_index(views: int) -> int:
    v = max(0, int(views))
    for i, tier in enumerate(VIEW_TIERS):
        if tier["max"] is None or v < tier["max"]:
            return i
    return len(VIEW_TIERS) - 1


def tier_midpoint(index: int) -> float:
    tier = VIEW_TIERS[index]
    lo = tier["min"]
    hi = tier["max"] if tier["max"] is not None else lo * 3
    return math.sqrt(lo * hi) if lo > 0 else hi / 2


def train_tier_classifier() -> bool:
    try:
        import pandas as pd
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return False

    rows = database.list_labeled_for_training()
    if len(rows) < 5:
        return False

    df = pd.DataFrame([{**database.merge_features(r), "views": r["views"]} for r in rows])
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    df = df.fillna(0)

    y = df["views"].astype(int).apply(views_to_tier_index).values
    X = df[FEATURE_COLUMNS].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = GradientBoostingClassifier(
        n_estimators=80,
        max_depth=3,
        random_state=42,
    )
    clf.fit(X_scaled, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(TIER_MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf, "scaler": scaler, "features": FEATURE_COLUMNS}, f)
    return True


def _views_based_probs(predicted_views: int, sigma_log: float = 0.85) -> np.ndarray:
    import numpy as np

    """Soft assignment of predicted views across tiers (log-normal spread)."""
    log_v = math.log1p(max(0, predicted_views))
    probs = np.zeros(len(VIEW_TIERS))
    for i, tier in enumerate(VIEW_TIERS):
        mid = math.log1p(tier_midpoint(i))
        dist = abs(log_v - mid)
        probs[i] = math.exp(-0.5 * (dist / sigma_log) ** 2)
    s = probs.sum()
    return probs / s if s > 0 else np.ones(len(VIEW_TIERS)) / len(VIEW_TIERS)


def predict_tier_probabilities(
    features: dict[str, Any],
    predicted_views: int | None,
) -> list[dict[str, Any]]:
    import numpy as np

    rows = database.list_labeled_for_training()
    n = len(rows)

    clf_probs: np.ndarray | None = None
    if TIER_MODEL_PATH.exists() and n >= 5:
        with open(TIER_MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        row = np.array([[float(features.get(c, 0) or 0) for c in bundle["features"]]])
        X = bundle["scaler"].transform(row)
        clf_probs = bundle["model"].predict_proba(X)[0]
        if len(clf_probs) < len(VIEW_TIERS):
            padded = np.zeros(len(VIEW_TIERS))
            padded[: len(clf_probs)] = clf_probs
            clf_probs = padded

    if clf_probs is None and n >= 5:
        empirical = np.zeros(len(VIEW_TIERS))
        for r in rows:
            empirical[views_to_tier_index(int(r["views"]))] += 1
        clf_probs = empirical / empirical.sum()

    views_probs = None
    if predicted_views is not None and predicted_views > 0:
        views_probs = _views_based_probs(predicted_views)

    if clf_probs is not None and views_probs is not None:
        blended = 0.55 * clf_probs + 0.45 * views_probs
    elif clf_probs is not None:
        blended = clf_probs
    elif views_probs is not None:
        blended = views_probs
    else:
        blended = np.array([0.45, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01], dtype=float)
        blended = blended / blended.sum()

    blended = blended / blended.sum()
    # P(reach at least this tier) = sum of probs for this tier and all higher
    cumulative_from_top = 0.0
    reach_at_least: list[float] = []
    for i in range(len(VIEW_TIERS) - 1, -1, -1):
        cumulative_from_top += float(blended[i])
        reach_at_least.insert(0, cumulative_from_top)

    result: list[dict[str, Any]] = []
    for i, tier in enumerate(VIEW_TIERS):
        result.append(
            {
                "tier_id": tier["id"],
                "label": tier["label"],
                "min_views": tier["min"],
                "max_views": tier["max"],
                "probability_pct": round(reach_at_least[i] * 100, 1),
                "tier_only_pct": round(float(blended[i]) * 100, 1),
            }
        )
    return result
