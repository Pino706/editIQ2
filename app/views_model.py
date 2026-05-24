"""ML model trained on real TikTok view counts (log-scale)."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

from app import database

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(
    os.getenv(
        "EDITIQ_MODEL_DIR",
        str(Path(tempfile.gettempdir()) / "editiq-models")
        if os.getenv("VERCEL")
        else str(PROJECT_ROOT / "data" / "models"),
    )
)
MODEL_PATH = MODEL_DIR / "views_model.pkl"
META_PATH = MODEL_DIR / "views_model_meta.json"

FEATURE_COLUMNS = [
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
    "hook_motion_3s",
    "motion_first_2s",
    "cuts_first_3s",
    "longest_scene_gap",
    "shake_intensity",
    "color_variance",
    "loop_potential_score",
    "audio_dynamic_range",
    "silence_ratio",
    "early_hold_score",
    "retention_proxy",
]

MIN_SAMPLES = 5
DATASET_GOAL = 200
RECOMMENDED_SAMPLES = 50
STRONG_SAMPLES = 100


def get_model_status() -> dict[str, Any]:
    stats = database.dataset_stats()
    if not MODEL_PATH.exists():
        return {
            "trained": False,
            "labeled": stats["labeled"],
            "goal": DATASET_GOAL,
            "can_train": stats["can_train"],
            "message": (
                f"No views model yet. Label {MIN_SAMPLES}+ videos with view counts, then train."
            ),
        }
    meta: dict[str, Any] = {"trained": True, **stats}
    if META_PATH.exists():
        meta.update(json.loads(META_PATH.read_text(encoding="utf-8")))
    meta["can_train"] = stats["can_train"]
    return meta


def train_model() -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import KFold
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {
            "success": False,
            "message": "Training dependencies are not installed in this deployment.",
        }

    rows = database.list_labeled_for_training()
    if len(rows) < MIN_SAMPLES:
        return {
            "success": False,
            "message": f"Need at least {MIN_SAMPLES} videos with views (have {len(rows)}).",
            "labeled": len(rows),
        }

    df = pd.DataFrame(
        [{**database.merge_features(r), "views": r["views"]} for r in rows]
    )
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    df = df.fillna(0)

    X = df[FEATURE_COLUMNS].astype(float).values
    y = np.log1p(df["views"].astype(float).values)

    n = len(df)
    uses_scaler = False
    scaler: StandardScaler | None = None
    X_model = X
    model_type = "CatBoostRegressor"

    try:
        from catboost import CatBoostRegressor

        def make_model() -> Any:
            return CatBoostRegressor(
                iterations=min(900, 180 + n * 10),
                depth=4 if n < RECOMMENDED_SAMPLES else 5,
                learning_rate=0.035 if n >= RECOMMENDED_SAMPLES else 0.055,
                loss_function="RMSE",
                l2_leaf_reg=8,
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
            )
    except Exception:
        model_type = "ExtraTreesRegressorFallback"
        uses_scaler = True
        scaler = StandardScaler()
        X_model = scaler.fit_transform(X)

        def make_model() -> Any:
            return ExtraTreesRegressor(
                n_estimators=min(500, 160 + n * 6),
                max_depth=None if n >= RECOMMENDED_SAMPLES else 8,
                min_samples_leaf=max(1, n // 30),
                random_state=42,
            )

    if n >= 8:
        n_splits = min(5, n)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_pred = np.zeros(n)
        fold_scores = []
        for train_idx, test_idx in kf.split(X_model):
            fold_model = make_model()
            fold_model.fit(X_model[train_idx], y[train_idx])
            pred = np.asarray(fold_model.predict(X_model[test_idx]), dtype=float)
            oof_pred[test_idx] = pred
            fold_scores.append(
                {
                    "r2": round(float(r2_score(y[test_idx], pred)), 4)
                    if len(test_idx) > 1
                    else None,
                    "mae_views": round(
                        float(mean_absolute_error(np.expm1(y[test_idx]), np.expm1(pred))),
                        0,
                    ),
                }
            )
    else:
        model_for_oof = make_model()
        model_for_oof.fit(X_model, y)
        oof_pred = np.asarray(model_for_oof.predict(X_model), dtype=float)
        fold_scores = []

    model = make_model()
    model.fit(X_model, y)
    train_pred = np.asarray(model.predict(X_model), dtype=float)
    train_r2 = float(r2_score(y, train_pred))
    test_r2 = float(r2_score(y, oof_pred)) if n >= 8 else train_r2
    actual_views = np.expm1(y)
    pred_views = np.expm1(oof_pred)
    mae_views = float(mean_absolute_error(actual_views, pred_views))
    median_ae_views = float(np.median(np.abs(actual_views - pred_views)))
    log_abs_errors = np.abs(y - oof_pred)
    residual_log_mae = float(np.mean(log_abs_errors))
    residual_log_p80 = float(np.percentile(log_abs_errors, 80))

    if hasattr(model, "get_feature_importance"):
        raw_importances = model.get_feature_importance()
    elif hasattr(model, "feature_importances_"):
        raw_importances = model.feature_importances_
    else:
        raw_importances = np.zeros(len(FEATURE_COLUMNS))
    total_importance = float(np.sum(raw_importances))
    if total_importance > 0:
        raw_importances = np.asarray(raw_importances, dtype=float) / total_importance
    importances = {
        FEATURE_COLUMNS[i]: round(float(raw_importances[i]), 4)
        for i in range(len(FEATURE_COLUMNS))
    }
    importances = dict(
        sorted(importances.items(), key=lambda x: x[1], reverse=True)
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "scaler": scaler,
        "features": FEATURE_COLUMNS,
        "target": "log_views",
        "model_type": model_type,
        "uses_scaler": uses_scaler,
        "residual_log_mae": residual_log_mae,
        "residual_log_p80": residual_log_p80,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    dataset_feature_stats: dict[str, dict[str, float]] = {}
    for col in FEATURE_COLUMNS:
        vals = df[col].astype(float).values
        dataset_feature_stats[col] = {
            "mean": round(float(np.mean(vals)), 4),
            "median": round(float(np.median(vals)), 4),
            "p25": round(float(np.percentile(vals, 25)), 4),
            "p75": round(float(np.percentile(vals, 75)), 4),
        }

    meta = {
        "target": "log_views",
        "samples": len(df),
        "train_r2": round(train_r2, 4),
        "cv_r2": round(test_r2, 4),
        "test_r2": round(test_r2, 4),
        "mae_views": round(mae_views, 0),
        "median_ae_views": round(median_ae_views, 0),
        "residual_log_mae": round(residual_log_mae, 4),
        "residual_log_p80": round(residual_log_p80, 4),
        "model_type": model_type,
        "fold_scores": fold_scores,
        "feature_importances": importances,
        "dataset_feature_stats": dataset_feature_stats,
        "recommended_samples": RECOMMENDED_SAMPLES,
        "dataset_goal": DATASET_GOAL,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for row in rows:
        feat = database.merge_features(row)
        pred = predict_views(feat)
        if pred and row.get("id"):
            database.update_predicted_views(row["id"], pred["predicted_views"])

    try:
        from app.view_tiers import train_tier_classifier

        train_tier_classifier()
    except Exception:
        pass

    return {"success": True, **meta}


def predict_views(features: dict[str, Any]) -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    import numpy as np

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    scaler = bundle["scaler"]
    cols = bundle["features"]

    row = np.array([[float(features.get(c, 0) or 0) for c in cols]])
    X = scaler.transform(row) if scaler is not None else row
    log_pred = float(model.predict(X)[0])
    predicted = int(max(0, round(np.expm1(log_pred))))
    spread = float(bundle.get("residual_log_p80") or bundle.get("residual_log_mae") or 0.65)
    low = int(max(0, round(np.expm1(max(0.0, log_pred - spread)))))
    high = int(max(predicted, round(np.expm1(log_pred + spread))))

    meta = {}
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    from app.prediction_explain import build_views_justification
    from app.view_tiers import predict_tier_probabilities

    tiers = predict_tier_probabilities(features, predicted)
    justification = build_views_justification(features, predicted)

    return {
        "predicted_views": predicted,
        "prediction_range_low": low,
        "prediction_range_high": high,
        "views_model_ready": True,
        "prediction_source": bundle.get("model_type", "views_ml"),
        "training_samples": meta.get("samples"),
        "model_type": meta.get("model_type") or bundle.get("model_type"),
        "tier_probabilities": tiers,
        "views_justification": justification,
    }
