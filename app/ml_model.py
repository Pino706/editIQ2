"""Optional scikit-learn model for viral score prediction."""

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
MODEL_PATH = MODEL_DIR / "viral_model.pkl"
META_PATH = MODEL_DIR / "viral_model_meta.json"

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
]

MIN_SAMPLES = 5


def get_model_status() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        return {
            "trained": False,
            "message": "No model trained yet. Analyze videos and call POST /api/train.",
        }
    meta = {}
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return {"trained": True, **meta}


def train_model(use_retention_target: bool = False) -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {
            "success": False,
            "message": "Training dependencies are not installed in this deployment.",
        }

    rows = database.get_all_for_training()
    if len(rows) < MIN_SAMPLES:
        return {
            "success": False,
            "message": f"Need at least {MIN_SAMPLES} analyzed videos (have {len(rows)}).",
        }

    df = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    df = df.fillna(0)

    target_col = "retention_pct" if use_retention_target else "viral_score"
    valid = df[FEATURE_COLUMNS + [target_col]].dropna(subset=[target_col])
    if len(valid) < MIN_SAMPLES:
        return {
            "success": False,
            "message": f"Not enough rows with {target_col} for training.",
        }

    X = valid[FEATURE_COLUMNS].values
    y = valid[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if len(valid) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
    else:
        X_train, y_train = X_scaled, y
        X_test, y_test = X_scaled, y

    model = GradientBoostingRegressor(
        n_estimators=80,
        max_depth=4,
        random_state=42,
    )
    model.fit(X_train, y_train)
    train_r2 = float(model.score(X_train, y_train))
    test_r2 = float(model.score(X_test, y_test)) if len(valid) >= 10 else train_r2

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "scaler": scaler, "features": FEATURE_COLUMNS}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    meta = {
        "target": target_col,
        "samples": len(valid),
        "train_r2": round(train_r2, 4),
        "test_r2": round(test_r2, 4),
        "model_type": "GradientBoostingRegressor",
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {"success": True, **meta}


def predict_if_available(features: dict[str, Any]) -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    import numpy as np

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    scaler = bundle["scaler"]
    cols = bundle["features"]

    row = np.array([[float(features.get(c, 0) or 0) for c in cols]])
    X = scaler.transform(row)
    pred = float(model.predict(X)[0])

    if bundle.get("target") == "retention_pct":
        viral = pred
    else:
        viral = pred

    return {
        "viral_score": round(min(100.0, max(0.0, viral)), 1),
        "is_ml_predicted": True,
    }
