import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from backend.app.database import get_all_analyses_df

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_model.pkl")

# Features list used for training and inference
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
    "av_sync_score"
]

def train_ml_model() -> dict:
    """
    Retrieves all video records from the database, filters for those with a
    retention_pct label, trains a RandomForestRegressor, and saves it.
    """
    df = get_all_analyses_df()
    
    if df.empty:
        return {"success": False, "message": "Il database è vuoto. Carica e analizza prima alcuni video."}
        
    # We need records where retention_pct is filled in manually by the user
    labeled_df = df[df["retention_pct"].notna()]
    
    if len(labeled_df) < 5:
        return {
            "success": False,
            "message": f"Dati insufficienti per addestrare il modello ML. Servono almeno 5 video con la percentuale di retention (ora ne hai {len(labeled_df)})."
        }
        
    X = labeled_df[FEATURE_COLUMNS]
    y = labeled_df["retention_pct"]
    
    # Train RandomForest Regressor
    # We use a simple model suited for small datasets
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # Save the trained model to disk
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
    except Exception as e:
        return {"success": False, "message": f"Errore nel salvataggio del modello: {str(e)}"}
        
    # Calculate training score
    r2_score = model.score(X, y)
    
    return {
        "success": True,
        "message": f"Modello ML addestrato con successo su {len(labeled_df)} video! (R² Score: {r2_score:.2f})",
        "sample_count": len(labeled_df),
        "r2_score": float(r2_score)
    }

def predict_retention(features: dict) -> float:
    """
    Predicts the retention percentage using the trained Random Forest model.
    Returns None if the model is not trained yet.
    """
    if not os.path.exists(MODEL_PATH):
        return None
        
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
            
        # Prepare input features vector
        input_data = [features[col] for col in FEATURE_COLUMNS]
        input_arr = np.array(input_data).reshape(1, -1)
        
        # Predict
        predicted_val = model.predict(input_arr)[0]
        return float(max(0.0, min(100.0, predicted_val)))
    except Exception:
        # Fallback to None if loading or prediction fails
        return None

def get_model_info() -> dict:
    """
    Returns the training status and metadata of the ML model.
    """
    is_trained = os.path.exists(MODEL_PATH)
    
    # Check how many labeled samples exist in the DB
    try:
        df = get_all_analyses_df()
        labeled_count = int(df["retention_pct"].notna().sum()) if not df.empty else 0
        total_count = len(df) if not df.empty else 0
    except Exception:
        labeled_count = 0
        total_count = 0
        
    return {
        "is_trained": is_trained,
        "labeled_count": labeled_count,
        "total_count": total_count,
        "required_labeled": 5,
        "features_used": FEATURE_COLUMNS
    }
