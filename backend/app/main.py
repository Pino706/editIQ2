import os
import shutil
import tempfile
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.database import init_db, save_video_analysis, get_all_analyses, get_analysis
from backend.app.analyzer import analyze_video_file
from backend.app.scorer import calculate_scores
from backend.app.ml_model import train_ml_model, predict_retention, get_model_info

# Initialize database on startup
init_db()

app = FastAPI(
    title="EditIQ API",
    description="Backend API for TikTok Video Edit Analyzer & Viral Performance Estimator",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temp uploads directory
TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/api/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    views: Optional[int] = Form(None),
    likes: Optional[int] = Form(None),
    shares: Optional[int] = Form(None),
    saves: Optional[int] = Form(None),
    retention_pct: Optional[float] = Form(None)
):
    # Validate format
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(status_code=400, detail="Formato file non supportato. Carica un video MP4, MOV, AVI o MKV.")

    # Save uploaded file temporarily
    temp_file_path = os.path.join(TEMP_DIR, f"{datetime.now().timestamp()}_{file.filename}")
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel salvataggio temporaneo del file: {str(e)}")

    try:
        # Perform feature extraction
        features = analyze_video_file(temp_file_path)
        
        # Calculate heuristic scores
        analysis_report = calculate_scores(features)
        
        # Try to predict retention using the ML model if trained
        predicted_retention = predict_retention(features)
        is_ml_predicted = 0
        
        # If ML model predicted a retention score, we inject it into the report
        if predicted_retention is not None:
            # We can use the ML prediction to adjust the heuristic viral score (e.g. average them)
            # This shows the model's output in action
            is_ml_predicted = 1
            analysis_report["predicted_retention"] = round(predicted_retention, 1)
            # Calibrate viral score with ML prediction (optional metric)
            # If retention is high, viral score should reflect it
            analysis_report["viral_score"] = round(0.5 * analysis_report["viral_score"] + 0.5 * predicted_retention, 1)
        else:
            analysis_report["predicted_retention"] = None

        # Prepare database entry
        db_data = {
            "filename": file.filename,
            "timestamp": datetime.now().isoformat(),
            "duration": features["duration"],
            "views": views,
            "likes": likes,
            "shares": shares,
            "saves": saves,
            "retention_pct": retention_pct,
            
            # Visual features
            "cuts_count": features["cuts_count"],
            "cuts_per_second": features["cuts_per_second"],
            "avg_scene_duration": features["avg_scene_duration"],
            "motion_intensity": features["motion_intensity"],
            "visual_change_rate": features["visual_change_rate"],
            "visual_stability": features["visual_stability"],
            "hook_speed": features["hook_speed"],
            
            # Audio features
            "audio_energy": features["audio_energy"],
            "audio_spikes_count": features["audio_spikes_count"],
            "audio_pacing": features["audio_pacing"],
            "av_sync_score": features["av_sync_score"],
            
            # Computed scores
            "viral_score": analysis_report["viral_score"],
            "hook_score": analysis_report["hook_score"],
            "pacing_score": analysis_report["pacing_score"],
            "retention_risk": analysis_report["retention_risk"],
            "feedback": analysis_report["feedback"],
            "is_ml_predicted": is_ml_predicted
        }

        # Save to database
        db_id = save_video_analysis(db_data)
        
        # Add details to response
        response_data = {
            "id": db_id,
            **db_data,
            "predicted_retention": analysis_report["predicted_retention"]
        }
        
        return response_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Errore durante l'analisi del video: {str(e)}")
        
    finally:
        # Cleanup uploaded video file from disk
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

@app.get("/api/history")
async def get_history():
    try:
        return get_all_analyses()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel recupero dello storico: {str(e)}")

@app.get("/api/analysis/{video_id}")
async def get_single_analysis(video_id: int):
    analysis = get_analysis(video_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analisi non trovata.")
    return analysis

@app.post("/api/train")
async def train_model():
    result = train_ml_model()
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.get("/api/model-status")
async def get_model_status():
    return get_model_info()

# Mount frontend files
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
os.makedirs(FRONTEND_DIR, exist_ok=True)

# Serves the index.html at root
@app.get("/")
async def read_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "EditIQ API is running. Build the frontend/index.html to view the dashboard."}

# Mount static files directory under /static
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
