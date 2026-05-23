"""FastAPI application for EditIQ."""

from __future__ import annotations

import re
import secrets
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field

from app import database
from app import tiktok_client
from app.account_intelligence import get_account_dashboard
from app.analyzer import analyze_video
from app import ml_model
from app.scorer import score_features
from app import views_model
from app.config import settings
from app.feature_insights import build_feature_insights
from app.intelligence import build_prediction_package
from app.prediction_explain import build_views_justification
from app.text_model import generate_analysis_advice
from app.tiktok_knowledge import doc_loaded, load_algorithm_doc
from app.view_tiers import predict_tier_probabilities

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="EditIQ", description="TikTok Edit Analyzer", version="1.0.0")

_VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi"}


def _parse_int_form(value: str | None, field: str = "value") -> int | None:
    if value is None or not str(value).strip():
        return None
    s = str(value).strip().replace(",", "").replace(" ", "")
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    try:
        return int(float(s))
    except ValueError as exc:
        raise HTTPException(
            400, f"Invalid {field}: use digits only (e.g. 50000), not '{value}'."
        ) from exc


def _resolve_suffix(filename: str, content_type: str | None) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in _VIDEO_SUFFIXES:
        return suffix
    if not suffix and content_type and content_type.startswith("video/"):
        sub = content_type.split("/")[-1].lower()
        mapping = {"quicktime": ".mov", "x-msvideo": ".avi", "webm": ".webm"}
        return mapping.get(sub, ".mp4")
    if not suffix:
        return ".mp4"
    return suffix


@app.on_event("startup")
def startup() -> None:
    database.init_db()


def _optional_int(value: str | None, field: str = "value") -> int | None:
    return _parse_int_form(value, field)


def _optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    views: Annotated[str | None, Form()] = None,
    likes: Annotated[str | None, Form()] = None,
    shares: Annotated[str | None, Form()] = None,
    saves: Annotated[str | None, Form()] = None,
    retention_pct: Annotated[str | None, Form()] = None,
):
    if not file.filename:
        raise HTTPException(400, "No file provided")
    suffix = _resolve_suffix(file.filename, file.content_type)
    if suffix not in _VIDEO_SUFFIXES:
        raise HTTPException(400, "Unsupported format. Use MP4/MOV/WebM.")

    try:
        metadata = {
            "views": _optional_int(views, "views") if views else None,
            "likes": _optional_int(likes, "likes") if likes else None,
            "shares": _optional_int(shares, "shares") if shares else None,
            "saves": _optional_int(saves, "saves") if saves else None,
            "retention_pct": _optional_float(retention_pct),
        }
    except HTTPException:
        raise

    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / f"upload{suffix}"
        with open(video_path, "wb") as out:
            shutil.copyfileobj(file.file, out)

        try:
            analysis = analyze_video(video_path)
        except Exception as exc:
            raise HTTPException(422, f"Analysis failed: {exc}") from exc

    features = analysis.to_features()
    features["duration"] = analysis.duration
    scores = score_features(features)

    try:
        views_pred = views_model.predict_views(features)
    except Exception:
        views_pred = None

    prediction = build_prediction_package(features, scores, views_pred)
    predicted_views = prediction["predicted_views"]
    metadata["predicted_views"] = predicted_views
    tier_probs = predict_tier_probabilities(features, predicted_views)
    feature_insights = build_feature_insights(features)
    justification = build_views_justification(
        features, predicted_views, feature_insights
    )
    ai_advice = await generate_analysis_advice(
        {
            "scores": scores,
            "prediction": prediction,
            "feature_insights": feature_insights,
            "account_context": {
                "account_fit_score": prediction.get("account_fit_score"),
                "trend_fit_score": prediction.get("trend_fit_score"),
                "account_reason": prediction.get("account_reason"),
                "why_prediction_changed": prediction.get("why_prediction_changed"),
            },
        }
    )

    record = database.build_record(
        filename=file.filename,
        duration=analysis.duration,
        features=features,
        scores=scores,
        metadata=metadata,
        is_ml=False,
    )
    video_id = database.insert_analysis(record)

    response: dict = {
        "id": video_id,
        "filename": file.filename,
        "duration": analysis.duration,
        "features": {k: v for k, v in features.items() if k != "timeline"},
        "scores": scores,
        "timeline": features.get("timeline", {}),
        "metadata": metadata,
        "is_ml_predicted": False,
        "views_model_ready": views_pred is not None,
        "prediction": prediction,
        "ai_advice": ai_advice,
    }
    if views_pred:
        response.update(views_pred)
    else:
        status = views_model.get_model_status()
        response["views_model_message"] = status.get("message", "")
        response["prediction_source"] = "heuristic_estimate"

    response["predicted_views"] = predicted_views
    response["prediction_range_low"] = prediction["prediction_range_low"]
    response["prediction_range_high"] = prediction["prediction_range_high"]
    response["tier_probabilities"] = tier_probs
    response["feature_insights"] = feature_insights
    response["views_justification"] = justification
    response["algorithm_doc_loaded"] = doc_loaded()

    return response


@app.get("/api/history")
def history(limit: int = 100):
    return {"items": database.list_history(limit=limit)}


@app.get("/api/analysis/{video_id}")
async def get_analysis(video_id: int):
    row = database.get_analysis(video_id)
    if row is None:
        raise HTTPException(404, "Analysis not found")
    features = database.merge_features(row)
    features["duration"] = row.get("duration", 0)
    pred = row.get("predicted_views")
    stored_prediction = (
        {
            "predicted_views": pred,
            "prediction_source": "stored_prediction",
            "training_samples": views_model.get_model_status().get("samples"),
        }
        if pred is not None
        else None
    )
    row["prediction"] = build_prediction_package(
        features,
        {
            "viral_score": row.get("viral_score"),
            "hook_score": row.get("hook_score"),
            "pacing_score": row.get("pacing_score"),
            "retention_risk": row.get("retention_risk"),
        },
        stored_prediction,
    )
    effective_pred = row["prediction"]["predicted_views"]
    row["predicted_views"] = effective_pred
    row["prediction_range_low"] = row["prediction"]["prediction_range_low"]
    row["prediction_range_high"] = row["prediction"]["prediction_range_high"]
    row["feature_insights"] = build_feature_insights(features)
    row["tier_probabilities"] = predict_tier_probabilities(features, effective_pred)
    row["views_justification"] = build_views_justification(
        features, effective_pred, row["feature_insights"]
    )
    row["ai_advice"] = await generate_analysis_advice(
        {
            "scores": row,
            "prediction": row["prediction"],
            "feature_insights": row["feature_insights"],
            "account_context": row["prediction"],
        }
    )
    return row


@app.post("/api/train")
def train(use_retention_target: bool = False):
    result = ml_model.train_model(use_retention_target=use_retention_target)
    if not result.get("success"):
        raise HTTPException(400, result.get("message", "Training failed"))
    return result


@app.get("/api/model-status")
def model_status():
    return ml_model.get_model_status()


@app.get("/api/dataset/stats")
def dataset_stats():
    return database.dataset_stats()


@app.get("/api/dataset/labeled")
def dataset_labeled():
    return {"items": database.list_labeled_summary()}


@app.post("/api/train-views")
def train_views():
    result = views_model.train_model()
    if not result.get("success"):
        raise HTTPException(400, result.get("message", "Training failed"))
    return result


@app.get("/api/views-model-status")
def views_model_status():
    return views_model.get_model_status()


@app.get("/api/algorithm-doc")
def algorithm_doc():
    text = load_algorithm_doc()
    return {
        "loaded": doc_loaded(),
        "path": "tiktokalgorithm.txt",
        "chars": len(text),
        "excerpt": text[:600] + ("…" if len(text) > 600 else ""),
    }


class ViewsUpdate(BaseModel):
    views: int = Field(..., gt=0)


@app.patch("/api/analysis/{video_id}")
def patch_analysis(video_id: int, body: ViewsUpdate):
    if not database.update_views(video_id, body.views):
        raise HTTPException(404, "Analysis not found")
    return {"id": video_id, "views": body.views}


@app.get("/api/tiktok/login")
def tiktok_login():
    if not settings.tiktok_configured:
        raise HTTPException(503, "TikTok OAuth is not configured on the server.")
    if not settings.token_encryption_key:
        raise HTTPException(503, "TOKEN_ENCRYPTION_KEY is required before connecting TikTok.")
    state = secrets.token_urlsafe(32)
    try:
        return RedirectResponse(tiktok_client.build_login_url(state))
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/api/tiktok/callback")
async def tiktok_callback(code: str | None = None, state: str | None = None, error: str | None = None):
    if error:
        return RedirectResponse(f"/?tiktok=error&reason={error}")
    if not code or not state:
        return RedirectResponse("/?tiktok=error&reason=missing_code")
    try:
        await tiktok_client.complete_oauth(code=code, state=state)
    except Exception as exc:
        return RedirectResponse(f"/?tiktok=error&reason={type(exc).__name__}")
    return RedirectResponse("/?tiktok=connected")


@app.get("/api/tiktok/account")
def tiktok_account():
    dashboard = get_account_dashboard()
    dashboard["configured"] = settings.tiktok_configured
    return dashboard


@app.get("/api/tiktok/videos")
def tiktok_videos():
    dashboard = get_account_dashboard()
    return {
        "connected": dashboard.get("connected", False),
        "videos": dashboard.get("videos", []),
        "summary": dashboard.get("summary"),
    }


@app.post("/api/tiktok/refresh")
async def tiktok_refresh():
    try:
        await tiktok_client.refresh_connected_account()
    except Exception as exc:
        raise HTTPException(502, str(exc)) from exc
    dashboard = get_account_dashboard()
    dashboard["configured"] = settings.tiktok_configured
    return dashboard


@app.post("/api/tiktok/disconnect")
def tiktok_disconnect():
    database.disconnect_tiktok_account()
    return {"connected": False}


@app.get("/")
async def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(404, "Frontend not found")
    return FileResponse(index_path)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
