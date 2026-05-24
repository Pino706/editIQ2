"""FastAPI application for EditIQ."""

from __future__ import annotations

import csv
import importlib.util
import io
import os
import re
import tempfile
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field

from app import database
from app import tiktok_client
from app.account_intelligence import get_account_dashboard
from app.action_plan import build_action_plan
from app.analyzer import analyze_video, probe_video_metadata
from app import ml_model
from app.scorer import score_features
from app import views_model
from app.config import settings
from app.dataset_quality import build_dataset_quality
from app.feature_insights import build_feature_insights
from app.intelligence import build_prediction_package
from app.prediction_explain import build_views_justification
from app.text_model import generate_analysis_advice
from app.tiktok_knowledge import doc_loaded, load_algorithm_doc
from app.view_tiers import predict_tier_probabilities

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = (
    PROJECT_ROOT / "frontend"
    if (PROJECT_ROOT / "frontend" / "index.html").exists()
    else PROJECT_ROOT / "static"
)
TIKTOK_VERIFICATION_FILE = "tiktokznm3K87KIom2s7CoSPfgatDPVse1jFFD.txt"

app = FastAPI(title="EditIQ", description="TikTok Edit Analyzer", version="1.0.0")

_VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
MAX_UPLOAD_BYTES = int(os.getenv("EDITIQ_MAX_UPLOAD_MB", "250")) * 1024 * 1024
MAX_VIDEO_SECONDS = float(os.getenv("EDITIQ_MAX_VIDEO_SECONDS", "180"))


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


def _copy_upload_with_limit(src, dest: Path, limit_bytes: int = MAX_UPLOAD_BYTES) -> int:
    total = 0
    with open(dest, "wb") as out:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit_bytes:
                raise HTTPException(
                    413,
                    f"Video is too large. Limit is {limit_bytes // (1024 * 1024)}MB.",
                )
            out.write(chunk)
    return total


def _optional_text(value: str | None, max_len: int = 2000) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned[:max_len] if cleaned else None


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
    style_profile: Annotated[str | None, Form()] = None,
    niche: Annotated[str | None, Form()] = None,
    caption: Annotated[str | None, Form()] = None,
    hashtags: Annotated[str | None, Form()] = None,
    sound_name: Annotated[str | None, Form()] = None,
    posted_at: Annotated[str | None, Form()] = None,
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
            "style_profile": _optional_text(style_profile, 120),
            "niche": _optional_text(niche, 120),
            "caption": _optional_text(caption, 2000),
            "hashtags": _optional_text(hashtags, 500),
            "sound_name": _optional_text(sound_name, 180),
            "posted_at": _optional_text(posted_at, 80),
        }
    except HTTPException:
        raise

    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / f"upload{suffix}"
        _copy_upload_with_limit(file.file, video_path)

        try:
            probe = probe_video_metadata(video_path)
        except Exception as exc:
            raise HTTPException(400, f"Invalid or unreadable video file: {exc}") from exc
        if probe["duration"] and probe["duration"] > MAX_VIDEO_SECONDS:
            raise HTTPException(
                413,
                f"Video is too long ({probe['duration']:.1f}s). Limit is {MAX_VIDEO_SECONDS:.0f}s.",
            )

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
    action_plan = build_action_plan(features, scores)
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
    response["action_plan"] = action_plan
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
    row_scores = {
        "viral_score": row.get("viral_score"),
        "hook_score": row.get("hook_score"),
        "pacing_score": row.get("pacing_score"),
        "retention_risk": row.get("retention_risk"),
    }
    row["action_plan"] = build_action_plan(
        {**features, "timeline": row.get("timeline") or {}},
        row_scores,
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


@app.get("/api/dataset/quality")
def dataset_quality():
    return build_dataset_quality()


@app.get("/api/export/dataset.csv")
def export_dataset_csv(labeled_only: bool = False):
    rows = database.list_export_rows(labeled_only=labeled_only)
    out = io.StringIO()
    fields = [
        "id",
        "filename",
        "timestamp",
        "duration",
        "views",
        "likes",
        "shares",
        "saves",
        "retention_pct",
        "predicted_views",
        "style_profile",
        "niche",
        "caption",
        "hashtags",
        "sound_name",
        "posted_at",
        *views_model.FEATURE_COLUMNS,
        "viral_score",
        "hook_score",
        "pacing_score",
        "retention_risk",
    ]
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        merged = {**row, **database.merge_features(row)}
        writer.writerow({k: merged.get(k) for k in fields})
    payload = out.getvalue()
    return StreamingResponse(
        iter([payload]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=editiq-dataset.csv"},
    )


@app.get("/api/export/history.json")
def export_history_json(labeled_only: bool = False):
    return {"items": database.list_export_rows(labeled_only=labeled_only)}


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
    views: int | None = Field(None, gt=0)
    likes: int | None = Field(None, ge=0)
    shares: int | None = Field(None, ge=0)
    saves: int | None = Field(None, ge=0)
    retention_pct: float | None = Field(None, ge=0, le=100)
    style_profile: str | None = Field(None, max_length=120)
    niche: str | None = Field(None, max_length=120)
    caption: str | None = Field(None, max_length=2000)
    hashtags: str | None = Field(None, max_length=500)
    sound_name: str | None = Field(None, max_length=180)
    posted_at: str | None = Field(None, max_length=80)


@app.patch("/api/analysis/{video_id}")
def patch_analysis(video_id: int, body: ViewsUpdate):
    fields = body.model_dump(exclude_unset=True)
    if not database.update_analysis_labels(video_id, fields):
        raise HTTPException(404, "Analysis not found")
    return {"id": video_id, "analysis": database.get_analysis(video_id)}


@app.get("/api/health")
def health():
    checks: dict[str, dict] = {}
    checks["database"] = {
        "ok": database.DB_PATH.exists() or database.DB_PATH.parent.exists(),
        "path": str(database.DB_PATH),
        "persistent_warning": bool(os.getenv("VERCEL") and "tmp" in str(database.DB_PATH).lower()),
    }
    checks["models"] = {
        "ok": views_model.MODEL_DIR.exists() or views_model.MODEL_DIR.parent.exists(),
        "path": str(views_model.MODEL_DIR),
        "views_model": views_model.MODEL_PATH.exists(),
    }
    for mod in ["cv2", "numpy", "pandas", "sklearn", "catboost", "httpx", "cryptography"]:
        checks[mod] = {"ok": importlib.util.find_spec(mod) is not None}
    try:
        import imageio_ffmpeg

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        checks["ffmpeg"] = {"ok": Path(ffmpeg_path).exists(), "path": ffmpeg_path}
    except Exception as exc:
        checks["ffmpeg"] = {"ok": False, "error": str(exc)}
    checks["tiktok"] = {"ok": settings.tiktok_configured, "configured": settings.tiktok_configured}
    checks["ollama"] = {"ok": bool(settings.ollama_base_url), "url": settings.ollama_base_url}
    ok = all(v.get("ok", False) for k, v in checks.items() if k not in {"tiktok", "ollama"})
    return {
        "ok": ok,
        "environment": "vercel" if os.getenv("VERCEL") else "local",
        "checks": checks,
    }


@app.get("/api/tiktok/login")
def tiktok_login():
    if not settings.tiktok_configured:
        raise HTTPException(503, "TikTok OAuth is not configured on the server.")
    if not settings.token_encryption_key:
        raise HTTPException(503, "TOKEN_ENCRYPTION_KEY is required before connecting TikTok.")
    try:
        return RedirectResponse(tiktok_client.build_login_url())
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
        detail = quote(str(exc) or type(exc).__name__)
        return RedirectResponse(f"/?tiktok=error&reason={type(exc).__name__}&detail={detail}")
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


@app.get(f"/{TIKTOK_VERIFICATION_FILE}")
async def tiktok_verification_file():
    verification_path = STATIC_DIR / TIKTOK_VERIFICATION_FILE
    if not verification_path.exists():
        raise HTTPException(404, "Verification file not found")
    return FileResponse(verification_path, media_type="text/plain; charset=utf-8")


@app.get("/{page_name}")
async def static_page(page_name: str):
    allowed_pages = {
        "privacy": "privacy.html",
        "cookie-policy": "cookie-policy.html",
        "terms": "terms.html",
        "do-not-sell": "do-not-sell.html",
    }
    if page_name not in allowed_pages:
        raise HTTPException(404, "Page not found")
    page_path = STATIC_DIR / allowed_pages[page_name]
    if not page_path.exists():
        raise HTTPException(404, "Page not found")
    return FileResponse(page_path)


@app.get("/")
async def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(404, "Frontend not found")
    return FileResponse(index_path)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
