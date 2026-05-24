"""Prediction packaging that blends video analysis, ML, and account context."""

from __future__ import annotations

from typing import Any

from app.account_intelligence import get_account_dashboard, score_account_fit


def build_prediction_package(
    features: dict[str, Any],
    scores: dict[str, Any],
    views_prediction: dict[str, Any] | None,
) -> dict[str, Any]:
    account_dashboard = get_account_dashboard()
    account_summary = account_dashboard.get("summary") if account_dashboard.get("connected") else None
    account_fit = score_account_fit(features, account_summary)

    base_views = _base_predicted_views(scores, views_prediction)
    adjusted_views = int(max(0, round(base_views * account_fit["account_adjustment_factor"])))
    interval = _prediction_interval(adjusted_views, views_prediction)
    viral_probability = _viral_probability(adjusted_views, scores, account_fit)
    visual_quality = _visual_quality_score(features)
    overall = _overall_score(scores, account_fit, visual_quality, viral_probability)
    video_strength = round(
        0.4 * float(scores.get("viral_score") or 0)
        + 0.25 * float(scores.get("hook_score") or 0)
        + 0.2 * float(scores.get("pacing_score") or 0)
        + 0.15 * visual_quality,
        1,
    )
    engagement = _engagement_forecast(adjusted_views, account_summary, scores, account_fit)

    return {
        "predicted_views_base": base_views,
        "predicted_views": adjusted_views,
        "prediction_range_low": interval["low"],
        "prediction_range_high": interval["high"],
        "prediction_source": (views_prediction or {}).get("prediction_source", "heuristic_estimate"),
        "viral_probability": viral_probability,
        "overall_score": overall,
        "video_strength_score": video_strength,
        "hook_score": scores.get("hook_score"),
        "retention_risk": scores.get("retention_risk"),
        "account_fit_score": account_fit["account_fit_score"],
        "trend_fit_score": account_fit["trend_fit_score"],
        "visual_quality_score": visual_quality,
        "pacing_score": scores.get("pacing_score"),
        "account_adjustment_factor": account_fit["account_adjustment_factor"],
        "account_reason": account_fit["account_reason"],
        "why_prediction_changed": _why_changed(base_views, adjusted_views, account_fit),
        "final_recommendation": _recommendation(scores, account_fit, viral_probability),
        "account_context_available": bool(account_dashboard.get("connected")),
        "engagement_forecast": engagement,
    }


def _base_predicted_views(scores: dict[str, Any], views_prediction: dict[str, Any] | None) -> int:
    if views_prediction and views_prediction.get("predicted_views") is not None:
        return int(views_prediction["predicted_views"])
    return int(max(500, float(scores.get("viral_score") or 0) * 1000))


def _prediction_interval(predicted: int, views_prediction: dict[str, Any] | None) -> dict[str, int]:
    if views_prediction:
        low = views_prediction.get("prediction_range_low")
        high = views_prediction.get("prediction_range_high")
        if low is not None and high is not None:
            factor = predicted / max(1, int(views_prediction.get("predicted_views") or predicted))
            return {"low": int(max(0, round(low * factor))), "high": int(max(0, round(high * factor)))}
    return {"low": int(predicted * 0.55), "high": int(predicted * 1.65)}


def _viral_probability(predicted_views: int, scores: dict[str, Any], account_fit: dict[str, Any]) -> float:
    view_signal = min(100.0, predicted_views / 2000)
    score_signal = float(scores.get("viral_score") or 0)
    account_signal = account_fit["account_fit_score"] if account_fit["account_fit_score"] is not None else 50
    return round(max(0.0, min(100.0, 0.45 * score_signal + 0.35 * view_signal + 0.2 * account_signal)), 1)


def _visual_quality_score(features: dict[str, Any]) -> float:
    motion = min(100.0, float(features.get("motion_intensity") or 0) * 650)
    stability = min(100.0, float(features.get("visual_stability") or 0) * 100)
    change = min(100.0, float(features.get("visual_change_rate") or 0) * 100)
    color = min(100.0, float(features.get("color_variance") or 0) * 140)
    return round(0.35 * motion + 0.3 * stability + 0.2 * change + 0.15 * color, 1)


def _overall_score(
    scores: dict[str, Any],
    account_fit: dict[str, Any],
    visual_quality: float,
    viral_probability: float,
) -> float:
    account = account_fit["account_fit_score"] if account_fit["account_fit_score"] is not None else 55
    return round(
        0.3 * float(scores.get("viral_score") or 0)
        + 0.22 * float(scores.get("hook_score") or 0)
        + 0.18 * float(scores.get("pacing_score") or 0)
        + 0.15 * visual_quality
        + 0.15 * account
        + 0.1 * viral_probability,
        1,
    )


def _why_changed(base: int, adjusted: int, account_fit: dict[str, Any]) -> str:
    if account_fit["account_fit_score"] is None:
        return "No connected account context was available, so the forecast uses video signals only."
    if adjusted > base:
        return f"Account context lifted the forecast from {_format_views(base)} to {_format_views(adjusted)} because recent momentum and fit are favorable."
    if adjusted < base:
        return f"Account context lowered the forecast from {_format_views(base)} to {_format_views(adjusted)} because recent momentum or fit is weaker."
    return "Account context did not materially change the forecast."


def _recommendation(scores: dict[str, Any], account_fit: dict[str, Any], viral_probability: float) -> str:
    if viral_probability >= 75:
        return "Publish candidate: strong enough to test, then compare actual retention against the forecast."
    if (scores.get("hook_score") or 0) < 60:
        return "Revise before publishing: strengthen the first 1-2 seconds and make the opening change unmistakable."
    if account_fit["account_fit_score"] is not None and account_fit["account_fit_score"] < 50:
        return "Revise for account fit: align pacing and structure with your best recent uploads."
    return "Good test candidate after one polish pass on pacing, audio sync, and dead zones."


def _engagement_forecast(
    predicted_views: int,
    account_summary: dict[str, Any] | None,
    scores: dict[str, Any],
    account_fit: dict[str, Any],
) -> dict[str, Any]:
    if not account_summary or not predicted_views:
        return {
            "source": "fallback",
            "summary": "Connect TikTok to estimate comments, shares, and engagement from your account averages.",
            "predicted_likes": None,
            "predicted_comments": None,
            "predicted_shares": None,
            "expected_engagement_rate_pct": None,
        }

    avg_views = max(1, int(account_summary.get("average_views") or 1))
    avg_likes = int(account_summary.get("average_likes") or 0)
    avg_comments = int(account_summary.get("average_comments") or 0)
    avg_shares = int(account_summary.get("average_shares") or 0)
    quality_lift = 1.0 + ((float(scores.get("viral_score") or 50) - 50) / 100) * 0.18
    fit_lift = 1.0 + (((account_fit.get("account_fit_score") or 50) - 50) / 100) * 0.16
    trend_lift = 1.0 + (((account_fit.get("trend_fit_score") or 50) - 50) / 100) * 0.12
    factor = max(0.55, min(1.55, quality_lift * fit_lift * trend_lift))

    like_rate = avg_likes / avg_views
    comment_rate = avg_comments / avg_views
    share_rate = avg_shares / avg_views
    likes = int(round(predicted_views * like_rate * factor))
    comments = int(round(predicted_views * comment_rate * factor))
    shares = int(round(predicted_views * share_rate * factor))
    er = ((likes + comments + shares) / max(1, predicted_views)) * 100
    return {
        "source": "account_average_trend",
        "summary": (
            "Estimated from your connected account averages, adjusted by current video strength, "
            "account fit, and recent trend."
        ),
        "predicted_likes": likes,
        "predicted_comments": comments,
        "predicted_shares": shares,
        "expected_engagement_rate_pct": round(er, 2),
        "baseline_average_views": avg_views,
        "adjustment_factor": round(factor, 3),
    }


def _format_views(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
