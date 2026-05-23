"""Heuristic viral-potential scoring for TikTok edits."""

from __future__ import annotations

from typing import Any

from app.tiktok_knowledge import algorithm_tips


def score_features(features: dict[str, Any]) -> dict[str, Any]:
    hook = _hook_score(features)
    pacing = _pacing_score(features)
    motion = _motion_score(features)
    av_sync = float(features.get("av_sync_score", 0.5))
    retention_risk = _retention_risk(features)

    viral = (
        0.30 * hook
        + 0.25 * pacing
        + 0.25 * motion
        + 0.20 * (av_sync * 100)
    )
    viral = round(min(100.0, max(0.0, viral)), 1)

    feedback = _build_feedback(features, hook, pacing, motion, av_sync, retention_risk)

    return {
        "viral_score": viral,
        "hook_score": round(hook, 1),
        "pacing_score": round(pacing, 1),
        "retention_risk": round(retention_risk, 1),
        "feedback": feedback,
    }


def _hook_score(features: dict[str, Any]) -> float:
    hook_speed = float(features.get("hook_speed", 3.0))
    motion = float(features.get("motion_intensity", 0.0))

    if hook_speed <= 0.5:
        speed_score = 100.0
    elif hook_speed <= 1.0:
        speed_score = 90.0
    elif hook_speed <= 2.0:
        speed_score = 70.0
    elif hook_speed <= 3.0:
        speed_score = 45.0
    else:
        speed_score = max(10.0, 40.0 - (hook_speed - 3.0) * 10)

    early_motion = min(100.0, motion * 400)
    return 0.65 * speed_score + 0.35 * early_motion


def _pacing_score(features: dict[str, Any]) -> float:
    cps = float(features.get("cuts_per_second", 0.0))
    optimal_low, optimal_high = 1.0, 2.5

    if optimal_low <= cps <= optimal_high:
        return 95.0
    if cps < optimal_low:
        if cps < 0.3:
            return 25.0
        return 25.0 + (cps / optimal_low) * 55.0
    if cps <= 4.0:
        return max(50.0, 95.0 - (cps - optimal_high) * 20.0)
    return max(30.0, 60.0 - (cps - 4.0) * 15.0)


def _motion_score(features: dict[str, Any]) -> float:
    motion = float(features.get("motion_intensity", 0.0))
    stability = float(features.get("visual_stability", 0.5))
    base = min(100.0, motion * 350)
    return 0.75 * base + 0.25 * (stability * 100)


def _retention_risk(features: dict[str, Any]) -> float:
    cps = float(features.get("cuts_per_second", 0.0))
    motion = float(features.get("motion_intensity", 0.0))
    avg_scene = float(features.get("avg_scene_duration", 5.0))

    risk = 20.0
    if avg_scene > 4.0:
        risk += min(40.0, (avg_scene - 4.0) * 12)
    if cps < 0.5:
        risk += 25.0
    elif cps < 1.0:
        risk += 10.0
    if motion < 0.03:
        risk += 20.0
    elif motion < 0.06:
        risk += 8.0
    return min(100.0, risk)


def _build_feedback(
    features: dict[str, Any],
    hook: float,
    pacing: float,
    motion: float,
    av_sync: float,
    retention_risk: float,
) -> dict[str, list[str]]:
    strengths: list[str] = []
    improvements: list[str] = []

    hook_speed = float(features.get("hook_speed", 3.0))
    cps = float(features.get("cuts_per_second", 0.0))

    if hook >= 75:
        strengths.append("Strong intro hook — first cut lands early.")
    elif hook_speed > 2.0:
        improvements.append(
            f"Hook is slow ({hook_speed:.1f}s to first cut). Aim for a cut within 1–2s."
        )

    if pacing >= 75:
        strengths.append(f"Solid pacing at {cps:.2f} cuts/sec (TikTok sweet spot).")
    elif cps < 1.0:
        improvements.append(
            "Transitions are too rare — pacing feels slow. Target 1–2.5 cuts/sec."
        )
    elif cps > 2.5:
        improvements.append(
            "Very fast cuts — consider breathing room so viewers can follow."
        )

    if motion >= 60:
        strengths.append("Good motion energy keeps the edit dynamic.")
    elif motion < 40:
        improvements.append("Low motion intensity — add movement or punch-ins.")

    if av_sync >= 0.6:
        strengths.append("Cuts align well with audio beats.")
    else:
        improvements.append(
            "Audio–video sync is weak — snap cuts to beat drops within ~150ms."
        )

    if retention_risk < 35:
        strengths.append("Low retention risk from pacing gaps.")
    elif retention_risk >= 60:
        improvements.append(
            "High retention risk — long static scenes or slow sections detected."
        )

    if not strengths:
        strengths.append("Baseline edit detected — room to optimize hook and pacing.")
    if not improvements:
        improvements.append("Edit looks well-balanced. A/B test on TikTok for validation.")

    for tip in algorithm_tips(features):
        if tip not in improvements:
            improvements.append(tip)

    return {"strengths": strengths, "improvements": improvements}
