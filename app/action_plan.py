"""Actionable editing guidance derived from timelines and score bottlenecks."""

from __future__ import annotations

from typing import Any


def build_action_plan(
    features: dict[str, Any],
    scores: dict[str, Any],
) -> dict[str, Any]:
    timeline = features.get("timeline") or {}
    cut_times = [float(t) for t in timeline.get("cut_times", [])]
    motion = timeline.get("motion", []) or []
    audio = timeline.get("audio_energy", []) or []
    spikes = [float(t) for t in timeline.get("audio_spike_times", [])]
    duration = float(features.get("duration") or 0.0)

    fixes = _priority_fixes(features, scores, cut_times, motion, spikes, duration)
    heatmap = _heatmap_segments(features, cut_times, motion, audio, spikes, duration)
    summaries = {
        "motion": _motion_summary(motion),
        "audio": _audio_summary(audio, spikes),
        "timeline": _timeline_summary(cut_times, duration),
    }
    return {
        "fixes": fixes[:5],
        "timeline_heatmap": heatmap,
        "chart_summaries": summaries,
    }


def _priority_fixes(
    features: dict[str, Any],
    scores: dict[str, Any],
    cut_times: list[float],
    motion: list[dict[str, Any]],
    spikes: list[float],
    duration: float,
) -> list[dict[str, Any]]:
    fixes: list[dict[str, Any]] = []
    hook_speed = float(features.get("hook_speed") or 3.0)
    cps = float(features.get("cuts_per_second") or 0.0)
    av_sync = float(features.get("av_sync_score") or 0.0)
    retention_risk = float(scores.get("retention_risk") or 0.0)
    motion_intensity = float(features.get("motion_intensity") or 0.0)
    longest_gap = float(features.get("longest_scene_gap") or 0.0)
    silence_ratio = float(features.get("silence_ratio") or 0.0)

    if hook_speed > 1.5:
        fixes.append(
            _fix(
                98,
                "Move the first visual change earlier",
                "Hook is the first ranking gate. Add a cut, zoom, text reveal, or motion spike inside the first 1.0s.",
                0.0,
                min(2.0, max(hook_speed, 1.0)),
                "hook",
            )
        )
    elif hook_speed <= 0.8:
        fixes.append(
            _fix(
                74,
                "Preserve the fast hook",
                "The opening already changes quickly. Do not add an empty intro before export.",
                0.0,
                min(1.2, duration or 1.2),
                "protect",
            )
        )

    if longest_gap > 3.0:
        gap = _longest_gap(cut_times, duration)
        fixes.append(
            _fix(
                92,
                "Break the longest dead zone",
                f"Longest scene gap is {longest_gap:.1f}s. Add a cut, punch-in, caption beat, or B-roll inside this interval.",
                gap["start"],
                gap["end"],
                "retention",
            )
        )

    if cps < 0.8:
        fixes.append(
            _fix(
                88,
                "Increase visual rhythm",
                f"Pacing is {cps:.2f} cuts/sec. For your first pass, target 1.0-2.2 cuts/sec without making the action unreadable.",
                0.0,
                min(duration, 6.0),
                "pacing",
            )
        )
    elif cps > 3.2:
        fixes.append(
            _fix(
                71,
                "Add one breathing beat",
                f"Pacing is very fast at {cps:.2f} cuts/sec. Keep intensity, but give key moments a readable frame.",
                0.0,
                min(duration, 8.0),
                "clarity",
            )
        )

    if av_sync < 0.55 and spikes:
        first_spike = spikes[0]
        fixes.append(
            _fix(
                86,
                "Snap key cuts to beat peaks",
                "Audio-video sync is limiting perceived quality. Align major cuts within roughly 150ms of visible beat spikes.",
                max(0.0, first_spike - 0.25),
                min(duration, first_spike + 0.25),
                "sync",
            )
        )

    if motion_intensity < 0.055:
        low = _lowest_motion_window(motion)
        fixes.append(
            _fix(
                82,
                "Add motion density",
                "Motion is low for short-form edits. Add camera movement, speed ramp, animated text, or a tighter crop.",
                low["start"],
                low["end"],
                "motion",
            )
        )

    if silence_ratio > 0.22:
        fixes.append(
            _fix(
                76,
                "Remove silent drag",
                f"Silence ratio is {silence_ratio:.0%}. Fill dead air or tighten sections where audio energy drops.",
                0.0,
                min(duration, 4.0),
                "audio",
            )
        )

    if retention_risk >= 65:
        fixes.append(
            _fix(
                90,
                "Treat retention as the main blocker",
                "The edit is likely losing viewers before the model can reward later quality. Fix hook, gaps, and motion before polishing effects.",
                0.0,
                min(duration, 5.0),
                "retention",
            )
        )

    fixes.sort(key=lambda x: x["priority"], reverse=True)
    return fixes


def _heatmap_segments(
    features: dict[str, Any],
    cut_times: list[float],
    motion: list[dict[str, Any]],
    audio: list[dict[str, Any]],
    spikes: list[float],
    duration: float,
) -> list[dict[str, Any]]:
    if duration <= 0:
        duration = max([float(p.get("t", 0)) for p in motion + audio] or [10.0])
    segments: list[dict[str, Any]] = []
    hook_speed = float(features.get("hook_speed") or 3.0)
    if hook_speed > 1.5:
        segments.append(_segment(0, min(duration, hook_speed), 0.92, "Slow hook", "First visual change arrives late."))
    else:
        segments.append(_segment(0, min(duration, 1.2), 0.22, "Fast hook", "Opening change is early."))

    gap = _longest_gap(cut_times, duration)
    if gap["end"] - gap["start"] > 3.0:
        segments.append(_segment(gap["start"], gap["end"], 0.82, "Long gap", "Static interval can hurt retention."))

    low = _lowest_motion_window(motion)
    if low["severity"] > 0.62:
        segments.append(_segment(low["start"], low["end"], low["severity"], "Low motion", "Motion dips below your edit average."))

    if spikes and cut_times:
        missed = _missed_sync_windows(cut_times, spikes, duration)
        segments.extend(missed[:3])

    if len(segments) < 3:
        thirds = max(duration / 3, 1.0)
        for i in range(3):
            start = i * thirds
            segments.append(_segment(start, min(duration, start + thirds), 0.34, "Baseline", "No major risk detected here."))

    return sorted(segments, key=lambda x: x["start"])


def _fix(priority: int, title: str, detail: str, start: float, end: float, kind: str) -> dict[str, Any]:
    return {
        "priority": priority,
        "title": title,
        "detail": detail,
        "start": round(max(0.0, start), 2),
        "end": round(max(start, end), 2),
        "kind": kind,
    }


def _segment(start: float, end: float, severity: float, label: str, reason: str) -> dict[str, Any]:
    sev = max(0.0, min(1.0, severity))
    return {
        "start": round(max(0.0, start), 2),
        "end": round(max(start, end), 2),
        "severity": round(sev, 2),
        "level": "risk" if sev >= 0.7 else ("watch" if sev >= 0.45 else "good"),
        "label": label,
        "reason": reason,
    }


def _longest_gap(cut_times: list[float], duration: float) -> dict[str, float]:
    points = [0.0] + sorted([t for t in cut_times if t > 0]) + ([duration] if duration > 0 else [])
    if len(points) < 2:
        return {"start": 0.0, "end": max(duration, 0.0)}
    start, end = max(zip(points, points[1:]), key=lambda p: p[1] - p[0])
    return {"start": float(start), "end": float(end)}


def _lowest_motion_window(motion: list[dict[str, Any]], window_s: float = 1.2) -> dict[str, float]:
    if not motion:
        return {"start": 0.0, "end": window_s, "severity": 0.7}
    values = [(float(p.get("t", 0)), float(p.get("v", 0))) for p in motion]
    best = (values[0][0], values[0][0] + window_s, values[0][1])
    for t, _ in values:
        window = [v for mt, v in values if t <= mt <= t + window_s]
        if not window:
            continue
        avg = sum(window) / len(window)
        if avg < best[2]:
            best = (t, t + window_s, avg)
    severity = max(0.0, min(0.95, 0.78 - best[2] * 3.5))
    return {"start": best[0], "end": best[1], "severity": severity}


def _missed_sync_windows(cut_times: list[float], spikes: list[float], duration: float) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for spike in spikes[:8]:
        if not any(abs(spike - cut) <= 0.15 for cut in cut_times):
            segments.append(
                _segment(
                    max(0.0, spike - 0.18),
                    min(duration, spike + 0.18),
                    0.58,
                    "Missed beat",
                    "A beat spike has no nearby cut.",
                )
            )
    return segments


def _motion_summary(motion: list[dict[str, Any]]) -> str:
    if not motion:
        return "No motion curve available for this analysis."
    vals = [float(p.get("v", 0)) for p in motion]
    ts = [float(p.get("t", 0)) for p in motion]
    peak_i = max(range(len(vals)), key=lambda i: vals[i])
    low_i = min(range(len(vals)), key=lambda i: vals[i])
    return f"Motion peaks around {ts[peak_i]:.1f}s and is weakest around {ts[low_i]:.1f}s."


def _audio_summary(audio: list[dict[str, Any]], spikes: list[float]) -> str:
    if not audio:
        return "No audio energy curve was detected."
    vals = [float(p.get("v", 0)) for p in audio]
    ts = [float(p.get("t", 0)) for p in audio]
    peak_i = max(range(len(vals)), key=lambda i: vals[i])
    return f"Audio energy peaks around {ts[peak_i]:.1f}s with {len(spikes)} detected beat spikes."


def _timeline_summary(cut_times: list[float], duration: float) -> str:
    if not cut_times:
        return "No hard cuts were detected, so pacing depends mostly on motion and audio."
    first = cut_times[0]
    return f"{len(cut_times)} cuts detected across {duration:.1f}s. First cut lands at {first:.1f}s."
