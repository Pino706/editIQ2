"""Local Ollama-powered text layer for explanations and advice."""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import settings


async def generate_analysis_advice(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = _build_prompt(payload)
    try:
        async with httpx.AsyncClient(timeout=18.0) as client:
            res = await client.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.25, "num_predict": 420},
                },
            )
        res.raise_for_status()
        raw = res.json().get("response", "{}")
        parsed = json.loads(raw)
        return _normalize_llm_response(parsed, source="ollama")
    except Exception:
        return _fallback_advice(payload)


def _build_prompt(payload: dict[str, Any]) -> str:
    compact = {
        "scores": payload.get("scores"),
        "prediction": payload.get("prediction"),
        "top_features": payload.get("feature_insights", [])[:8],
        "account": payload.get("account_context"),
    }
    return (
        "You are EditIQ, a TikTok video intelligence analyst. "
        "Return concise JSON only with keys: summary, final_recommendation, "
        "what_is_strong, what_is_weak, what_should_change, account_context_note. "
        "Use practical editing advice, no hype, no guarantees.\n\n"
        f"DATA:\n{json.dumps(compact, ensure_ascii=False)}"
    )


def _normalize_llm_response(data: dict[str, Any], source: str) -> dict[str, Any]:
    def listify(key: str) -> list[str]:
        val = data.get(key, [])
        if isinstance(val, str):
            return [val]
        if isinstance(val, list):
            return [str(x) for x in val[:5]]
        return []

    return {
        "source": source,
        "summary": str(data.get("summary") or ""),
        "final_recommendation": str(data.get("final_recommendation") or ""),
        "what_is_strong": listify("what_is_strong"),
        "what_is_weak": listify("what_is_weak"),
        "what_should_change": listify("what_should_change"),
        "account_context_note": str(data.get("account_context_note") or ""),
    }


def _fallback_advice(payload: dict[str, Any]) -> dict[str, Any]:
    prediction = payload.get("prediction", {})
    scores = payload.get("scores", {})
    account = payload.get("account_context") or {}
    strong = []
    weak = []
    changes = []
    if (scores.get("hook_score") or 0) >= 70:
        strong.append("The opening hook is strong enough to support the first retention gate.")
    else:
        weak.append("The opening hook is likely the first bottleneck.")
        changes.append("Add a visual change, motion spike, or sharper cut in the first 1-2 seconds.")
    if (scores.get("pacing_score") or 0) >= 65:
        strong.append("Pacing is active enough for short-form testing.")
    else:
        weak.append("The pacing may feel flat compared with stronger edits.")
        changes.append("Tighten long gaps and align key cuts with audio energy peaks.")
    if account.get("account_fit_score") is not None and account["account_fit_score"] < 50:
        weak.append("The edit is not fully aligned with recent account momentum.")
        changes.append("Compare this edit against your best recent uploads and match the strongest rhythm pattern.")
    return {
        "source": "fallback",
        "summary": "Forecast generated from the numeric model and available account context.",
        "final_recommendation": prediction.get("final_recommendation", "Refine the hook and pacing before publishing."),
        "what_is_strong": strong[:4],
        "what_is_weak": weak[:4],
        "what_should_change": changes[:4],
        "account_context_note": account.get("account_reason", "No account context connected."),
    }
