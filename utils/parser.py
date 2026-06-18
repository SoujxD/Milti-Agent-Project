"""Helpers for parsing and normalizing LLM outputs."""

from __future__ import annotations

import json
import re
from typing import Any


DEFAULT_RESPONSE = {
    "summary": "",
    "key_insights": [],
    "patterns": [],
    "recommendations": [],
    "confidence": "low",
}


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = re.split(r"[\n;•-]+", value)
        return [part.strip() for part in parts if part.strip()]
    return [str(value).strip()]


def normalize_response(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize user- or model-generated JSON into the expected schema."""
    normalized = dict(DEFAULT_RESPONSE)
    normalized["summary"] = str(data.get("summary", "")).strip()
    normalized["key_insights"] = _coerce_list(data.get("key_insights"))
    normalized["patterns"] = _coerce_list(data.get("patterns"))
    normalized["recommendations"] = _coerce_list(data.get("recommendations"))
    confidence = str(data.get("confidence", "low")).strip().lower()
    normalized["confidence"] = confidence if confidence in {"high", "medium", "low"} else "low"
    return normalized


def extract_json(text: str) -> dict[str, Any]:
    """Attempt to parse JSON from raw LLM text, including fenced blocks."""
    stripped = (text or "").strip()
    if not stripped:
        return dict(DEFAULT_RESPONSE)

    candidates = [stripped]
    fenced_match = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    candidates.extend(fenced_match)

    brace_match = re.search(r"(\{.*\})", stripped, flags=re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(1))

    for candidate in candidates:
        try:
            return normalize_response(json.loads(candidate))
        except json.JSONDecodeError:
            continue

    return normalize_response({"summary": stripped, "confidence": "low"})
