"""LLM client with OpenRouter support and a deterministic local fallback."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(slots=True)
class LLMResponse:
    """Container for raw LLM output plus call metadata."""

    text: str
    provider: str
    model: str
    metadata: dict[str, Any]


class LLMClient:
    """Small wrapper around OpenRouter with a safe mock fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = OPENROUTER_URL,
        default_temperature: float = 0.2,
        app_name: str = "ISE547 Multi-Agent Analytics",
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url
        self.default_temperature = default_temperature
        self.app_name = app_name

    def generate(self, prompt: str, model: str, response_format: dict[str, Any] | None = None) -> LLMResponse:
        """Generate model output using OpenRouter or a deterministic local fallback."""
        if self.api_key:
            try:
                return self._generate_openrouter(prompt, model, response_format=response_format)
            except Exception as exc:  # pragma: no cover - network-dependent
                fallback = self._generate_mock(prompt, model)
                fallback.metadata["error"] = str(exc)
                fallback.metadata["fallback_used"] = True
                return fallback
        return self._generate_mock(prompt, model)

    def _generate_openrouter(
        self, prompt: str, model: str, response_format: dict[str, Any] | None = None
    ) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://openrouter.ai",
            "X-Title": self.app_name,
        }
        payload: dict[str, Any] = {
            "model": model,
            "temperature": self.default_temperature,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise business analytics assistant. Return JSON when requested.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        if response_format:
            payload["response_format"] = response_format

        response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        return LLMResponse(text=text, provider="openrouter", model=model, metadata=data)

    def _generate_mock(self, prompt: str, model: str) -> LLMResponse:
        digest = hashlib.sha256(f"{model}|{prompt}".encode("utf-8")).hexdigest()
        confidence_levels = ["low", "medium", "high"]
        confidence = confidence_levels[int(digest[0], 16) % len(confidence_levels)]
        prompt_style = self._extract_between(prompt, "Prompt style:", "Question:").strip() or "structured_json"
        question = self._extract_between(prompt, "Question:", "Dataset context:").strip()
        context = self._extract_between(prompt, "Dataset context:", "Return JSON in exactly this schema:").strip()
        rows = self._parse_context_rows(context)
        response = self._build_question_aware_mock(question, rows, confidence, model, prompt_style)
        text = json.dumps(
            response,
            indent=2,
        )
        return LLMResponse(text=text, provider="mock", model=model, metadata={"digest": digest})

    def _extract_between(self, text: str, start: str, end: str) -> str:
        try:
            return text.split(start, 1)[1].split(end, 1)[0]
        except IndexError:
            return ""

    def _parse_context_rows(self, context: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line in context.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            if "|" in cleaned:
                cleaned = cleaned.split("|", 1)[1].strip()
            row: dict[str, Any] = {}
            for chunk in cleaned.split(", "):
                if ": " not in chunk:
                    continue
                key, value = chunk.split(": ", 1)
                row[key.strip()] = self._coerce_value(value.strip())
            if row:
                rows.append(row)
        return rows

    def _coerce_value(self, value: str) -> Any:
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            numeric = float(value)
            return int(numeric) if numeric.is_integer() else numeric
        except ValueError:
            return value

    def _find_column(self, rows: list[dict[str, Any]], keywords: list[str]) -> str | None:
        if not rows:
            return None
        columns = list(rows[0].keys())
        normalized = {column: re.sub(r"[^a-z0-9]", "", column.lower()) for column in columns}
        for keyword in keywords:
            target = re.sub(r"[^a-z0-9]", "", keyword.lower())
            for column in columns:
                if target in normalized[column]:
                    return column
        return None

    def _avg(self, rows: list[dict[str, Any]], column: str, predicate: callable | None = None) -> float | None:
        values = []
        for row in rows:
            if predicate and not predicate(row):
                continue
            value = row.get(column)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if not values:
            return None
        return sum(values) / len(values)

    def _mode(self, rows: list[dict[str, Any]], column: str, predicate: callable | None = None) -> str | None:
        counts: dict[str, int] = {}
        for row in rows:
            if predicate and not predicate(row):
                continue
            value = row.get(column)
            if value is None:
                continue
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda item: item[1])[0]

    def _model_profile(self, model: str) -> dict[str, Any]:
        model_lower = model.lower()
        if "llama" in model_lower:
            return {"detail": 3, "keyword_bias": 1.0, "risk": "balanced"}
        if "mistral" in model_lower:
            return {"detail": 2, "keyword_bias": 0.9, "risk": "concise"}
        if "gemma" in model_lower:
            return {"detail": 2, "keyword_bias": 0.8, "risk": "cautious"}
        if "qwen" in model_lower:
            return {"detail": 3, "keyword_bias": 1.1, "risk": "analytical"}
        return {"detail": 2, "keyword_bias": 0.9, "risk": "balanced"}

    def _prompt_profile(self, prompt_style: str) -> dict[str, Any]:
        profiles = {
            "basic": {"recommendations": 2, "patterns": 1, "strict_grounding": False, "tone": "simple"},
            "structured_json": {"recommendations": 3, "patterns": 2, "strict_grounding": False, "tone": "structured"},
            "executive": {"recommendations": 3, "patterns": 1, "strict_grounding": False, "tone": "executive"},
            "evidence_constrained": {"recommendations": 2, "patterns": 2, "strict_grounding": True, "tone": "evidence"},
        }
        return profiles.get(prompt_style, profiles["structured_json"])

    def _build_question_aware_mock(
        self,
        question: str,
        rows: list[dict[str, Any]],
        confidence: str,
        model: str,
        prompt_style: str,
    ) -> dict[str, Any]:
        q = question.lower()
        model_profile = self._model_profile(model)
        prompt_profile = self._prompt_profile(prompt_style)
        revenue_col = self._find_column(rows, ["revenue", "converted", "purchase", "order"])
        customer_col = self._find_column(rows, ["visitor", "customer", "segment", "browser", "device"])
        channel_col = self._find_column(rows, ["traffic", "channel", "source", "campaign"])
        time_col = self._find_column(rows, ["month", "date", "week", "season", "day"])
        bounce_col = self._find_column(rows, ["bounce", "exit", "friction"])
        value_col = self._find_column(rows, ["page value", "pagevalue", "duration", "product", "cart", "basket"])
        category_col = self._find_column(rows, ["category", "department"])
        brand_col = self._find_column(rows, ["brand"])

        positives = []
        negatives = []
        if revenue_col:
            for row in rows:
                value = row.get(revenue_col)
                if value in {True, 1, "1", "true", "True"}:
                    positives.append(row)
                else:
                    negatives.append(row)

        summary = "The retrieved dataset context highlights a few stronger-performing segments and behaviors relevant to the question."
        insights: list[str] = []
        patterns: list[str] = []
        recommendations: list[str] = []

        def maybe_add_keyword(term: str, sentence: str) -> str:
            return f"{sentence} This points to {term}." if model_profile["keyword_bias"] > 0.95 else sentence

        if any(term in q for term in ["segment", "visitor", "customer", "target"]):
            if customer_col:
                best_customer = self._mode(positives or rows, customer_col)
                insights.append(maybe_add_keyword("customer segmentation", f"{customer_col} {best_customer} appears most often in the strongest retrieved records."))
                patterns.append(f"{customer_col} is a key grouping variable for answering this question.")
                recommendations.append(f"Prioritize campaigns and journeys for the {best_customer} segment.")

        if any(term in q for term in ["channel", "traffic", "source", "acquisition"]):
            if channel_col:
                best_channel = self._mode(positives or rows, channel_col)
                insights.append(maybe_add_keyword("channel efficiency", f"{channel_col} {best_channel} is the most prominent high-performing acquisition group in the retrieved context."))
                recommendations.append(f"Allocate more testing budget toward {channel_col} {best_channel}.")

        if any(term in q for term in ["month", "time", "season", "week", "day"]):
            if time_col:
                best_time = self._mode(positives or rows, time_col)
                insights.append(maybe_add_keyword("seasonality", f"{time_col} {best_time} appears most frequently among the stronger retrieved rows."))
                patterns.append(f"Performance may vary meaningfully across {time_col.lower()} groupings.")

        if any(term in q for term in ["category", "categories", "product", "products", "brand", "brands", "merchandise"]):
            if category_col:
                best_category = self._mode(positives or rows, category_col)
                insights.append(maybe_add_keyword("category mix", f"{category_col} {best_category} appears most often in the stronger retrieved records."))
                recommendations.append(f"Prioritize assortment, merchandising, and promotion around {best_category}.")
            if brand_col:
                best_brand = self._mode(positives or rows, brand_col)
                insights.append(maybe_add_keyword("brand concentration", f"{brand_col} {best_brand} appears repeatedly in the higher-intent retrieved rows."))
                patterns.append(f"{brand_col} concentration may be shaping commercial performance in this slice of evidence.")
                recommendations.append(f"Test deeper placement and pricing support for {best_brand}.")

        if any(term in q for term in ["bounce", "exit", "friction", "drop"]):
            if bounce_col:
                pos_avg = self._avg(positives, bounce_col) if positives else None
                neg_avg = self._avg(negatives, bounce_col) if negatives else None
                if pos_avg is not None and neg_avg is not None:
                    insights.append(maybe_add_keyword("reduced abandonment", f"{bounce_col} is lower in stronger rows ({pos_avg:.3f}) than in weaker rows ({neg_avg:.3f})."))
                    patterns.append(f"Lower {bounce_col} aligns with better business outcomes in the retrieved evidence.")
                recommendations.append(f"Reduce {bounce_col} through better landing-page relevance and clearer product discovery.")

        if any(term in q for term in ["page", "value", "product", "duration", "engagement"]):
            if value_col:
                pos_avg = self._avg(positives, value_col) if positives else None
                neg_avg = self._avg(negatives, value_col) if negatives else None
                if pos_avg is not None and neg_avg is not None:
                    insights.append(maybe_add_keyword("higher engagement", f"{value_col} is higher in stronger rows ({pos_avg:.2f}) than in weaker rows ({neg_avg:.2f})."))
                    patterns.append(f"Higher {value_col} appears alongside stronger commercial intent.")
                recommendations.append(f"Use {value_col} as a signal for identifying high-intent traffic and pages.")

        if any(term in q for term in ["recommend", "improve", "increase", "optimize"]) or not recommendations:
            recommendations.append("Focus on the segments, channels, and behaviors that recur in the strongest retrieved records.")
            recommendations.append("Validate these patterns on the full dataset before scaling budget or UX changes.")

        if not insights:
            generic_fields = ", ".join(list(rows[0].keys())[:4]) if rows else "the available columns"
            insights.append(f"The retrieved evidence is most informative around {generic_fields}.")
            patterns.append("The answer should be interpreted as a directional read from the top retrieved rows rather than a full-dataset estimate.")

        summary_topics = []
        for label, column in [
            ("customer segments", customer_col),
            ("channels", channel_col),
            ("time periods", time_col),
            ("category and brand signals", category_col or brand_col),
            ("behavior signals", value_col or bounce_col),
        ]:
            if column and (label not in summary_topics):
                summary_topics.append(label)
        if summary_topics:
            connector = ", ".join(summary_topics[:3])
            if prompt_profile["tone"] == "executive":
                summary = f"Executive read: the clearest performance signals for this question come from {connector}."
            elif prompt_profile["tone"] == "evidence":
                summary = f"Based strictly on retrieved evidence, the strongest observable signals come from {connector}."
            else:
                summary = f"For this question, the strongest signals in the retrieved context come from {connector}."

        if prompt_profile["strict_grounding"] and not rows:
            recommendations = ["Insufficient retrieved evidence to support a confident recommendation."]
            confidence = "low"

        if model_profile["risk"] == "cautious" and recommendations:
            recommendations[-1] = "Validate the pattern with a wider dataset slice before scaling decisions."
        elif model_profile["risk"] == "analytical" and patterns:
            patterns.append("This pattern should be compared across additional segments before final prioritization.")
        elif model_profile["risk"] == "concise":
            summary = summary.replace("the strongest", "the clearest")

        return {
            "summary": summary,
            "key_insights": insights[: max(1, min(model_profile["detail"], 3))],
            "patterns": (patterns[: prompt_profile["patterns"]] or ["The retrieved records show a directional pattern but not enough evidence for a broader claim."]),
            "recommendations": recommendations[: prompt_profile["recommendations"]],
            "confidence": confidence if rows else "low",
        }
