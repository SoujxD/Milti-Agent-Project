"""Data analyst RAG agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from utils.dataset_adapter import load_analysis_dataset
from utils.llm_client import LLMClient
from utils.parser import extract_json
from utils.retriever import EcommerceRetriever, RetrievalResult


PROMPT_STYLES = {
    "basic": (
        "You are a senior business analyst. Answer the question using the provided dataset context when available. "
        "Be concise, practical, and honest about uncertainty."
    ),
    "structured_json": (
        "You are a rigorous analytics agent. Return valid JSON only and organize the answer into summary, key insights, "
        "patterns, recommendations, and confidence."
    ),
    "executive": (
        "You are preparing a short executive briefing for business stakeholders. Focus on decision-ready findings, "
        "commercial implications, and next steps."
    ),
    "evidence_constrained": (
        "Use only the supplied dataset evidence. If support is weak, say so clearly. Avoid claims that cannot be tied "
        "to the retrieved context."
    ),
}


DEFAULT_MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-7b-instruct",
    "google/gemma-2-9b-it",
    "qwen/qwen-2.5-7b-instruct",
]

# Prompt style used by the LangChain code path when no explicit style is given.
DEFAULT_LC_PROMPT_STYLE = "executive"

# Maps the legacy categorical confidence to the new 0-1 float scale.
_CONFIDENCE_TO_FLOAT = {"low": 0.3, "medium": 0.6, "high": 0.9}


def build_analyst_prompt(question: str, prompt_style: str, context: str, rag_enabled: bool = True) -> str:
    """Construct the analyst prompt shared by the legacy and LangChain paths.

    Kept identical to the original ``AnalystRAGAgent._build_prompt`` template so
    the deterministic mock generator in :mod:`utils.llm_client` continues to
    parse the prompt markers (``Prompt style:`` / ``Question:`` / etc.).
    """
    instruction = PROMPT_STYLES[prompt_style]
    context_block = context if rag_enabled and context else "RAG disabled. Respond from general statistical reasoning only."
    return f"""
{instruction}

Prompt style:
{prompt_style}

Question:
{question}

Dataset context:
{context_block}

Return JSON in exactly this schema:
{{
  "summary": "short paragraph",
  "key_insights": ["insight 1", "insight 2"],
  "patterns": ["pattern 1", "pattern 2"],
  "recommendations": ["recommendation 1", "recommendation 2"],
  "confidence": "high"
}}
""".strip()


class AnalystAnswer(BaseModel):
    """Structured analyst answer used by the LangChain code path."""

    answer: str = Field(description="Direct, decision-ready answer to the question.")
    key_findings: list[str] = Field(
        default_factory=list,
        description="3-5 concise, evidence-backed findings.",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Actionable recommendations for stakeholders.",
    )
    metrics_cited: list[str] = Field(
        default_factory=list,
        description="Specific metrics or values referenced in the answer.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the answer on a 0-1 scale.",
    )


@dataclass(slots=True)
class AnalystAgentResult:
    """Single analyst response payload."""

    question: str
    model: str
    prompt_style: str
    rag_enabled: bool
    raw_response: str
    parsed_response: dict[str, Any]
    retrieved_context: str
    retrieval_results: list[RetrievalResult]


class AnalystRAGAgent:
    """RAG-enabled business analyst for the e-commerce dataset."""

    def __init__(
        self,
        dataset_path: str | Path,
        llm_client: LLMClient | None = None,
        models: list[str] | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        adapted = load_analysis_dataset(self.dataset_path)
        self.dataframe = adapted.dataframe
        self.retriever = EcommerceRetriever(self.dataframe)
        self.llm_client = llm_client or LLMClient()
        self.models = models or DEFAULT_MODELS

    def available_models(self) -> list[str]:
        return self.models

    def available_prompt_styles(self) -> list[str]:
        return list(PROMPT_STYLES)

    def _build_prompt(self, question: str, prompt_style: str, context: str, rag_enabled: bool) -> str:
        return build_analyst_prompt(question, prompt_style, context, rag_enabled)

    def answer_question(
        self,
        question: str,
        model: str,
        prompt_style: str = "structured_json",
        rag_enabled: bool = True,
        top_k: int = 5,
    ) -> AnalystAgentResult:
        """Generate an analyst answer and include retrieval traces."""
        if prompt_style not in PROMPT_STYLES:
            raise ValueError(f"Unsupported prompt style: {prompt_style}")

        context = ""
        retrieval_results: list[RetrievalResult] = []
        if rag_enabled:
            context, retrieval_results = self.retriever.build_context(question, top_k=top_k)

        prompt = self._build_prompt(question, prompt_style, context, rag_enabled)
        llm_response = self.llm_client.generate(prompt=prompt, model=model)
        parsed = extract_json(llm_response.text)

        return AnalystAgentResult(
            question=question,
            model=model,
            prompt_style=prompt_style,
            rag_enabled=rag_enabled,
            raw_response=llm_response.text,
            parsed_response=parsed,
            retrieved_context=context,
            retrieval_results=retrieval_results,
        )


def _coerce_to_analyst_answer(parsed: dict[str, Any]) -> AnalystAnswer:
    """Coerce the legacy mock/LLM JSON schema into the AnalystAnswer model."""
    confidence: Any = parsed.get("confidence", "low")
    if isinstance(confidence, str):
        confidence = _CONFIDENCE_TO_FLOAT.get(confidence.strip().lower(), 0.5)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    key_findings = [str(item).strip() for item in parsed.get("key_insights", []) if str(item).strip()]
    recommendations = [str(item).strip() for item in parsed.get("recommendations", []) if str(item).strip()]

    return AnalystAnswer.model_validate(
        {
            "answer": str(parsed.get("summary", "")).strip(),
            "key_findings": key_findings[:5],
            "recommendations": recommendations,
            "metrics_cited": [],
            "confidence": confidence,
        }
    )


def _mock_analyst_answer(prompt: str) -> AnalystAnswer:
    """Build an AnalystAnswer from the project's deterministic mock generator."""
    response = LLMClient().generate(prompt=prompt, model=DEFAULT_MODELS[0])
    return _coerce_to_analyst_answer(extract_json(response.text))


def _legacy_context_text(page_content: str) -> str:
    """Convert a retriever Document into the ``key: value, ...`` form.

    The Chroma retriever emits ``key=value | key=value`` page content, while the
    deterministic mock generator's context parser expects the legacy
    ``key: value, key: value`` shape. Normalizing here keeps the mock answer
    grounded in retrieved rows; it is harmless for a real LLM, which reads either
    form. Documents already in legacy form (TF-IDF fallback) pass through intact.
    """
    pairs: list[str] = []
    for part in page_content.split(" | "):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, _, value = part.partition("=")
            pairs.append(f"{key.strip()}: {value.strip()}")
        else:
            pairs.append(part)
    return ", ".join(pairs)


def run_analyst_lc(question: str, use_rag: bool = True) -> dict[str, Any]:
    """LangChain-native analyst entry point returning a structured dict.

    Prefers ``ChatOpenAI(...).with_structured_output(AnalystAnswer)``. When no
    ``OPENAI_API_KEY`` is set or the API call fails, it falls back to the
    project's deterministic mock generator so the path works offline / key-free.
    Returns ``AnalystAnswer`` fields plus the retrieved context strings.
    """
    docs: list[Any] = []
    if use_rag:
        from utils.lc_retriever import build_or_load_retriever

        retriever = build_or_load_retriever()
        docs = list(retriever.invoke(question))

    # Plain legacy-form lines (no index/pipe prefix) so the mock generator's
    # row parser can read them directly; a real LLM reads them equally well.
    context = "\n".join(_legacy_context_text(doc.page_content) for doc in docs)
    prompt = build_analyst_prompt(question, DEFAULT_LC_PROMPT_STYLE, context, rag_enabled=use_rag)

    answer: AnalystAnswer | None = None
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(AnalystAnswer)
            answer = llm.invoke(prompt)  # type: ignore[assignment]
        except Exception:  # pragma: no cover - network/key dependent
            answer = None

    if answer is None:
        answer = _mock_analyst_answer(prompt)

    return {
        **answer.model_dump(),
        "retrieved_contexts": [doc.page_content for doc in docs],
    }
