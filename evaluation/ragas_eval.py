"""Ragas-based RAG evaluation for the LangChain analyst path.

Builds an evaluation dataset by running ``run_analyst_lc`` over the project's
question bank, then scores it with Ragas reference-free and reference-based
metrics. Ragas needs an LLM judge, so when ``OPENAI_API_KEY`` is absent the run
is skipped gracefully and a placeholder summary is written instead of crashing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from agents.analyst_agent import run_analyst_lc
from utils.lc_retriever import EMBED_MODEL


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_QUESTIONS_PATH = BASE_DIR / "data" / "evaluation_questions.json"


def _resolve_questions_path(questions_path: str | Path) -> Path:
    path = Path(questions_path)
    if not path.exists():
        path = BASE_DIR / questions_path
    return path


def build_ragas_dataset(
    questions_path: str | Path = "data/evaluation_questions.json",
    limit: int = 20,
) -> Dataset:
    """Run the analyst over the question bank and assemble a Ragas dataset."""
    path = _resolve_questions_path(questions_path)
    questions = json.loads(path.read_text())[:limit]

    rows: list[dict[str, Any]] = []
    for item in questions:
        result = run_analyst_lc(item["question"])
        rows.append(
            {
                "user_input": item["question"],
                "response": result["answer"],
                "retrieved_contexts": result["retrieved_contexts"],
                "reference": item.get("ground_truth", ""),
            }
        )
    return Dataset.from_list(rows)


def run_ragas_evaluation(limit: int = 20) -> dict[str, Any]:
    """Evaluate the analyst with Ragas, skipping gracefully without an LLM judge."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "ragas_summary.json"

    dataset = build_ragas_dataset(limit=limit)

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Ragas evaluation requires an LLM judge (set OPENAI_API_KEY). "
            "Skipping and writing a placeholder summary."
        )
        summary = {"status": "skipped", "reason": "no OPENAI_API_KEY"}
        summary_path.write_text(json.dumps(summary, indent=2))
        return summary

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI

    judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBED_MODEL))

    # NOTE: ragas 0.4.x requires metric *instances* (the spec listed the classes).
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithoutReference(),
        LLMContextRecall(),
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    results_df = result.to_pandas()
    results_df.to_csv(OUTPUT_DIR / "ragas_results.csv", index=False)

    summary: dict[str, Any] = {"status": "completed", "rows": int(len(results_df))}
    for metric in metrics:
        if metric.name in results_df.columns:
            summary[metric.name] = round(float(results_df[metric.name].mean()), 4)
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Ragas evaluation summary:")
    print(json.dumps(summary, indent=2))
    return summary
