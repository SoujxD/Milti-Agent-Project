"""Experiment runner for multi-model and multi-prompt evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from agents.analyst_agent import AnalystRAGAgent
from evaluation.metrics import (
    avg_response_length,
    business_specificity_score,
    completeness_score,
    groundedness_score,
    insight_count,
    json_validity_score,
    keyword_score,
    llm_judge_scores,
    recommendation_score,
    retrieval_usefulness_score,
    unique_insight_ratio,
)
from utils.llm_client import LLMClient


class EvaluationPipeline:
    """Runs full experiments and aggregates results."""

    def __init__(
        self,
        agent: AnalystRAGAgent,
        questions_path: str | Path,
        output_dir: str | Path,
    ) -> None:
        self.agent = agent
        self.questions_path = Path(questions_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.questions = json.loads(self.questions_path.read_text())

    def run(
        self,
        models: list[str] | None = None,
        prompt_styles: list[str] | None = None,
        rag_options: list[bool] | None = None,
        limit: int | None = None,
        enable_judge: bool = True,
        judge_model: str = "openai/gpt-4o-mini",
    ) -> pd.DataFrame:
        """Run a complete factorial evaluation."""
        models = models or self.agent.available_models()
        prompt_styles = prompt_styles or self.agent.available_prompt_styles()
        rag_options = rag_options or [True, False]
        selected_questions = self.questions[:limit] if limit else self.questions
        judge_client = LLMClient() if enable_judge else None

        records: list[dict[str, Any]] = []
        for model in models:
            for prompt_style in prompt_styles:
                for rag_enabled in rag_options:
                    for item in selected_questions:
                        result = self.agent.answer_question(
                            question=item["question"],
                            model=model,
                            prompt_style=prompt_style,
                            rag_enabled=rag_enabled,
                        )
                        parsed = result.parsed_response
                        record = {
                            "question_id": item["id"],
                            "question": item["question"],
                            "category": item["category"],
                            "model": model,
                            "prompt_style": prompt_style,
                            "rag_enabled": rag_enabled,
                            "summary": parsed.get("summary", ""),
                            "raw_response": result.raw_response,
                            "parsed_response": json.dumps(parsed),
                            "retrieved_context": result.retrieved_context,
                            "keyword_score": keyword_score(parsed, item.get("expected_keywords", [])),
                            "recommendation_score": recommendation_score(parsed),
                            "completeness_score": completeness_score(parsed),
                            "groundedness_score": groundedness_score(parsed, result.retrieved_context),
                            "insight_count": insight_count(parsed),
                            "unique_insight_ratio": unique_insight_ratio(parsed),
                            "avg_response_length": avg_response_length(parsed),
                            "business_specificity_score": business_specificity_score(parsed),
                            "json_validity_score": json_validity_score(result.raw_response, parsed),
                            "retrieval_usefulness_score": retrieval_usefulness_score(parsed, result.retrieved_context),
                        }
                        judge_scores = (
                            llm_judge_scores(
                                question=item["question"],
                                parsed_response=parsed,
                                context=result.retrieved_context,
                                judge_model=judge_model,
                                client=judge_client,
                            )
                            if enable_judge
                            else {
                                "judge_usefulness": 0.0,
                                "judge_clarity": 0.0,
                                "judge_correctness": 0.0,
                            }
                        )
                        record.update(judge_scores)
                        record["overall_score"] = round(
                            (
                                record["keyword_score"]
                                + record["recommendation_score"]
                                + record["completeness_score"]
                                + record["groundedness_score"]
                            )
                            / 4,
                            4,
                        )
                        record["judge_average"] = round(
                            (
                                record["judge_usefulness"]
                                + record["judge_clarity"]
                                + record["judge_correctness"]
                            )
                            / 3,
                            4,
                        )
                        record["enhanced_overall_score"] = round(
                            (
                                record["overall_score"]
                                + record["business_specificity_score"]
                                + record["json_validity_score"]
                                + record["retrieval_usefulness_score"]
                                + (record["judge_average"] / 5)
                            )
                            / 5,
                            4,
                        )
                        records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(self.output_dir / "results.csv", index=False)
        self.aggregate_results(df)
        return df

    def aggregate_results(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Create grouped summary tables and export them."""
        metric_columns = [
            "keyword_score",
            "recommendation_score",
            "completeness_score",
            "groundedness_score",
            "business_specificity_score",
            "json_validity_score",
            "retrieval_usefulness_score",
            "unique_insight_ratio",
            "avg_response_length",
            "insight_count",
            "judge_usefulness",
            "judge_clarity",
            "judge_correctness",
            "judge_average",
            "overall_score",
            "enhanced_overall_score",
        ]
        model_table = df.groupby("model")[metric_columns].mean().sort_values("enhanced_overall_score", ascending=False)
        prompt_table = df.groupby("prompt_style")[metric_columns].mean().sort_values("enhanced_overall_score", ascending=False)
        rag_table = df.groupby("rag_enabled")[metric_columns].mean().sort_values("enhanced_overall_score", ascending=False)
        category_table = df.groupby("category")[metric_columns].mean().sort_values("enhanced_overall_score", ascending=False)
        quality_table = df[metric_columns].mean().to_frame(name="average_value")

        tables = {
            "model_comparison": model_table,
            "prompt_comparison": prompt_table,
            "rag_comparison": rag_table,
            "category_comparison": category_table,
            "quality_metrics": quality_table,
        }
        for name, table in tables.items():
            table.to_csv(self.output_dir / f"{name}.csv")
        return tables

    def top_examples(self, df: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
        """Return the best-scoring rows for presentation generation."""
        top_df = df.sort_values("overall_score", ascending=False).head(top_n)
        return top_df.to_dict(orient="records")
