"""CLI entrypoint for running demos, evaluation, and presentation generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from agents.analyst_agent import AnalystRAGAgent


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "dataset.csv"
QUESTIONS_PATH = BASE_DIR / "data" / "evaluation_questions.json"
OUTPUT_DIR = BASE_DIR / "outputs"


def build_agent() -> AnalystRAGAgent:
    return AnalystRAGAgent(dataset_path=DATASET_PATH)


def run_demo(question: str, model: str, prompt_style: str, rag_enabled: bool) -> None:
    agent = build_agent()
    result = agent.answer_question(question=question, model=model, prompt_style=prompt_style, rag_enabled=rag_enabled)
    print("Question:", question)
    print("Retrieved context:\n", result.retrieved_context or "[RAG disabled]")
    print("Parsed response:\n", result.parsed_response)


def run_evaluation(limit: int | None = None, enable_judge: bool = True, judge_model: str = "openai/gpt-4o-mini") -> pd.DataFrame:
    from evaluation.evaluator import EvaluationPipeline

    agent = build_agent()
    pipeline = EvaluationPipeline(agent=agent, questions_path=QUESTIONS_PATH, output_dir=OUTPUT_DIR)
    return pipeline.run(limit=limit, enable_judge=enable_judge, judge_model=judge_model)


def run_presentation() -> Path:
    from agents.presentation_agent import PresentationGeneratorAgent

    presenter = PresentationGeneratorAgent(output_dir=OUTPUT_DIR)
    output_path, _, _ = presenter.create_presentation(dataset_path=DATASET_PATH, output_path=OUTPUT_DIR / "presentation.pptx")
    return output_path


def run_graph(question: str) -> dict:
    from agents.graph import build_graph

    graph = build_graph()
    return graph.invoke({"question": question, "messages": []})


def run_ragas(limit: int) -> None:
    from evaluation.ragas_eval import run_ragas_evaluation

    run_ragas_evaluation(limit=limit)


def print_graph_state(state: dict) -> None:
    print("Question:", state.get("question"))
    print("Final route:", state.get("next_agent"))

    analyst_output = state.get("analyst_output")
    if analyst_output:
        print("\nAnalyst output:")
        print(json.dumps(analyst_output, indent=2))

    presentation_path = state.get("presentation_path")
    if presentation_path:
        print("\nPresentation saved to:", presentation_path)

    messages = state.get("messages", [])
    if messages:
        print("\nAgent trace:")
        for message in messages:
            print(" -", getattr(message, "content", str(message)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent business analytics system")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="Run a single analyst-agent demo")
    demo_parser.add_argument("--question", required=True)
    demo_parser.add_argument("--model", default="meta-llama/llama-3.1-8b-instruct")
    demo_parser.add_argument("--prompt-style", default="structured_json")
    demo_parser.add_argument("--no-rag", action="store_true")

    eval_parser = subparsers.add_parser("evaluate", help="Run the experiment pipeline")
    eval_parser.add_argument("--limit", type=int, default=None)
    eval_parser.add_argument("--no-judge", action="store_true", help="Skip usefulness/clarity/correctness judge ratings")
    eval_parser.add_argument("--judge-model", default="openai/gpt-4o-mini")

    subparsers.add_parser("presentation", help="Generate the PowerPoint deck")

    graph_parser = subparsers.add_parser("graph", help="Run the LangGraph supervisor over a question")
    graph_parser.add_argument("--question", required=True)

    ragas_parser = subparsers.add_parser("ragas", help="Run the Ragas RAG evaluation")
    ragas_parser.add_argument("--limit", type=int, default=20)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "demo":
        run_demo(args.question, args.model, args.prompt_style, not args.no_rag)
    elif args.command == "evaluate":
        df = run_evaluation(limit=args.limit, enable_judge=not args.no_judge, judge_model=args.judge_model)
        print(f"Saved {len(df)} evaluation rows to {OUTPUT_DIR / 'results.csv'}")
    elif args.command == "presentation":
        path = run_presentation()
        print(f"Saved presentation to {path}")
    elif args.command == "graph":
        state = run_graph(args.question)
        print_graph_state(state)
    elif args.command == "ragas":
        run_ragas(args.limit)


if __name__ == "__main__":
    main()
