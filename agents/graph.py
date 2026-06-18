"""LangGraph supervisor orchestrating the analyst and presenter agents.

A supervisor node routes between an ``analyst`` node (structured RAG answer) and
a ``presenter`` node (PowerPoint deck), looping back to the supervisor after each
until it decides to ``FINISH``. Routing uses an LLM when ``OPENAI_API_KEY`` is
present and a deterministic rule otherwise, so the graph runs fully offline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from agents.analyst_agent import run_analyst_lc


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

PRESENTATION_KEYWORDS = ["deck", "slides", "presentation", "powerpoint", "ppt"]


# ─────────────────────────────────────────
# State & routing schema
# ─────────────────────────────────────────
class AgentState(TypedDict):
    """Shared state passed between graph nodes."""

    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    analyst_output: dict | None
    presentation_path: str | None
    next_agent: str | None


class RouterDecision(BaseModel):
    """Structured supervisor routing decision."""

    next_agent: Literal["analyst", "presenter", "FINISH"] = Field(
        description="Which agent should act next, or FINISH when the work is done."
    )
    reasoning: str = Field(description="Short justification for the routing choice.")


# ─────────────────────────────────────────
# Supervisor
# ─────────────────────────────────────────
def _rule_based_route(
    question: str,
    analyst_output: dict | None,
    presentation_path: str | None,
) -> str:
    """Deterministic routing used when no LLM is available."""
    if analyst_output is None:
        return "analyst"
    wants_deck = any(keyword in question.lower() for keyword in PRESENTATION_KEYWORDS)
    if wants_deck and presentation_path is None:
        return "presenter"
    return "FINISH"


def _supervisor_prompt(state: AgentState) -> str:
    return f"""You are a supervisor coordinating two agents for a business analytics task.

Available agents:
- "analyst": answers the question with retrieval-augmented analysis.
- "presenter": builds a PowerPoint deck from the dataset.

Current status:
- Question: {state.get("question", "")}
- Analyst has produced output: {state.get("analyst_output") is not None}
- Presentation already generated: {state.get("presentation_path") is not None}

Routing rules:
- If the analyst has not produced output yet, choose "analyst".
- Choose "presenter" only if the user is asking for a deck/slides/presentation and it has not been generated yet.
- Otherwise choose "FINISH".

Decide the next step."""


def supervisor_node(state: AgentState) -> dict[str, Any]:
    """Decide which agent runs next (LLM-routed, with a deterministic fallback)."""
    question = state.get("question", "")
    analyst_output = state.get("analyst_output")
    presentation_path = state.get("presentation_path")

    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI

            router = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(RouterDecision)
            decision = router.invoke(_supervisor_prompt(state))
            return {"next_agent": decision.next_agent}
        except Exception:  # pragma: no cover - network/key dependent
            pass  # fall through to the deterministic rule

    return {"next_agent": _rule_based_route(question, analyst_output, presentation_path)}


# ─────────────────────────────────────────
# Worker nodes
# ─────────────────────────────────────────
def analyst_node(state: AgentState) -> dict[str, Any]:
    """Run the structured LangChain analyst over the question."""
    output = run_analyst_lc(state["question"])
    answer = output.get("answer", "")
    return {
        "analyst_output": output,
        "messages": [AIMessage(content=f"[analyst] {answer}")],
    }


def _generate_presentation(analyst_output: dict | None) -> Path:
    """Adapter from the graph state to the existing presentation generator.

    The existing ``PresentationGeneratorAgent`` is dataset-driven: it expects a
    ``dataset_path`` + ``output_path``, not the analyst's structured output. We
    therefore map the graph request onto a deck built from the default dataset.

    # TODO(review): the dataset-driven presenter does not yet consume
    # analyst_output (answer/key_findings/recommendations). Threading those
    # findings into the slide content would make the deck reflect the analysis.
    """
    from agents.presentation_agent import PresentationGeneratorAgent

    presenter = PresentationGeneratorAgent(output_dir=OUTPUT_DIR)
    output_path, _slides, _charts = presenter.create_presentation(
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_DIR / "presentation.pptx",
    )
    return output_path


def presenter_node(state: AgentState) -> dict[str, Any]:
    """Generate a PowerPoint deck and record its path."""
    path = _generate_presentation(state.get("analyst_output"))
    return {
        "presentation_path": str(path),
        "messages": [AIMessage(content=f"[presenter] generated deck at {path}")],
    }


# ─────────────────────────────────────────
# Wiring
# ─────────────────────────────────────────
def route_from_supervisor(state: AgentState) -> str:
    """Map the supervisor's decision onto a graph edge."""
    next_agent = state.get("next_agent")
    if next_agent == "analyst":
        return "analyst"
    if next_agent == "presenter":
        return "presenter"
    return END


def build_graph():
    """Build and compile the supervisor graph."""
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("presenter", presenter_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"analyst": "analyst", "presenter": "presenter", END: END},
    )
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("presenter", "supervisor")

    return graph.compile()
