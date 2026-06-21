"""Tests for the supervisor routing logic in ``agents/graph.py``.

The supervisor's decision logic is isolated in ``_rule_based_route`` (the
deterministic, offline path) and ``route_from_supervisor`` (state -> edge).
Testing these directly verifies the multi-agent orchestration without needing
an LLM or API key: analyst runs first, presenter runs only when a deck is
requested and not yet built, and the graph FINISHES once work is complete.
"""

from __future__ import annotations

from langgraph.graph import END

from agents.graph import _rule_based_route, route_from_supervisor


# ─────────────────────────────────────────
# _rule_based_route: the offline decision logic
# ─────────────────────────────────────────
def test_routes_to_analyst_when_no_output_yet():
    # No analyst output exists -> analyst must run first, regardless of question.
    decision = _rule_based_route("Make me a deck", analyst_output=None, presentation_path=None)
    assert decision == "analyst"


def test_routes_to_presenter_when_deck_requested_and_analysis_done():
    # Analysis done, question asks for a deck, no deck yet -> presenter.
    decision = _rule_based_route(
        "Build me some slides on top customers",
        analyst_output={"answer": "done"},
        presentation_path=None,
    )
    assert decision == "presenter"


def test_finishes_when_analysis_done_and_no_deck_requested():
    # Plain question, analysis done, no deck asked for -> FINISH.
    decision = _rule_based_route(
        "Which customers convert best?",
        analyst_output={"answer": "done"},
        presentation_path=None,
    )
    assert decision == "FINISH"


def test_finishes_when_deck_already_generated():
    # Deck requested but already built -> no repeat, FINISH.
    decision = _rule_based_route(
        "Make a presentation",
        analyst_output={"answer": "done"},
        presentation_path="outputs/presentation.pptx",
    )
    assert decision == "FINISH"


def test_deck_keyword_detection_is_case_insensitive():
    decision = _rule_based_route(
        "Generate a POWERPOINT please",
        analyst_output={"answer": "done"},
        presentation_path=None,
    )
    assert decision == "presenter"


def test_non_deck_question_with_analysis_does_not_route_to_presenter():
    # "report" is not a presentation keyword -> should FINISH, not present.
    decision = _rule_based_route(
        "Give me a report on revenue",
        analyst_output={"answer": "done"},
        presentation_path=None,
    )
    assert decision == "FINISH"


# ─────────────────────────────────────────
# route_from_supervisor: decision -> graph edge
# ─────────────────────────────────────────
def test_route_edge_to_analyst():
    assert route_from_supervisor({"next_agent": "analyst"}) == "analyst"


def test_route_edge_to_presenter():
    assert route_from_supervisor({"next_agent": "presenter"}) == "presenter"


def test_route_edge_finish_maps_to_end():
    assert route_from_supervisor({"next_agent": "FINISH"}) == END


def test_route_edge_unknown_maps_to_end():
    # Any unrecognized decision falls through to END (safe default).
    assert route_from_supervisor({"next_agent": None}) == END


# ─────────────────────────────────────────
# Full routing sequence (integration of the rule logic)
# ─────────────────────────────────────────
def test_full_deck_request_sequence():
    """Simulate the supervisor loop for a deck request: analyst -> presenter -> FINISH."""
    question = "Make me a deck about top customers"
    analyst_output = None
    presentation_path = None

    # Step 1: nothing done yet -> analyst
    step1 = _rule_based_route(question, analyst_output, presentation_path)
    assert step1 == "analyst"
    analyst_output = {"answer": "customers analyzed"}  # analyst ran

    # Step 2: analysis done, deck wanted, none built -> presenter
    step2 = _rule_based_route(question, analyst_output, presentation_path)
    assert step2 == "presenter"
    presentation_path = "outputs/presentation.pptx"  # presenter ran

    # Step 3: both done -> FINISH
    step3 = _rule_based_route(question, analyst_output, presentation_path)
    assert step3 == "FINISH"


def test_full_plain_question_sequence():
    """A plain question runs the analyst once, then FINISHES (no presenter)."""
    question = "Which traffic source converts best?"
    analyst_output = None

    step1 = _rule_based_route(question, analyst_output, None)
    assert step1 == "analyst"
    analyst_output = {"answer": "traffic analyzed"}

    step2 = _rule_based_route(question, analyst_output, None)
    assert step2 == "FINISH"
