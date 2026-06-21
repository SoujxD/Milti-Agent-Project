"""Unit tests for the analyst evaluation metrics.

These cover the pure scoring functions in ``evaluation/metrics.py``. They use
small hand-built analyst responses so each metric's behavior is checked against
a known expected value, with no model calls or network access required.
"""

from __future__ import annotations

import pytest

from evaluation.metrics import (
    avg_response_length,
    business_specificity_score,
    completeness_score,
    groundedness_score,
    insight_count,
    json_validity_score,
    keyword_score,
    recommendation_score,
    retrieval_usefulness_score,
    unique_insight_ratio,
)


@pytest.fixture
def sample_response() -> dict:
    """A representative well-formed analyst response."""
    return {
        "summary": "New visitors drive most sessions but convert poorly.",
        "key_insights": ["New visitors dominate traffic", "Conversion is low for new visitors"],
        "patterns": ["Bounce rate rises on weekends"],
        "recommendations": ["Target new visitors with onboarding", "Reduce bounce on weekends"],
        "confidence": 0.8,
    }


@pytest.fixture
def empty_response() -> dict:
    """An empty response to exercise the zero-floor branches."""
    return {
        "summary": "",
        "key_insights": [],
        "patterns": [],
        "recommendations": [],
        "confidence": 0.0,
    }


# ─────────────────────────────────────────
# keyword_score
# ─────────────────────────────────────────
def test_keyword_score_full_match(sample_response):
    # both keywords appear in the joined text
    score = keyword_score(sample_response, ["visitors", "bounce"])
    assert score == 1.0


def test_keyword_score_partial_match(sample_response):
    # one of two keywords present -> 0.5
    score = keyword_score(sample_response, ["visitors", "nonexistentterm"])
    assert score == 0.5


def test_keyword_score_empty_keywords_returns_zero(sample_response):
    assert keyword_score(sample_response, []) == 0.0


# ─────────────────────────────────────────
# recommendation_score
# ─────────────────────────────────────────
def test_recommendation_score_actionable(sample_response):
    # 2 recommendations, contains action terms ("target", "reduce")
    # coverage = min(2/3, 1.0) = 0.6667, actionability = 1.0
    score = recommendation_score(sample_response)
    assert score == pytest.approx(0.6667, abs=1e-3)


def test_recommendation_score_no_recommendations(empty_response):
    assert recommendation_score(empty_response) == 0.0


def test_recommendation_score_non_actionable():
    # recommendations present but no action verbs -> actionability 0.5
    resp = {"recommendations": ["The data shows trends", "Numbers were collected"]}
    score = recommendation_score(resp)
    # coverage = min(2/3,1)=0.6667, actionability=0.5 -> 0.3333
    assert score == pytest.approx(0.3333, abs=1e-3)


# ─────────────────────────────────────────
# completeness_score
# ─────────────────────────────────────────
def test_completeness_full(sample_response):
    # summary, key_insights, patterns, recommendations populated -> 4/5.
    # NOTE: confidence is a float (0.8); completeness only credits non-empty
    # str/list fields, so a float confidence is not counted. This documents the
    # function's actual behavior rather than asserting an idealized 1.0.
    assert completeness_score(sample_response) == pytest.approx(0.8, abs=1e-6)


def test_completeness_empty(empty_response):
    # confidence 0.0 is not a non-empty str/list, summary empty, lists empty -> 0
    assert completeness_score(empty_response) == 0.0


def test_completeness_partial():
    resp = {
        "summary": "Some summary",
        "key_insights": ["one"],
        "patterns": [],
        "recommendations": [],
        "confidence": "high",
    }
    # summary + key_insights + confidence populated = 3/5
    assert completeness_score(resp) == pytest.approx(0.6, abs=1e-6)


# ─────────────────────────────────────────
# groundedness_score
# ─────────────────────────────────────────
def test_groundedness_no_context(sample_response):
    assert groundedness_score(sample_response, "") == 0.0


def test_groundedness_with_overlap(sample_response):
    # context contains many of the answer tokens -> score > 0
    context = "new visitors dominate traffic and conversion and bounce on weekends"
    score = groundedness_score(sample_response, context)
    assert 0.0 < score <= 1.0


# ─────────────────────────────────────────
# insight_count / unique_insight_ratio
# ─────────────────────────────────────────
def test_insight_count(sample_response):
    # 2 insights + 1 pattern + 2 recommendations = 5
    assert insight_count(sample_response) == 5


def test_unique_insight_ratio_all_unique(sample_response):
    assert unique_insight_ratio(sample_response) == 1.0


def test_unique_insight_ratio_with_duplicates():
    resp = {
        "key_insights": ["New visitors dominate", "New visitors dominate"],
        "patterns": [],
        "recommendations": [],
    }
    # 2 items, 1 unique after normalization -> 0.5
    assert unique_insight_ratio(resp) == 0.5


def test_unique_insight_ratio_empty(empty_response):
    assert unique_insight_ratio(empty_response) == 0.0


# ─────────────────────────────────────────
# avg_response_length
# ─────────────────────────────────────────
def test_avg_response_length_empty(empty_response):
    assert avg_response_length(empty_response) == 0.0


def test_avg_response_length_positive(sample_response):
    assert avg_response_length(sample_response) > 0.0


# ─────────────────────────────────────────
# business_specificity_score
# ─────────────────────────────────────────
def test_business_specificity_rewards_terms(sample_response):
    # contains visitor, conversion, bounce, traffic, session... -> capped toward 1
    score = business_specificity_score(sample_response)
    assert 0.0 < score <= 1.0


def test_business_specificity_generic_text():
    resp = {"summary": "things happened and stuff occurred", "key_insights": [], "patterns": [], "recommendations": []}
    assert business_specificity_score(resp) == 0.0


# ─────────────────────────────────────────
# json_validity_score
# ─────────────────────────────────────────
def test_json_validity_valid_schema(sample_response):
    import json
    raw = json.dumps(sample_response)
    # valid json (1.0) + full schema (1.0) -> 1.0
    assert json_validity_score(raw, sample_response) == 1.0


def test_json_validity_malformed_raw(sample_response):
    # raw is not valid json; falls back to populated-schema * 0.5
    score = json_validity_score("not json at all", sample_response)
    # all 5 required keys populated -> 1.0 * 0.5 = 0.5
    assert score == pytest.approx(0.5, abs=1e-6)


# ─────────────────────────────────────────
# retrieval_usefulness_score
# ─────────────────────────────────────────
def test_retrieval_usefulness_no_context(sample_response):
    assert retrieval_usefulness_score(sample_response, "") == 0.0


def test_retrieval_usefulness_with_context(sample_response):
    context = "new_visitor sessions show bounce and conversion patterns across traffic sources"
    score = retrieval_usefulness_score(sample_response, context)
    assert 0.0 <= score <= 1.0
