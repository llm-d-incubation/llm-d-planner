"""Test UseCaseQualityScorer catalog fallback."""

import pytest

from planner.recommendation.quality.usecase_scorer import UseCaseQualityScorer


@pytest.mark.unit
def test_set_catalog_fallback_provides_score():
    scorer = UseCaseQualityScorer()
    scorer.set_catalog_fallback({"catalog/only-model": 72.5})
    score = scorer.get_quality_score("catalog/only-model", "chatbot_conversational")
    assert score == pytest.approx(72.5)


@pytest.mark.unit
def test_csv_score_takes_precedence_over_fallback():
    """If a model has a CSV score, the fallback is not used."""
    scorer = UseCaseQualityScorer()
    # Pick a model that exists in the CSV data
    csv_score = scorer.get_quality_score("granite 3.3 8b (non-reasoning)", "chatbot_conversational")
    # Now set a fallback with a different score
    scorer.set_catalog_fallback({"granite 3.3 8b (non-reasoning)": 99.9})
    after = scorer.get_quality_score("granite 3.3 8b (non-reasoning)", "chatbot_conversational")
    assert after == csv_score  # CSV wins


@pytest.mark.unit
def test_fallback_returns_zero_for_unknown_model():
    scorer = UseCaseQualityScorer()
    scorer.set_catalog_fallback({"some/model": 50.0})
    score = scorer.get_quality_score("completely/unknown", "chatbot_conversational")
    assert score == 0.0
