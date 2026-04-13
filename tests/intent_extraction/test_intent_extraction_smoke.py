"""Smoke tests for intent extraction.

These run as part of ``make test-integration`` (and therefore ``make test``)
to catch basic regressions without running the full scenario suite.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_smoke_chatbot_extraction(intent_extractor):
    """Smoke test: basic chatbot intent extraction works."""
    intent = intent_extractor.extract_intent("I need a chatbot for 500 users")

    assert intent.use_case == "chatbot_conversational"
    assert intent.experience_class == "conversational"
    assert intent.user_count > 0


@pytest.mark.integration
def test_smoke_code_completion_extraction(intent_extractor):
    """Smoke test: basic code completion intent extraction works."""
    intent = intent_extractor.extract_intent("Fast code completion for 100 developers")

    assert intent.use_case == "code_completion"
    assert intent.experience_class == "instant"
    assert intent.user_count > 0
