"""Full intent extraction test suite.

Parametrized from tests/fixtures/intent_extraction_scenarios.json.
Run explicitly with: make test-intent
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from planner.shared.schemas import ConversationMessage

from .conftest import assert_intent_matches

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _load_scenarios():
    """Load test scenarios from JSON fixture file."""
    with open(FIXTURES_DIR / "intent_extraction_scenarios.json") as f:
        return json.load(f)["scenarios"]


@pytest.mark.intent_extraction
@pytest.mark.parametrize("scenario", _load_scenarios(), ids=lambda s: s["id"])
def test_intent_extraction(intent_extractor, scenario):
    """Test that intent extraction produces expected results for each scenario."""
    # Build conversation history if provided
    history = None
    if scenario.get("conversation_history"):
        history = [ConversationMessage(**msg) for msg in scenario["conversation_history"]]

    intent = intent_extractor.extract_intent(
        user_message=scenario["user_message"],
        conversation_history=history,
    )

    assert_intent_matches(intent, scenario["expected"])
