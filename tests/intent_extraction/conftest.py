"""Fixtures and helpers for intent extraction tests."""

from __future__ import annotations

import logging

import pytest

from planner.intent_extraction.extractor import IntentExtractor
from planner.llm.ollama_client import OllamaClient
from planner.shared.schemas import DeploymentIntent

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def ollama_client():
    """Create OllamaClient, skipping all tests if Ollama is not available."""
    client = OllamaClient()
    if not client.is_available():
        pytest.skip("Ollama service is not available — skipping intent extraction tests")
    return client


@pytest.fixture(scope="session")
def intent_extractor(ollama_client):
    """Create IntentExtractor with a verified OllamaClient."""
    return IntentExtractor(llm_client=ollama_client)


def assert_intent_matches(intent: DeploymentIntent, expected: dict) -> None:
    """Validate a DeploymentIntent against flexible expected values.

    Supported matchers per field type:
      - str fields (use_case, experience_class):
          plain string              — exact match
          {"one_of": [...]}         — value must be in the list
      - user_count: {"min": N, "max": M} range check
      - list fields (domain_specialization, preferred_gpu_types, preferred_models):
          {"exact": [...]}           — sorted equality
          {"contains": [...]}        — all items must be present
          {"contains_any": [...]}    — at least one item must be present
          {"contains_any_partial": [...]} — at least one item partially matches
                                           (case-insensitive substring)
      - priority fields: {"one_of": [...]} — value must be in the list
    """
    # use_case: exact match or one_of
    if "use_case" in expected:
        exp = expected["use_case"]
        if isinstance(exp, dict) and "one_of" in exp:
            assert intent.use_case in exp["one_of"], (
                f"use_case: got '{intent.use_case}', expected one of {exp['one_of']}"
            )
        else:
            assert intent.use_case == exp, f"use_case: got '{intent.use_case}', expected '{exp}'"

    # experience_class: exact match or one_of
    if "experience_class" in expected:
        exp = expected["experience_class"]
        if isinstance(exp, dict) and "one_of" in exp:
            assert intent.experience_class in exp["one_of"], (
                f"experience_class: got '{intent.experience_class}', "
                f"expected one of {exp['one_of']}"
            )
        else:
            assert intent.experience_class == exp, (
                f"experience_class: got '{intent.experience_class}', expected '{exp}'"
            )

    # user_count: range check
    if "user_count" in expected:
        spec = expected["user_count"]
        assert spec["min"] <= intent.user_count <= spec["max"], (
            f"user_count: got {intent.user_count}, expected range [{spec['min']}, {spec['max']}]"
        )

    # List fields
    for field in ["domain_specialization", "preferred_gpu_types", "preferred_models"]:
        if field not in expected:
            continue
        spec = expected[field]
        actual: list[str] = getattr(intent, field)

        if "exact" in spec:
            assert sorted(actual) == sorted(spec["exact"]), (
                f"{field}: got {actual}, expected exactly {spec['exact']}"
            )

        if "contains" in spec:
            for item in spec["contains"]:
                assert item in actual, f"{field}: expected '{item}' in {actual}"

        if "contains_any" in spec:
            assert any(item in actual for item in spec["contains_any"]), (
                f"{field}: expected any of {spec['contains_any']} in {actual}"
            )

        if "contains_any_partial" in spec:
            # Case-insensitive substring match: at least one actual item
            # must contain one of the expected substrings.
            lower_actual = [a.lower() for a in actual]
            matched = any(
                substr.lower() in a for substr in spec["contains_any_partial"] for a in lower_actual
            )
            assert matched, (
                f"{field}: expected any of {spec['contains_any_partial']} "
                f"(case-insensitive partial) in {actual}"
            )

    # Priority fields
    for field in [
        "accuracy_priority",
        "cost_priority",
        "latency_priority",
    ]:
        if field not in expected:
            continue
        spec = expected[field]
        actual_val: str = getattr(intent, field)

        if "one_of" in spec:
            assert actual_val in spec["one_of"], (
                f"{field}: got '{actual_val}', expected one of {spec['one_of']}"
            )
