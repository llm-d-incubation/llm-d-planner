"""Unit tests for IntentExtractor._clean_llm_output() use_case normalization."""

import pytest

from planner.intent_extraction.extractor import IntentExtractor
from planner.shared.schemas import DeploymentIntent


@pytest.fixture
def extractor():
    """IntentExtractor with no LLM client (only testing _clean_llm_output)."""
    return IntentExtractor(llm_client=None)


def _base_intent(**overrides) -> dict:
    """Build a minimal valid raw LLM output dict with overrides."""
    data = {
        "use_case": "chatbot_conversational",
        "experience_class": "conversational",
        "user_count": 500,
        "accuracy_priority": "medium",
        "cost_priority": "medium",
        "latency_priority": "medium",
    }
    data.update(overrides)
    return data


# --- Alias mapping tests ---


@pytest.mark.unit
@pytest.mark.parametrize(
    "hallucinated, expected",
    [
        ("text_summarization", "summarization_short"),
        ("summarization", "summarization_short"),
        ("document_summarization", "long_document_summarization"),
        ("chatbot", "chatbot_conversational"),
        ("chat", "chatbot_conversational"),
        ("code_gen", "code_generation_detailed"),
        ("code_generation", "code_generation_detailed"),
        ("rag", "document_analysis_rag"),
        ("document_qa", "document_analysis_rag"),
        ("legal_analysis", "research_legal_analysis"),
        ("research_analysis", "research_legal_analysis"),
        ("research", "research_legal_analysis"),
        ("content", "content_generation"),
    ],
)
def test_clean_llm_output_maps_alias_to_valid_use_case(extractor, hallucinated, expected):
    """Known hallucinated use_case values are mapped to valid ones."""
    raw = _base_intent(use_case=hallucinated)
    cleaned = extractor._clean_llm_output(raw)
    assert cleaned["use_case"] == expected


# --- Valid values pass through unchanged ---


@pytest.mark.unit
@pytest.mark.parametrize(
    "valid_use_case",
    [
        "chatbot_conversational",
        "code_completion",
        "code_generation_detailed",
        "translation",
        "content_generation",
        "summarization_short",
        "document_analysis_rag",
        "long_document_summarization",
        "research_legal_analysis",
    ],
)
def test_clean_llm_output_preserves_valid_use_case(extractor, valid_use_case):
    """Already-valid use_case values pass through unchanged."""
    raw = _base_intent(use_case=valid_use_case)
    cleaned = extractor._clean_llm_output(raw)
    assert cleaned["use_case"] == valid_use_case


# --- Fuzzy matching tests ---


@pytest.mark.unit
@pytest.mark.parametrize(
    "typo, expected",
    [
        ("summarization_shorts", "summarization_short"),
        ("chatbot_conversatonal", "chatbot_conversational"),
        ("code_compltion", "code_completion"),
    ],
)
def test_clean_llm_output_fuzzy_matches_close_typos(extractor, typo, expected):
    """Typos close to valid values are fuzzy-matched."""
    raw = _base_intent(use_case=typo)
    cleaned = extractor._clean_llm_output(raw)
    assert cleaned["use_case"] == expected


# --- Case-insensitive normalization ---


@pytest.mark.unit
@pytest.mark.parametrize(
    "mixed_case, expected",
    [
        ("Text_Summarization", "summarization_short"),
        ("CHATBOT", "chatbot_conversational"),
        ("Code_Completion", "code_completion"),
        ("DOCUMENT_ANALYSIS_RAG", "document_analysis_rag"),
    ],
)
def test_clean_llm_output_handles_case_insensitive(extractor, mixed_case, expected):
    """Mixed-case use_case values are lowercased before alias/fuzzy matching."""
    raw = _base_intent(use_case=mixed_case)
    cleaned = extractor._clean_llm_output(raw)
    assert cleaned["use_case"] == expected


# --- Garbage values are NOT matched ---


@pytest.mark.unit
@pytest.mark.parametrize("garbage", ["banana", "xyz_123", "do_something"])
def test_clean_llm_output_does_not_match_garbage(extractor, garbage):
    """Completely unrelated values are left unchanged (Pydantic will reject them)."""
    raw = _base_intent(use_case=garbage)
    cleaned = extractor._clean_llm_output(raw)
    assert cleaned["use_case"] == garbage


# --- *_mentioned gating of priorities ---


@pytest.mark.unit
@pytest.mark.parametrize("prefix", ["accuracy", "cost", "latency"])
def test_mentioned_true_preserves_priority(extractor, prefix):
    """When *_mentioned is true, the LLM's priority is kept."""
    raw = _base_intent(
        **{
            f"{prefix}_priority": "high",
            f"{prefix}_mentioned": True,
        }
    )
    cleaned = extractor._clean_llm_output(raw)
    assert cleaned[f"{prefix}_priority"] == "high"


@pytest.mark.unit
@pytest.mark.parametrize("prefix", ["accuracy", "cost", "latency"])
def test_mentioned_false_resets_non_medium_priority(extractor, prefix):
    """When *_mentioned is false, a non-medium priority is forced to medium."""
    raw = _base_intent(
        **{
            f"{prefix}_priority": "high",
            f"{prefix}_mentioned": False,
        }
    )
    cleaned = extractor._clean_llm_output(raw)
    assert cleaned[f"{prefix}_priority"] == "medium"


@pytest.mark.unit
@pytest.mark.parametrize("prefix", ["accuracy", "cost", "latency"])
def test_mentioned_false_keeps_medium_priority(extractor, prefix):
    """When *_mentioned is false but priority is already medium, no change needed."""
    raw = _base_intent(
        **{
            f"{prefix}_priority": "medium",
            f"{prefix}_mentioned": False,
        }
    )
    cleaned = extractor._clean_llm_output(raw)
    assert cleaned[f"{prefix}_priority"] == "medium"


@pytest.mark.unit
@pytest.mark.parametrize("prefix", ["accuracy", "cost", "latency"])
def test_mentioned_missing_defaults_to_trust(extractor, prefix):
    """When *_mentioned is absent, the priority is trusted (default True)."""
    raw = _base_intent(**{f"{prefix}_priority": "high"})
    assert f"{prefix}_mentioned" not in raw
    cleaned = extractor._clean_llm_output(raw)
    assert cleaned[f"{prefix}_priority"] == "high"


@pytest.mark.unit
@pytest.mark.parametrize("prefix", ["accuracy", "cost", "latency"])
def test_mentioned_string_true_preserves_priority(extractor, prefix):
    """String 'true' is parsed correctly as truthy."""
    raw = _base_intent(
        **{
            f"{prefix}_priority": "high",
            f"{prefix}_mentioned": "true",
        }
    )
    cleaned = extractor._clean_llm_output(raw)
    assert cleaned[f"{prefix}_priority"] == "high"


@pytest.mark.unit
@pytest.mark.parametrize("prefix", ["accuracy", "cost", "latency"])
def test_mentioned_string_false_resets_priority(extractor, prefix):
    """String 'false' is parsed correctly as falsy."""
    raw = _base_intent(
        **{
            f"{prefix}_priority": "high",
            f"{prefix}_mentioned": "false",
        }
    )
    cleaned = extractor._clean_llm_output(raw)
    assert cleaned[f"{prefix}_priority"] == "medium"


@pytest.mark.unit
@pytest.mark.parametrize("prefix", ["accuracy", "cost", "latency"])
def test_mentioned_string_False_case_insensitive(extractor, prefix):
    """String 'False' (capitalized) is treated as falsy."""
    raw = _base_intent(
        **{
            f"{prefix}_priority": "low",
            f"{prefix}_mentioned": "False",
        }
    )
    cleaned = extractor._clean_llm_output(raw)
    assert cleaned[f"{prefix}_priority"] == "medium"


@pytest.mark.unit
def test_mentioned_fields_not_in_cleaned_output(extractor):
    """*_mentioned fields are consumed and not passed through to the final dict."""
    raw = _base_intent(
        accuracy_mentioned=True,
        cost_mentioned=False,
        latency_mentioned=True,
    )
    cleaned = extractor._clean_llm_output(raw)
    for prefix in ("accuracy", "cost", "latency"):
        assert f"{prefix}_mentioned" not in cleaned


@pytest.mark.unit
def test_mentioned_false_resets_multiple_priorities(extractor):
    """All three priorities can be independently gated by their *_mentioned flags."""
    raw = _base_intent(
        accuracy_priority="high",
        accuracy_mentioned=True,
        cost_priority="low",
        cost_mentioned=False,
        latency_priority="high",
        latency_mentioned=False,
    )
    cleaned = extractor._clean_llm_output(raw)
    assert cleaned["accuracy_priority"] == "high"
    assert cleaned["cost_priority"] == "medium"
    assert cleaned["latency_priority"] == "medium"


# --- user_count=0 guard ---


@pytest.mark.unit
@pytest.mark.parametrize("bad_count", [0, -1, -100])
def test_user_count_zero_or_negative_defaults_to_1000(extractor, bad_count):
    """user_count <= 0 (LLM schema default echo) is replaced with 100."""
    raw = _base_intent(user_count=bad_count)
    cleaned = extractor._clean_llm_output(raw)
    assert cleaned["user_count"] == 100


@pytest.mark.unit
def test_user_count_positive_preserved(extractor):
    """A valid positive user_count is kept as-is."""
    raw = _base_intent(user_count=10)
    cleaned = extractor._clean_llm_output(raw)
    assert cleaned["user_count"] == 10


# --- Integration: full parse with mocked hallucinated value ---


@pytest.mark.unit
def test_parse_extracted_intent_succeeds_with_hallucinated_use_case(extractor):
    """A hallucinated use_case is normalized and parses into a valid DeploymentIntent."""
    raw = _base_intent(use_case="text_summarization")
    intent = extractor._parse_extracted_intent(raw)
    assert isinstance(intent, DeploymentIntent)
    assert intent.use_case == "summarization_short"


@pytest.mark.unit
def test_parse_extracted_intent_fails_with_garbage_use_case(extractor):
    """A garbage use_case that can't be normalized still raises ValueError."""
    raw = _base_intent(use_case="banana")
    with pytest.raises(ValueError, match="Invalid intent data"):
        extractor._parse_extracted_intent(raw)
