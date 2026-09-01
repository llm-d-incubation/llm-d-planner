"""Tests for range-percentile SLO defaults and traffic profile generation."""

from typing import Literal

import pytest

from planner.knowledge_base.use_cases import _format_display_name
from planner.shared.schemas import DeploymentIntent, SLOTargets
from planner.specification.traffic_profile import TrafficProfileGenerator


def _make_intent(
    latency_priority: Literal["low", "medium", "high"] = "medium",
) -> DeploymentIntent:
    return DeploymentIntent(
        use_case="chatbot_conversational",
        user_count=1000,
        latency_priority=latency_priority,
    )


@pytest.mark.unit
class TestSLORangePercentile:
    def test_medium_uses_50th_percentile(self):
        gen = TrafficProfileGenerator()
        slo = gen.generate_slo_targets(_make_intent("medium"))
        # chatbot_conversational TTFT range: 100-500
        # 50th percentile: 100 + (500-100)*0.5 = 300, rounded to nearest 5 = 300
        assert slo.ttft_target_ms == 300

    def test_high_uses_25th_percentile(self):
        gen = TrafficProfileGenerator()
        slo = gen.generate_slo_targets(_make_intent("high"))
        # 25th percentile: 100 + (500-100)*0.25 = 200
        assert slo.ttft_target_ms == 200

    def test_low_uses_75th_percentile(self):
        gen = TrafficProfileGenerator()
        slo = gen.generate_slo_targets(_make_intent("low"))
        # 75th percentile: 100 + (500-100)*0.75 = 400
        assert slo.ttft_target_ms == 400

    def test_ranges_populated(self):
        gen = TrafficProfileGenerator()
        slo = gen.generate_slo_targets(_make_intent("medium"))
        assert slo.ttft_range is not None
        assert slo.ttft_range.min == 100
        assert slo.ttft_range.max == 500
        assert slo.itl_range is not None
        assert slo.e2e_range is not None

    def test_all_metrics_use_same_percentile(self):
        gen = TrafficProfileGenerator()
        slo = gen.generate_slo_targets(_make_intent("high"))
        # ITL range for chatbot: 15-50
        # 25th: 15 + (50-15)*0.25 = 23.75 -> round to 25
        assert slo.itl_target_ms == 25
        # E2E range for chatbot: 3940-13300
        # 25th: 3940 + (13300-3940)*0.25 = 6280
        assert slo.e2e_target_ms == 6280

    def test_code_completion_high_priority(self):
        intent = DeploymentIntent(
            use_case="code_completion", user_count=500, latency_priority="high"
        )
        gen = TrafficProfileGenerator()
        slo = gen.generate_slo_targets(intent)
        # TTFT range: 50-200, 25th: 50 + 150*0.25 = 87.5 -> 90
        assert slo.ttft_target_ms == 90

    def test_unknown_use_case_returns_none(self):
        from planner.knowledge_base.use_cases import UseCaseRepository

        repo = UseCaseRepository()
        assert repo.get_use_case("nonexistent_use_case") is None


@pytest.mark.unit
class TestTrafficProfileGeneration:
    def test_chatbot_traffic_profile(self):
        gen = TrafficProfileGenerator()
        intent = _make_intent("medium")
        profile = gen.generate_profile(intent)
        assert profile.prompt_tokens == 512
        assert profile.output_tokens == 256

    def test_qps_calculation(self):
        gen = TrafficProfileGenerator()
        intent = DeploymentIntent(use_case="chatbot_conversational", user_count=1000)
        profile = gen.generate_profile(intent)
        # 1000 * 0.2 active * 0.4 req/min / 60 * 2.0 peak = 2.67
        assert profile.expected_qps == 2.67

    def test_code_generation_traffic_profile(self):
        gen = TrafficProfileGenerator()
        intent = DeploymentIntent(use_case="code_generation_detailed", user_count=100)
        profile = gen.generate_profile(intent)
        assert profile.prompt_tokens == 1024
        assert profile.output_tokens == 1024

    def test_small_user_count_gets_minimum_qps(self):
        gen = TrafficProfileGenerator()
        intent = DeploymentIntent(use_case="chatbot_conversational", user_count=1)
        profile = gen.generate_profile(intent)
        assert profile.expected_qps == 0.1


@pytest.mark.unit
class TestFormatDisplayName:
    def test_simple_case(self):
        assert _format_display_name("chatbot_conversational") == "Chatbot Conversational"

    def test_rag_acronym(self):
        assert _format_display_name("document_analysis_rag") == "Document Analysis RAG"

    def test_ai_acronym(self):
        assert _format_display_name("conversational_ai") == "Conversational AI"

    def test_llm_acronym(self):
        assert _format_display_name("llm_deployment") == "LLM Deployment"
