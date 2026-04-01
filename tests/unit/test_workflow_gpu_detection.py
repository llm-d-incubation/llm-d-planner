"""Unit tests for GPU detection integration in RecommendationWorkflow."""

import contextlib
from unittest.mock import MagicMock, patch

import pytest


def _make_specs():
    """Create minimal specifications dict for testing."""
    return {
        "intent": {
            "use_case": "chatbot_conversational",
            "experience_class": "conversational",
            "user_count": 100,
            "preferred_gpu_types": [],
        },
        "traffic_profile": {
            "prompt_tokens": 512,
            "output_tokens": 256,
            "expected_qps": 5.0,
        },
        "slo_targets": {
            "ttft_p95_target_ms": 500,
            "itl_p95_target_ms": 50,
            "e2e_p95_target_ms": 15000,
        },
    }


@pytest.mark.unit
class TestWorkflowGPUDetection:
    """Verify all three workflow methods call detect_cluster_gpus.

    Uses module-level patching of ConfigFinder to avoid PostgreSQL
    connection attempts during instantiation.
    """

    @patch("planner.orchestration.workflow.detect_cluster_gpus", return_value=["H100"])
    @patch("planner.orchestration.workflow.ConfigFinder")
    def test_generate_recommendation_from_specs_calls_detector(
        self, mock_config_finder, mock_detect
    ):
        mock_finder = mock_config_finder.return_value
        mock_finder.plan_all_capacities.return_value = []

        from planner.orchestration.workflow import RecommendationWorkflow

        workflow = RecommendationWorkflow()
        # Will raise ValueError because no configs returned — that's fine
        with contextlib.suppress(ValueError):
            workflow.generate_recommendation_from_specs(_make_specs())

        mock_detect.assert_called_once()
        call_kwargs = mock_finder.plan_all_capacities.call_args
        assert call_kwargs.kwargs.get("cluster_gpu_types") == ["H100"]

    @patch("planner.orchestration.workflow.detect_cluster_gpus", return_value=["A100-80"])
    @patch("planner.orchestration.workflow.ConfigFinder")
    @patch("planner.orchestration.workflow.Analyzer")
    def test_generate_ranked_recommendations_calls_detector(
        self, mock_analyzer_cls, mock_config_finder, mock_detect
    ):
        """Tests generate_ranked_recommendations() — the method with its
        own inline plan_all_capacities() call (does NOT delegate)."""
        from planner.shared.schemas import (
            DeploymentIntent,
            DeploymentSpecification,
            SLOTargets,
            TrafficProfile,
        )

        mock_finder = mock_config_finder.return_value
        # Return a list with at least one config to avoid early return path
        mock_finder.plan_all_capacities.return_value = [MagicMock()]

        # Mock Analyzer to avoid complex schema validation
        mock_analyzer = mock_analyzer_cls.return_value
        mock_analyzer.generate_ranked_lists.return_value = {
            "best_accuracy": [],
            "lowest_cost": [],
            "lowest_latency": [],
            "simplest": [],
            "balanced": [],
        }
        mock_analyzer.get_unique_configs_count.return_value = 0

        from planner.orchestration.workflow import RecommendationWorkflow

        workflow = RecommendationWorkflow()
        # Mock generate_specification with proper schema objects
        intent = DeploymentIntent(
            use_case="chatbot_conversational",
            experience_class="conversational",
            user_count=100,
            preferred_gpu_types=[],
            additional_context=None,
        )
        traffic = TrafficProfile(prompt_tokens=512, output_tokens=256, expected_qps=5.0)
        slo = SLOTargets(ttft_p95_target_ms=500, itl_p95_target_ms=50, e2e_p95_target_ms=15000)
        spec = DeploymentSpecification(intent=intent, traffic_profile=traffic, slo_targets=slo)
        workflow.generate_specification = MagicMock(return_value=(spec, intent, traffic, slo))  # type: ignore[method-assign]
        workflow.generate_ranked_recommendations("test message")

        mock_detect.assert_called_once()
        call_kwargs = mock_finder.plan_all_capacities.call_args
        assert call_kwargs.kwargs.get("cluster_gpu_types") == ["A100-80"]

    @patch("planner.orchestration.workflow.detect_cluster_gpus", return_value=["A100-80"])
    @patch("planner.orchestration.workflow.ConfigFinder")
    def test_generate_ranked_from_spec_calls_detector(self, mock_config_finder, mock_detect):
        mock_finder = mock_config_finder.return_value
        mock_finder.plan_all_capacities.return_value = []

        from planner.orchestration.workflow import RecommendationWorkflow

        workflow = RecommendationWorkflow()
        workflow.generate_ranked_recommendations_from_spec(_make_specs())

        mock_detect.assert_called_once()
        call_kwargs = mock_finder.plan_all_capacities.call_args
        assert call_kwargs.kwargs.get("cluster_gpu_types") == ["A100-80"]

    @patch("planner.orchestration.workflow.detect_cluster_gpus", return_value=[])
    @patch("planner.orchestration.workflow.ConfigFinder")
    def test_empty_detection_passes_empty_list(self, mock_config_finder, mock_detect):
        mock_finder = mock_config_finder.return_value
        mock_finder.plan_all_capacities.return_value = []

        from planner.orchestration.workflow import RecommendationWorkflow

        workflow = RecommendationWorkflow()
        workflow.generate_ranked_recommendations_from_spec(_make_specs())

        call_kwargs = mock_finder.plan_all_capacities.call_args
        assert call_kwargs.kwargs.get("cluster_gpu_types") == []
