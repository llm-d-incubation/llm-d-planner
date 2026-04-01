"""Test ConfigFinder uses injected quality scorer."""

from unittest.mock import MagicMock, patch

import pytest

from planner.knowledge_base.benchmarks import BenchmarkData
from planner.recommendation.config_finder import ConfigFinder
from planner.shared.schemas import DeploymentIntent, SLOTargets, TrafficProfile


def _make_bench(
    model: str = "RedHatAI/test-model",
    hardware: str = "H100",
    hw_count: int = 1,
) -> BenchmarkData:
    return BenchmarkData(
        {
            "model_hf_repo": model,
            "hardware": hardware,
            "hardware_count": hw_count,
            "framework": "vllm",
            "framework_version": "0.8.4",
            "prompt_tokens": 512,
            "output_tokens": 256,
            "mean_input_tokens": 512,
            "mean_output_tokens": 256,
            "ttft_mean": 50,
            "ttft_p90": 70,
            "ttft_p95": 80,
            "ttft_p99": 100,
            "itl_mean": 10,
            "itl_p90": 15,
            "itl_p95": 20,
            "itl_p99": 25,
            "e2e_mean": 3000,
            "e2e_p90": 4000,
            "e2e_p95": 5000,
            "e2e_p99": 6000,
            "tps_mean": 1000,
            "tps_p90": 900,
            "tps_p95": 800,
            "tps_p99": 700,
            "tokens_per_second": 1000,
            "requests_per_second": 7,
        }
    )


def _make_intent(use_case: str = "chatbot_conversational") -> DeploymentIntent:
    return DeploymentIntent(
        use_case=use_case,
        user_count=100,
        experience_class="conversational",
    )


def _make_traffic() -> TrafficProfile:
    return TrafficProfile(
        prompt_tokens=512,
        output_tokens=256,
        expected_qps=5.0,
    )


def _make_slo() -> SLOTargets:
    return SLOTargets(
        ttft_p95_target_ms=200,
        itl_p95_target_ms=50,
        e2e_p95_target_ms=7000,
    )


@pytest.mark.unit
def test_config_finder_accepts_quality_scorer():
    """ConfigFinder should accept an optional quality_scorer parameter."""
    mock_source = MagicMock()
    mock_catalog = MagicMock()
    mock_scorer = MagicMock()
    mock_scorer.get_quality_score.return_value = 75.0

    finder = ConfigFinder(
        benchmark_repo=mock_source,
        catalog=mock_catalog,
        quality_scorer=mock_scorer,
    )
    assert finder._quality_scorer is mock_scorer


@pytest.mark.unit
def test_config_finder_quality_scorer_defaults_to_none():
    """When no quality_scorer provided, _quality_scorer should be None."""
    mock_source = MagicMock()
    mock_catalog = MagicMock()

    finder = ConfigFinder(benchmark_repo=mock_source, catalog=mock_catalog)
    assert finder._quality_scorer is None


@pytest.mark.unit
def test_injected_scorer_used_in_plan_all_capacities():
    """When quality_scorer is injected, plan_all_capacities uses it instead of score_model_quality."""
    mock_source = MagicMock()
    bench = _make_bench()
    mock_source.find_configurations_meeting_slo.return_value = [bench]

    mock_catalog = MagicMock()
    mock_catalog.get_all_models.return_value = []
    mock_catalog.calculate_gpu_cost.return_value = 2.70

    mock_scorer = MagicMock()
    mock_scorer.get_quality_score.return_value = 82.5

    finder = ConfigFinder(
        benchmark_repo=mock_source,
        catalog=mock_catalog,
        quality_scorer=mock_scorer,
    )

    results = finder.plan_all_capacities(
        traffic_profile=_make_traffic(),
        slo_targets=_make_slo(),
        intent=_make_intent(),
    )

    # Verify the injected scorer was called
    mock_scorer.get_quality_score.assert_called()
    # Should have been called with the model name and use case
    call_args = mock_scorer.get_quality_score.call_args_list[0]
    assert call_args[0][0] == "RedHatAI/test-model"
    assert call_args[0][1] == "chatbot_conversational"

    # Verify we got results with accuracy score from the injected scorer
    assert len(results) >= 1
    assert results[0].scores.accuracy_score >= 82  # 82.5 truncated to int


@pytest.mark.unit
def test_default_scorer_used_when_none_injected():
    """When no quality_scorer injected, plan_all_capacities uses the default score_model_quality."""
    mock_source = MagicMock()
    bench = _make_bench()
    mock_source.find_configurations_meeting_slo.return_value = [bench]

    mock_catalog = MagicMock()
    mock_catalog.get_all_models.return_value = []
    mock_catalog.calculate_gpu_cost.return_value = 2.70

    finder = ConfigFinder(
        benchmark_repo=mock_source,
        catalog=mock_catalog,
        # No quality_scorer -- uses default
    )

    with patch(
        "planner.recommendation.quality.score_model_quality",
        return_value=60.0,
    ) as mock_default:
        results = finder.plan_all_capacities(
            traffic_profile=_make_traffic(),
            slo_targets=_make_slo(),
            intent=_make_intent(),
        )

        mock_default.assert_called()
        assert len(results) >= 1
        assert results[0].scores.accuracy_score >= 60


@pytest.mark.unit
def test_injected_scorer_fallback_on_zero():
    """When injected scorer returns 0 for model name, it retries with bench.model_hf_repo."""
    mock_source = MagicMock()
    bench = _make_bench(model="RedHatAI/some-model")
    mock_source.find_configurations_meeting_slo.return_value = [bench]

    # Add a model to the catalog so model.name differs from bench.model_hf_repo
    mock_model = MagicMock()
    mock_model.model_id = "redhatai/some-model"
    mock_model.name = "Some Model Display Name"
    mock_model.size_parameters = "8B"
    mock_catalog = MagicMock()
    mock_catalog.get_all_models.return_value = [mock_model]
    mock_catalog.calculate_gpu_cost.return_value = 2.70

    mock_scorer = MagicMock()
    # Return 0 for display name, then 70 for model_hf_repo
    mock_scorer.get_quality_score.side_effect = [0.0, 70.0]

    finder = ConfigFinder(
        benchmark_repo=mock_source,
        catalog=mock_catalog,
        quality_scorer=mock_scorer,
    )

    results = finder.plan_all_capacities(
        traffic_profile=_make_traffic(),
        slo_targets=_make_slo(),
        intent=_make_intent(),
    )

    # The scorer should have been called twice: once with display name, once with hf repo
    assert mock_scorer.get_quality_score.call_count == 2
    assert len(results) >= 1
    assert results[0].scores.accuracy_score >= 70
