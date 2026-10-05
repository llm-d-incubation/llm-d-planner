"""Library configuration controls HuggingFace authentication during estimation."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from planner import Planner
from planner.capacity_planner import get_model_config_from_hf, get_model_info_from_hf
from planner.shared.schemas import DeploymentIntent

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("configured_token", "environment_token", "expected_token"),
    [
        ("hf_configured", "hf_environment", "hf_configured"),
        (None, "hf_environment", "hf_environment"),
        (None, None, None),
    ],
)
def test_planner_uses_configured_hf_token_for_estimation(
    configured_token, environment_token, expected_token, monkeypatch, mock_scoring_engine
):
    if environment_token is None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HF_TOKEN", environment_token)
    get_model_config_from_hf.cache_clear()
    get_model_info_from_hf.cache_clear()

    with (
        patch("planner.planner.build_scoring_engine", side_effect=mock_scoring_engine),
        patch("planner.orchestration.workflow.detect_cluster_gpus", return_value=None),
        patch("planner.capacity_planner.HfApi") as hf_api,
        patch(
            "planner.capacity_planner.AutoConfig.from_pretrained",
            return_value=SimpleNamespace(),
        ) as auto_config,
        patch("planner.recommendation.estimator.check_model_fits_gpu", return_value=[1]) as fits,
        patch("planner.gpu_recommender._LLM_OPTIMIZER_AVAILABLE", True),
        patch(
            "planner.gpu_recommender.GPURecommender.get_gpu_results", return_value=({}, {})
        ) as estimate,
    ):
        planner = Planner(hf_token=configured_token)
        try:
            fixtures = Path(__file__).parent.parent / "fixtures"
            planner.load_benchmarks(fixtures / "test_benchmarks.json")
            spec = planner.generate_specification(
                DeploymentIntent(
                    use_case="chatbot_conversational",
                    user_count=100,
                    preferred_models=["test/private-model"],
                    preferred_gpu_types=["H100"],
                )
            )
            planner.generate_recommendations(spec, enable_estimated=True)

            auto_config.assert_called_once()
            assert auto_config.call_args.kwargs["token"] == expected_token
            hf_api.assert_called_once_with(token=expected_token)
            fits.assert_called_once()
            assert fits.call_args.kwargs["hf_token"] == expected_token
            estimate.assert_called_once()
        finally:
            planner._benchmark_repo.close()
            get_model_config_from_hf.cache_clear()
            get_model_info_from_hf.cache_clear()
