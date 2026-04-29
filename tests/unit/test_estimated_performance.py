"""Unit tests for estimated performance flow (roofline estimation)."""

from unittest.mock import MagicMock, patch

import pytest

from planner.knowledge_base.benchmarks import BenchmarkData
from planner.recommendation.config_finder import ConfigFinder
from planner.recommendation.estimator import (
    CATALOG_TO_ROOFLINE_GPU,
    convert_estimation_to_benchmark,
    generate_estimated_configs,
)
from planner.shared.schemas import DeploymentIntent, SLOTargets, TrafficProfile


def _make_intent(
    preferred_gpu_types: list[str] | None = None,
    preferred_models: list[str] | None = None,
) -> DeploymentIntent:
    """Create a minimal DeploymentIntent for testing."""
    return DeploymentIntent(
        use_case="chatbot_conversational",
        experience_class="conversational",
        user_count=100,
        preferred_gpu_types=preferred_gpu_types or [],
        preferred_models=preferred_models or [],
    )


def _make_traffic() -> TrafficProfile:
    return TrafficProfile(
        prompt_tokens=512,
        output_tokens=256,
        expected_qps=5.0,
    )


def _make_slo() -> SLOTargets:
    return SLOTargets(
        ttft_p95_target_ms=500,
        itl_p95_target_ms=50,
        e2e_p95_target_ms=15000,
    )


def _make_benchmark_data(
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    hardware: str = "H100",
    hardware_count: int = 1,
    source: str = "blis",
    confidence_level: str = "benchmarked",
) -> BenchmarkData:
    """Create a BenchmarkData with default values."""
    return BenchmarkData(
        {
            "model_hf_repo": model,
            "hardware": hardware,
            "hardware_count": hardware_count,
            "framework": "vllm",
            "framework_version": "0.6.2",
            "prompt_tokens": 512,
            "output_tokens": 256,
            "mean_input_tokens": 512,
            "mean_output_tokens": 256,
            "ttft_mean": 100,
            "ttft_p90": 120,
            "ttft_p95": 130,
            "ttft_p99": 150,
            "itl_mean": 10,
            "itl_p90": 12,
            "itl_p95": 14,
            "itl_p99": 18,
            "e2e_mean": 3000,
            "e2e_p90": 3500,
            "e2e_p95": 4000,
            "e2e_p99": 5000,
            "tps_mean": 50.0,
            "tps_p90": 45.0,
            "tps_p95": 42.0,
            "tps_p99": 38.0,
            "tokens_per_second": 50.0,
            "requests_per_second": 10.0,
            "estimated": False,
            "source": source,
            "confidence_level": confidence_level,
            "model_uri": None,
        }
    )


@pytest.mark.unit
class TestConvertEstimationToBenchmark:
    """Test convert_estimation_to_benchmark() function."""

    def test_basic_conversion(self):
        bench = convert_estimation_to_benchmark(
            model_id="meta-llama/Llama-3.3-70B-Instruct",
            gpu_type="H100",
            gpu_count=2,
            prompt_tokens=512,
            output_tokens=256,
            ttft_ms=150.0,
            itl_ms=20.0,
            e2e_latency_ms=5000.0,
            output_throughput_tps=80.0,
        )

        assert bench.model_hf_repo == "meta-llama/Llama-3.3-70B-Instruct"
        assert bench.hardware == "H100"
        assert bench.hardware_count == 2
        assert bench.estimated is True
        assert bench.source == "llm-optimizer"
        assert bench.confidence_level == "estimated"
        assert bench.framework == "vllm"
        assert bench.framework_version == "estimated"

    def test_same_value_all_percentiles(self):
        bench = convert_estimation_to_benchmark(
            model_id="test/model",
            gpu_type="A100-80",
            gpu_count=1,
            prompt_tokens=512,
            output_tokens=256,
            ttft_ms=200.0,
            itl_ms=25.0,
            e2e_latency_ms=6000.0,
            output_throughput_tps=100.0,
        )

        # Roofline produces single-point estimates — all percentiles identical
        assert bench.ttft_mean == bench.ttft_p90 == bench.ttft_p95 == bench.ttft_p99 == 200.0
        assert bench.itl_mean == bench.itl_p90 == bench.itl_p95 == bench.itl_p99 == 25.0
        assert bench.e2e_mean == bench.e2e_p90 == bench.e2e_p95 == bench.e2e_p99 == 6000.0
        assert bench.tps_mean == bench.tps_p90 == bench.tps_p95 == bench.tps_p99 == 100.0

    def test_requests_per_second_calculation(self):
        bench = convert_estimation_to_benchmark(
            model_id="test/model",
            gpu_type="H100",
            gpu_count=1,
            prompt_tokens=512,
            output_tokens=256,
            ttft_ms=100.0,
            itl_ms=10.0,
            e2e_latency_ms=3000.0,
            output_throughput_tps=512.0,
        )

        # RPS = output_throughput_tps / output_tokens = 512 / 256 = 2.0
        assert bench.requests_per_second == 2.0

    def test_zero_output_tokens(self):
        """RPS should be 0 when output_tokens is 0 (avoid division by zero)."""
        bench = convert_estimation_to_benchmark(
            model_id="test/model",
            gpu_type="H100",
            gpu_count=1,
            prompt_tokens=512,
            output_tokens=0,
            ttft_ms=100.0,
            itl_ms=10.0,
            e2e_latency_ms=3000.0,
            output_throughput_tps=50.0,
        )

        assert bench.requests_per_second == 0.0

    def test_traffic_profile_fields(self):
        bench = convert_estimation_to_benchmark(
            model_id="test/model",
            gpu_type="L40",
            gpu_count=1,
            prompt_tokens=1024,
            output_tokens=512,
            ttft_ms=200.0,
            itl_ms=30.0,
            e2e_latency_ms=8000.0,
            output_throughput_tps=60.0,
        )

        assert bench.prompt_tokens == 1024
        assert bench.output_tokens == 512
        assert bench.mean_input_tokens == 1024
        assert bench.mean_output_tokens == 512

    def test_to_dict_round_trip(self):
        """Converted benchmark should produce a valid dict via to_dict()."""
        bench = convert_estimation_to_benchmark(
            model_id="test/model",
            gpu_type="H100",
            gpu_count=1,
            prompt_tokens=512,
            output_tokens=256,
            ttft_ms=100.0,
            itl_ms=10.0,
            e2e_latency_ms=3000.0,
            output_throughput_tps=50.0,
        )

        d = bench.to_dict()
        assert d["source"] == "llm-optimizer"
        assert d["confidence_level"] == "estimated"
        assert d["estimated"] is True
        assert d["model_hf_repo"] == "test/model"


@pytest.mark.unit
class TestCatalogToRooflineGPUMapping:
    """Test the CATALOG_TO_ROOFLINE_GPU mapping."""

    def test_h100_maps(self):
        assert CATALOG_TO_ROOFLINE_GPU["H100"] == "H100"

    def test_a100_80_maps_to_a100(self):
        assert CATALOG_TO_ROOFLINE_GPU["A100-80"] == "A100"

    def test_a100_40_maps(self):
        assert CATALOG_TO_ROOFLINE_GPU["A100-40"] == "A100-40GB"

    def test_unsupported_gpu_not_in_map(self):
        assert "L4" not in CATALOG_TO_ROOFLINE_GPU
        assert "A10G" not in CATALOG_TO_ROOFLINE_GPU
        assert "MI300X" not in CATALOG_TO_ROOFLINE_GPU


@pytest.mark.unit
class TestGenerateEstimatedConfigs:
    """Test generate_estimated_configs() orchestration."""

    def setup_method(self):
        self.mock_repo = MagicMock()
        self.mock_repo._get_connection.return_value = MagicMock()
        self.mock_catalog = MagicMock()

        # Set up GPU types in catalog
        gpu_h100 = MagicMock()
        gpu_h100.gpu_type = "H100"
        gpu_h100.memory_gb = 80
        gpu_a100 = MagicMock()
        gpu_a100.gpu_type = "A100-80"
        gpu_a100.memory_gb = 80
        self.mock_catalog.get_all_gpu_types.return_value = [gpu_h100, gpu_a100]
        self.mock_catalog.get_all_models.return_value = []

        self.finder = ConfigFinder(
            benchmark_repo=self.mock_repo,
            catalog=self.mock_catalog,
        )

    @patch("planner.recommendation.estimator.GPURecommender")
    @patch("planner.recommendation.estimator.get_model_config_from_hf")
    @patch("planner.recommendation.estimator.check_model_fits_gpu")
    def test_skips_covered_combinations(self, mock_fits, mock_config, mock_recommender_cls):
        """Models already in existing_benchmarks at all valid TPs should be skipped."""
        mock_config.return_value = MagicMock()
        mock_fits.return_value = [1, 2]  # Model fits at TP=1 and TP=2

        # Existing benchmarks cover both TP values
        existing = [
            _make_benchmark_data(
                model="meta-llama/Llama-3.1-8B-Instruct", hardware="H100", hardware_count=1
            ),
            _make_benchmark_data(
                model="meta-llama/Llama-3.1-8B-Instruct", hardware="H100", hardware_count=2
            ),
        ]

        results, warnings = generate_estimated_configs(
            traffic_profile=_make_traffic(),
            slo_targets=_make_slo(),
            preferred_models=["meta-llama/Llama-3.1-8B-Instruct"],
            existing_benchmarks=existing,
            gpu_types=["H100"],
            catalog=self.mock_catalog,
            benchmark_repo=self.mock_repo,
        )

        # Both TP values are covered — no estimation should run
        assert len(results) == 0
        mock_recommender_cls.assert_not_called()

    @patch("planner.recommendation.estimator.GPURecommender")
    @patch("planner.recommendation.estimator.get_model_config_from_hf")
    @patch("planner.recommendation.estimator.check_model_fits_gpu")
    def test_generates_for_uncovered_model(self, mock_fits, mock_config, mock_recommender_cls):
        """Uncovered model/GPU pairs should get roofline estimates."""
        mock_config.return_value = MagicMock()
        mock_fits.return_value = [2]  # Model fits at TP=2

        # Set up mock GPURecommender
        mock_result = MagicMock()
        mock_best_latency = MagicMock()
        mock_best_latency.ttft_ms = 200.0
        mock_best_latency.itl_ms = 25.0
        mock_best_latency.e2e_latency_s = 5.0
        mock_best_latency.output_throughput_tps = 80.0
        mock_result.best_configs = {"best_latency": mock_best_latency}

        mock_recommender = MagicMock()
        mock_recommender.get_gpu_results.return_value = (
            {"H100": mock_result},
            {},
        )
        mock_recommender_cls.return_value = mock_recommender

        results, warnings = generate_estimated_configs(
            traffic_profile=_make_traffic(),
            slo_targets=_make_slo(),
            preferred_models=["meta-llama/Llama-3.3-70B-Instruct"],
            existing_benchmarks=[],
            gpu_types=["H100"],
            catalog=self.mock_catalog,
            benchmark_repo=self.mock_repo,
        )

        assert len(results) == 1
        bench = results[0]
        assert bench.model_hf_repo == "meta-llama/Llama-3.3-70B-Instruct"
        assert bench.hardware == "H100"
        assert bench.hardware_count == 2
        assert bench.estimated is True
        assert bench.source == "llm-optimizer"
        assert bench.confidence_level == "estimated"
        assert bench.ttft_p95 == 200.0
        assert bench.e2e_p95 == 5000.0  # 5.0s * 1000

    @patch("planner.recommendation.estimator.GPURecommender")
    @patch("planner.recommendation.estimator.get_model_config_from_hf")
    @patch("planner.recommendation.estimator.check_model_fits_gpu")
    def test_generates_estimates_for_all_valid_tps(
        self, mock_fits, mock_config, mock_recommender_cls
    ):
        """Should generate one estimate per valid TP value."""
        mock_config.return_value = MagicMock()
        mock_fits.return_value = [1, 2, 4]  # Model fits at TP=1, 2, and 4

        # Set up mock GPURecommender to return results for each call
        mock_result = MagicMock()
        mock_best_latency = MagicMock()
        mock_best_latency.ttft_ms = 200.0
        mock_best_latency.itl_ms = 25.0
        mock_best_latency.e2e_latency_s = 5.0
        mock_best_latency.output_throughput_tps = 80.0
        mock_result.best_configs = {"best_latency": mock_best_latency}

        mock_recommender = MagicMock()
        mock_recommender.get_gpu_results.return_value = (
            {"H100": mock_result},
            {},
        )
        mock_recommender_cls.return_value = mock_recommender

        results, warnings = generate_estimated_configs(
            traffic_profile=_make_traffic(),
            slo_targets=_make_slo(),
            preferred_models=["meta-llama/Llama-3.1-8B-Instruct"],
            existing_benchmarks=[],
            gpu_types=["H100"],
            catalog=self.mock_catalog,
            benchmark_repo=self.mock_repo,
        )

        assert len(results) == 3
        tp_values = sorted(r.hardware_count for r in results)
        assert tp_values == [1, 2, 4]

        # GPURecommender should have been called once per TP value
        assert mock_recommender_cls.call_count == 3
        # Verify max_gpus was set to each TP value
        called_max_gpus = sorted(
            call.kwargs.get("max_gpus", call.args[3] if len(call.args) > 3 else None)
            for call in mock_recommender_cls.call_args_list
        )
        assert called_max_gpus == [1, 2, 4]

    @patch("planner.recommendation.estimator.GPURecommender")
    @patch("planner.recommendation.estimator.get_model_config_from_hf")
    @patch("planner.recommendation.estimator.check_model_fits_gpu")
    def test_skips_covered_tp_estimates_uncovered(
        self, mock_fits, mock_config, mock_recommender_cls
    ):
        """Should only estimate uncovered TP values when some are already in benchmarks."""
        mock_config.return_value = MagicMock()
        mock_fits.return_value = [1, 2]  # Model fits at TP=1 and TP=2

        # TP=1 already covered
        existing = [
            _make_benchmark_data(
                model="meta-llama/Llama-3.1-8B-Instruct", hardware="H100", hardware_count=1
            ),
        ]

        mock_result = MagicMock()
        mock_best_latency = MagicMock()
        mock_best_latency.ttft_ms = 150.0
        mock_best_latency.itl_ms = 20.0
        mock_best_latency.e2e_latency_s = 4.0
        mock_best_latency.output_throughput_tps = 100.0
        mock_result.best_configs = {"best_latency": mock_best_latency}

        mock_recommender = MagicMock()
        mock_recommender.get_gpu_results.return_value = (
            {"H100": mock_result},
            {},
        )
        mock_recommender_cls.return_value = mock_recommender

        results, warnings = generate_estimated_configs(
            traffic_profile=_make_traffic(),
            slo_targets=_make_slo(),
            preferred_models=["meta-llama/Llama-3.1-8B-Instruct"],
            existing_benchmarks=existing,
            gpu_types=["H100"],
            catalog=self.mock_catalog,
            benchmark_repo=self.mock_repo,
        )

        # Only TP=2 should be estimated (TP=1 is covered)
        assert len(results) == 1
        assert results[0].hardware_count == 2

    @patch("planner.recommendation.estimator.get_model_config_from_hf")
    @patch("planner.recommendation.estimator.check_model_fits_gpu")
    def test_model_fits_no_gpu_warning(self, mock_fits, mock_config):
        """Model that fits no GPUs should produce a warning."""
        mock_config.return_value = MagicMock()
        mock_fits.return_value = []  # Model doesn't fit

        results, warnings = generate_estimated_configs(
            traffic_profile=_make_traffic(),
            slo_targets=_make_slo(),
            preferred_models=["huge/model-that-doesnt-fit"],
            existing_benchmarks=[],
            gpu_types=["H100"],
            catalog=self.mock_catalog,
            benchmark_repo=self.mock_repo,
        )

        assert len(results) == 0
        assert any("does not fit" in w for w in warnings)

    @patch("planner.recommendation.estimator.get_model_config_from_hf")
    def test_hf_unreachable_warning(self, mock_config):
        """HuggingFace API failure should produce a warning."""
        mock_config.side_effect = Exception("Connection refused")

        results, warnings = generate_estimated_configs(
            traffic_profile=_make_traffic(),
            slo_targets=_make_slo(),
            preferred_models=["nonexistent/model"],
            existing_benchmarks=[],
            gpu_types=["H100"],
            catalog=self.mock_catalog,
            benchmark_repo=self.mock_repo,
        )

        assert len(results) == 0
        assert any("Could not estimate" in w for w in warnings)

    def test_no_models_returns_empty(self):
        """Empty preferred_models should return immediately."""
        results, warnings = generate_estimated_configs(
            traffic_profile=_make_traffic(),
            slo_targets=_make_slo(),
            preferred_models=[],
            existing_benchmarks=[],
            gpu_types=None,
            catalog=self.mock_catalog,
            benchmark_repo=self.mock_repo,
        )

        assert results == []
        assert warnings == []


@pytest.mark.unit
class TestPlanAllCapacitiesEstimated:
    """Test that plan_all_capacities passes estimated params correctly."""

    def setup_method(self):
        self.mock_repo = MagicMock()
        self.mock_repo.find_configurations_meeting_slo.return_value = []
        self.mock_catalog = MagicMock()
        self.mock_catalog.get_all_models.return_value = []
        self.finder = ConfigFinder(benchmark_repo=self.mock_repo, catalog=self.mock_catalog)

    @patch("planner.recommendation.config_finder.generate_estimated_configs")
    def test_calls_estimated_flow_when_enabled(self, mock_gen):
        """Should call generate_estimated_configs when enable_estimated=True and preferred_models set."""
        mock_gen.return_value = ([], [])

        self.finder.plan_all_capacities(
            traffic_profile=_make_traffic(),
            slo_targets=_make_slo(),
            intent=_make_intent(preferred_models=["test/model"]),
            preferred_models=["test/model"],
            enable_estimated=True,
        )

        mock_gen.assert_called_once()

    @patch("planner.recommendation.config_finder.generate_estimated_configs")
    def test_skips_estimated_flow_when_disabled(self, mock_gen):
        """Should not call generate_estimated_configs when enable_estimated=False."""
        mock_gen.return_value = ([], [])

        self.finder.plan_all_capacities(
            traffic_profile=_make_traffic(),
            slo_targets=_make_slo(),
            intent=_make_intent(preferred_models=["test/model"]),
            preferred_models=["test/model"],
            enable_estimated=False,
        )

        mock_gen.assert_not_called()

    @patch("planner.recommendation.config_finder.generate_estimated_configs")
    def test_skips_estimated_flow_without_models(self, mock_gen):
        """Should not call generate_estimated_configs when no preferred_models."""
        mock_gen.return_value = ([], [])

        self.finder.plan_all_capacities(
            traffic_profile=_make_traffic(),
            slo_targets=_make_slo(),
            intent=_make_intent(),
            enable_estimated=True,
        )

        mock_gen.assert_not_called()

    @patch("planner.recommendation.config_finder.generate_estimated_configs")
    def test_benchmark_metrics_include_confidence_level(self, mock_gen):
        """benchmark_metrics dict should include source and confidence_level."""
        # Return a benchmark from the DB query
        bench = _make_benchmark_data(source="blis", confidence_level="benchmarked")
        self.mock_repo.find_configurations_meeting_slo.return_value = [bench]

        mock_gpu_cost = MagicMock(return_value=4.0)
        self.mock_catalog.calculate_gpu_cost = mock_gpu_cost

        mock_gen.return_value = ([], [])

        results, _warnings = self.finder.plan_all_capacities(
            traffic_profile=_make_traffic(),
            slo_targets=_make_slo(),
            intent=_make_intent(),
            enable_estimated=True,
        )

        assert len(results) == 1
        metrics = results[0].benchmark_metrics
        assert metrics is not None
        assert metrics["source"] == "blis"
        assert metrics["confidence_level"] == "benchmarked"
        assert metrics["estimated"] is False
