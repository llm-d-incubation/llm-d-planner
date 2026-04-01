"""Unit tests for ConfigFinder cluster GPU intersection logic."""

from unittest.mock import MagicMock, patch

import pytest

from planner.recommendation.config_finder import ConfigFinder
from planner.shared.schemas import DeploymentIntent, SLOTargets, TrafficProfile


def _make_intent(
    preferred_gpu_types: list[str] | None = None,
) -> DeploymentIntent:
    """Create a minimal DeploymentIntent for testing."""
    return DeploymentIntent(
        use_case="chatbot_conversational",
        experience_class="conversational",
        user_count=100,
        preferred_gpu_types=preferred_gpu_types or [],
        additional_context=None,
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


@pytest.mark.unit
class TestClusterGPUIntersection:
    """Test intersection logic between cluster_gpu_types and preferred_gpu_types."""

    def setup_method(self):
        self.mock_repo = MagicMock()
        self.mock_repo.find_configurations_meeting_slo.return_value = []
        self.mock_catalog = MagicMock()
        self.mock_catalog.get_all_models.return_value = []
        self.finder = ConfigFinder(benchmark_repo=self.mock_repo, catalog=self.mock_catalog)

    def _call(self, cluster_gpu_types=None, preferred_gpu_types=None):
        """Helper to call plan_all_capacities and return gpu_types from the first repo call."""
        intent = _make_intent(preferred_gpu_types=preferred_gpu_types or [])
        self.finder.plan_all_capacities(
            traffic_profile=_make_traffic(),
            slo_targets=_make_slo(),
            intent=intent,
            cluster_gpu_types=cluster_gpu_types,
        )
        # Extract gpu_types from the FIRST call (fallback may add a second call)
        calls = self.mock_repo.find_configurations_meeting_slo.call_args_list
        first_call = calls[0]
        return first_call.kwargs.get("gpu_types") or first_call[1].get("gpu_types")

    def test_cluster_gpus_only_no_user_preference(self):
        """Cluster has GPUs, user has no preference -> use cluster GPUs."""
        gpu_types = self._call(cluster_gpu_types=["A100-80", "H100"])
        assert sorted(gpu_types) == ["A100-80", "H100"]

    def test_cluster_gpus_intersect_with_user_preference(self):
        """Cluster has A100-80+H100, user wants H100 -> intersection is H100."""
        gpu_types = self._call(
            cluster_gpu_types=["A100-80", "H100"],
            preferred_gpu_types=["H100"],
        )
        assert gpu_types == ["H100"]

    def test_empty_intersection_returns_no_configs(self):
        """Cluster has L4, user wants H100 -> empty intersection, early return."""
        intent = _make_intent(preferred_gpu_types=["H100"])
        result = self.finder.plan_all_capacities(
            traffic_profile=_make_traffic(),
            slo_targets=_make_slo(),
            intent=intent,
            cluster_gpu_types=["L4"],
        )
        assert result == []
        # Should NOT have called the benchmark repo at all
        self.mock_repo.find_configurations_meeting_slo.assert_not_called()

    def test_none_cluster_gpus_uses_user_preference(self):
        """No cluster detection (None) -> user preference only."""
        gpu_types = self._call(
            cluster_gpu_types=None,
            preferred_gpu_types=["H100"],
        )
        # Should normalize user preference and pass it
        assert "H100" in gpu_types

    def test_none_cluster_no_user_preference_no_filter(self):
        """No cluster detection, no user preference -> no filter (None)."""
        gpu_types = self._call(cluster_gpu_types=None)
        assert gpu_types is None

    def test_empty_list_cluster_gpus_uses_user_preference(self):
        """Empty list from detector (failed detection) -> user preference only."""
        gpu_types = self._call(
            cluster_gpu_types=[],
            preferred_gpu_types=["H100"],
        )
        assert "H100" in gpu_types

    def test_empty_list_cluster_no_user_preference_no_filter(self):
        """Empty list from detector, no user preference -> no filter."""
        gpu_types = self._call(cluster_gpu_types=[])
        assert gpu_types is None

    def test_cluster_gpu_no_benchmarks_falls_back_to_all(self):
        """Cluster GPU has no benchmark data -> retry without GPU filter."""
        # First call (with GPU filter) returns empty, second call (no filter) returns empty too
        # but we verify it retried without the filter.
        self.mock_repo.find_configurations_meeting_slo.side_effect = [[], []]
        intent = _make_intent()
        self.finder.plan_all_capacities(
            traffic_profile=_make_traffic(),
            slo_targets=_make_slo(),
            intent=intent,
            cluster_gpu_types=["B200"],
        )
        calls = self.mock_repo.find_configurations_meeting_slo.call_args_list
        assert len(calls) == 2
        # First call: filtered by cluster GPU
        assert calls[0].kwargs.get("gpu_types") == ["B200"]
        # Second call: no GPU filter (fallback)
        assert calls[1].kwargs.get("gpu_types") is None

    def test_cluster_gpu_no_benchmarks_no_fallback_when_user_pref(self):
        """Cluster+user GPU intersection has no benchmarks -> no fallback (user chose)."""
        self.mock_repo.find_configurations_meeting_slo.return_value = []
        intent = _make_intent(preferred_gpu_types=["H100"])
        result = self.finder.plan_all_capacities(
            traffic_profile=_make_traffic(),
            slo_targets=_make_slo(),
            intent=intent,
            cluster_gpu_types=["H100", "L4"],
        )
        assert result == []
        # Should only call once (no fallback since user expressed a GPU preference)
        assert self.mock_repo.find_configurations_meeting_slo.call_count == 1
