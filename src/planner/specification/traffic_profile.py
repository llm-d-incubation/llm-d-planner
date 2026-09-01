"""Traffic profile generation from deployment intent."""

import logging

from planner.knowledge_base.use_cases import UseCaseConfig, UseCaseRepository
from planner.shared.schemas import DeploymentIntent, SLOTargets, TrafficProfile

logger = logging.getLogger(__name__)

# Percentile values for each latency priority level
LATENCY_PRIORITY_PERCENTILE = {
    "high": 0.25,
    "medium": 0.50,
    "low": 0.75,
}


def _round_to_nearest(value: float, nearest: int = 5) -> int:
    """Round a value to the nearest multiple."""
    return int(round(value / nearest) * nearest)


class TrafficProfileGenerator:
    """Generate traffic profiles and SLO targets from deployment intent."""

    def __init__(
        self,
        use_case_repo: UseCaseRepository | None = None,
    ):
        self.use_case_repo = use_case_repo or UseCaseRepository()

    def generate_profile(self, intent: DeploymentIntent) -> TrafficProfile:
        """Generate traffic profile from deployment intent."""
        uc = self.use_case_repo.get_use_case(intent.use_case)
        if not uc:
            raise ValueError(f"Unknown use_case: {intent.use_case}")

        expected_qps = self._estimate_qps(uc, intent.user_count)

        return TrafficProfile(
            prompt_tokens=uc.prompt_tokens,
            output_tokens=uc.output_tokens,
            expected_qps=expected_qps,
        )

    def generate_slo_targets(self, intent: DeploymentIntent) -> SLOTargets:
        """Generate SLO targets using range-percentile defaults from use case config."""
        uc = self.use_case_repo.get_use_case(intent.use_case)
        if not uc:
            raise ValueError(f"Unknown use_case: {intent.use_case}")

        percentile = LATENCY_PRIORITY_PERCENTILE.get(intent.latency_priority, 0.5)

        ttft_target = _round_to_nearest(
            uc.ttft_range.min + (uc.ttft_range.max - uc.ttft_range.min) * percentile
        )
        itl_target = _round_to_nearest(
            uc.itl_range.min + (uc.itl_range.max - uc.itl_range.min) * percentile
        )
        e2e_target = _round_to_nearest(
            uc.e2e_range.min + (uc.e2e_range.max - uc.e2e_range.min) * percentile
        )

        return SLOTargets(
            ttft_target_ms=ttft_target,
            itl_target_ms=itl_target,
            e2e_target_ms=e2e_target,
            ttft_range=uc.ttft_range.model_copy(),
            itl_range=uc.itl_range.model_copy(),
            e2e_range=uc.e2e_range.model_copy(),
        )

    def _estimate_qps(self, uc: UseCaseConfig, user_count: int) -> float:
        """Estimate peak QPS based on user count and use case workload parameters."""
        expected_concurrent = int(user_count * uc.active_fraction)
        expected_rps = (expected_concurrent * uc.requests_per_active_user_per_min) / 60
        peak_rps = expected_rps * uc.peak_multiplier
        peak_rps = max(0.1, peak_rps)
        return float(round(peak_rps, 2))
