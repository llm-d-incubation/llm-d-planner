"""Traffic profile generation from deployment intent."""

import logging

from src.knowledge_base.slo_templates import SLOTemplateRepository
from src.shared.schemas import DeploymentIntent, SLOTargets, TrafficProfile

logger = logging.getLogger(__name__)


class TrafficProfileGenerator:
    """Generate traffic profiles and SLO targets from deployment intent."""

    def __init__(self, slo_repo: SLOTemplateRepository | None = None):
        """
        Initialize traffic profile generator.

        Args:
            slo_repo: SLO template repository (creates default if not provided)
        """
        self.slo_repo = slo_repo or SLOTemplateRepository()

    def generate_profile(self, intent: DeploymentIntent) -> TrafficProfile:
        """
        Generate traffic profile from deployment intent.

        Uses traffic profile from SLO templates aligned with GuideLLM configurations.

        Args:
            intent: Deployment intent

        Returns:
            TrafficProfile with exact GuideLLM traffic profile
        """
        # Get base template for use case
        template = self.slo_repo.get_template(intent.use_case)

        if not template:
            logger.warning(f"No template found for use_case={intent.use_case}, using defaults")
            return self._generate_default_profile(intent)

        # Calculate expected QPS based on user count and request frequency
        expected_qps = self._estimate_qps(
            user_count=intent.user_count,
            requests_per_user_per_day=10,  # Default assumption
            latency_priority=intent.latency_priority,
        )

        return TrafficProfile(
            prompt_tokens=template.prompt_tokens,
            output_tokens=template.output_tokens,
            expected_qps=expected_qps,
        )

    def generate_slo_targets(self, intent: DeploymentIntent) -> SLOTargets:
        """
        Generate SLO targets from deployment intent.

        Uses p95 SLO targets from templates, optionally adjusted for latency requirement.

        Args:
            intent: Deployment intent

        Returns:
            SLOTargets with p95 target latencies
        """
        # Get base template for use case
        template = self.slo_repo.get_template(intent.use_case)

        if not template:
            logger.warning(f"No template found for use_case={intent.use_case}, using defaults")
            return self._generate_default_slo(intent)

        # Adjust SLO targets based on latency priority
        ttft_target = self._adjust_slo_for_latency(
            template.ttft_p95_target_ms, intent.latency_priority
        )
        itl_target = self._adjust_slo_for_latency(
            template.itl_p95_target_ms, intent.latency_priority
        )
        e2e_target = self._adjust_slo_for_latency(
            template.e2e_p95_target_ms, intent.latency_priority
        )

        return SLOTargets(
            ttft_p95_target_ms=ttft_target,
            itl_p95_target_ms=itl_target,
            e2e_p95_target_ms=e2e_target,
        )

    def _estimate_qps(
        self, user_count: int, requests_per_user_per_day: int, latency_priority: str
    ) -> float:
        """
        Estimate peak QPS based on user count and usage patterns.

        Args:
            user_count: Number of users
            requests_per_user_per_day: Average requests per user per day
            latency_priority: Latency importance (low/medium/high)

        Returns:
            Estimated peak QPS
        """
        # Total requests per day
        total_requests_per_day = user_count * requests_per_user_per_day

        # Assume traffic is concentrated in peak hours (e.g., 8 hours)
        # with a peak-to-average ratio based on latency priority
        peak_hours = 8
        peak_ratio_map = {
            "high": 2.5,  # High latency priority = need more headroom
            "medium": 2.0,
            "low": 1.5,
        }
        peak_ratio = peak_ratio_map.get(latency_priority, 2.0)

        # Calculate average QPS during peak hours
        avg_qps_peak = total_requests_per_day / (peak_hours * 3600)

        # Apply peak ratio
        peak_qps = avg_qps_peak * peak_ratio

        # Ensure minimum QPS of 0.1 for small workloads
        peak_qps = max(0.1, peak_qps)

        return round(peak_qps, 2)

    def _adjust_slo_for_latency(self, base_target_ms: int, latency_priority: str) -> int:
        """
        Adjust SLO target based on user's latency priority.

        Args:
            base_target_ms: Base SLO target from template
            latency_priority: User's latency importance (low/medium/high)

        Returns:
            Adjusted SLO target
        """
        adjustment_factors = {
            "high": 0.9,  # High priority = 10% tighter targets
            "medium": 1.0,  # Use template as-is
            "low": 1.2,  # Low priority = 20% more relaxed
        }

        factor = adjustment_factors.get(latency_priority, 1.0)
        return int(base_target_ms * factor)

    def _generate_default_profile(self, intent: DeploymentIntent) -> TrafficProfile:
        """Generate default profile when no template available."""
        # Conservative defaults - use most common traffic profile (512→256)
        default_qps = intent.user_count * 0.01  # 1% of users active concurrently

        return TrafficProfile(
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=default_qps,
        )

    def _generate_default_slo(self, intent: DeploymentIntent) -> SLOTargets:
        """Generate default SLO targets when no template available."""
        # Using p95 targets for different latency priorities
        slo_map = {
            "high": (150, 25, 7000),  # High priority = tighter targets (ttft, itl, e2e)
            "medium": (300, 30, 25000),
            "low": (600, 40, 60000),
        }

        ttft, itl, e2e = slo_map.get(intent.latency_priority, (300, 30, 25000))

        return SLOTargets(ttft_p95_target_ms=ttft, itl_p95_target_ms=itl, e2e_p95_target_ms=e2e)
