"""Specification generation service.

Single source of truth for assembling a DeploymentSpecification from
a DeploymentIntent. Used by the Planner facade, the REST route handler,
and the workflow.
"""

import json
import logging
from pathlib import Path

from planner.data._resolver import data_path
from planner.knowledge_base.use_cases import UseCaseRepository
from planner.recommendation.quality.scoring import load_quality_weights
from planner.shared.schemas import (
    DeploymentIntent,
    DeploymentSpecification,
    Priorities,
    PriorityEntry,
    QualityWeights,
    WorkloadProfile,
)
from planner.specification.traffic_profile import TrafficProfileGenerator

logger = logging.getLogger(__name__)


class SpecificationService:
    """Generate complete deployment specifications from intent."""

    def __init__(
        self, data_dir: Path | None = None, traffic_gen: TrafficProfileGenerator | None = None
    ):
        self._data_dir = data_dir

        if traffic_gen is None:
            use_case_repo = UseCaseRepository(
                data_path=data_path("configuration/usecase_slo_workload.json", data_dir),
            )
            traffic_gen = TrafficProfileGenerator(use_case_repo=use_case_repo)
        self._traffic_gen = traffic_gen

        # Cache config data at init — these are static files
        self._quality_weights_by_use_case = load_quality_weights(
            data_path("configuration/quality_weights.json", data_dir)
        )
        self._priority_weights = self._load_priority_config(
            data_path("configuration/priority_weights.json", data_dir)
        )

    @staticmethod
    def _load_priority_config(path: Path) -> dict:
        """Load priority weights config once at init."""
        if not path.is_file():
            logger.warning("Priority weights file not found: %s", path)
            return {}
        with open(path) as f:
            data = json.load(f)
        pw: dict = data.get("priority_weights", {})
        return pw

    def generate(self, intent: DeploymentIntent) -> DeploymentSpecification:
        """Generate a complete specification from intent.

        Assembles SLO targets, workload profile, quality weights, and
        priorities from config files + intent parameters.

        Raises:
            ValueError: If use_case is unknown or config data is missing
        """
        slo_targets = self._traffic_gen.generate_slo_targets(intent)
        traffic = self._traffic_gen.generate_profile(intent)

        workload_profile = WorkloadProfile(
            prompt_tokens=traffic.prompt_tokens,
            output_tokens=traffic.output_tokens,
            expected_qps=traffic.expected_qps or 0.0,
        )

        quality_weights = self._get_quality_weights(intent.use_case)
        priorities = self._build_priorities(intent)

        return DeploymentSpecification(
            intent=intent,
            slo_targets=slo_targets,
            workload_profile=workload_profile,
            quality_weights=quality_weights,
            priorities=priorities,
        )

    def _get_quality_weights(self, use_case: str) -> QualityWeights:
        """Look up cached quality weights for a use case."""
        use_case_quality = self._quality_weights_by_use_case.get(use_case)
        if not use_case_quality:
            raise ValueError(f"No quality weights for use case: {use_case}")

        categories = use_case_quality.get("categories", {})
        return QualityWeights(categories=categories)

    def _build_priorities(self, intent: DeploymentIntent) -> Priorities:
        """Build Priorities from cached config and intent priority levels."""
        pw = self._priority_weights
        if not pw:
            raise ValueError("Priority weights config not loaded")

        return Priorities(
            quality=PriorityEntry(
                priority=intent.quality_priority,
                weight=pw["quality"][intent.quality_priority],
            ),
            cost=PriorityEntry(
                priority=intent.cost_priority,
                weight=pw["cost"][intent.cost_priority],
            ),
            latency=PriorityEntry(
                priority=intent.latency_priority,
                weight=pw["latency"][intent.latency_priority],
            ),
        )
