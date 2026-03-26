"""Recommendation-related schemas for GPU configs, scores, and ranked responses."""

from typing import Literal

from pydantic import BaseModel, Field

from .intent import DeploymentIntent
from .specification import DeploymentSpecification, SLOTargets, TrafficProfile


class GPUConfig(BaseModel):
    """GPU configuration specification."""

    gpu_type: str = Field(..., description="GPU type (e.g., NVIDIA-L4, NVIDIA-A100-80GB)")
    gpu_count: int = Field(..., description="Total number of GPUs")
    tensor_parallel: int = Field(1, description="Tensor parallelism degree")
    replicas: int = Field(1, description="Number of independent replicas")


class ConfigurationScores(BaseModel):
    """Scores for a deployment configuration (0-100 scale)."""

    accuracy_score: int = Field(..., description="Model accuracy/capability score (0-100)")
    price_score: int = Field(..., description="Cost efficiency score - inverse of cost (0-100)")
    latency_score: int = Field(..., description="SLO headroom score (0-100)")
    complexity_score: int = Field(..., description="Deployment simplicity score (0-100)")
    balanced_score: float = Field(..., description="Weighted composite score (0-100)")
    slo_status: Literal["compliant", "near_miss", "exceeds"] = Field(
        ..., description="SLO compliance status"
    )


class DeploymentRecommendation(BaseModel):
    """Complete deployment recommendation with all specifications."""

    model_config = {"protected_namespaces": ()}

    # Input intent
    intent: DeploymentIntent

    # Generated specifications
    traffic_profile: TrafficProfile
    slo_targets: SLOTargets

    # Recommended configuration (None when no viable config found)
    model_id: str | None = Field(None, description="Recommended model identifier")
    model_name: str | None = Field(None, description="Human-readable model name")
    model_uri: str | None = Field(None, description="Model artifact URI (e.g., OCI registry URI)")
    gpu_config: GPUConfig | None = None

    # Performance predictions (None when no viable config found)
    predicted_ttft_p95_ms: int | None = None
    predicted_itl_p95_ms: int | None = None
    predicted_e2e_p95_ms: int | None = None
    predicted_throughput_qps: float | None = None

    # All percentile metrics from benchmark (for UI to display based on user selection)
    benchmark_metrics: dict | None = Field(
        default=None, description="All percentile metrics from benchmark"
    )

    # Cost estimation (None when no viable config found)
    cost_per_hour_usd: float | None = None
    cost_per_month_usd: float | None = None

    # Metadata
    meets_slo: bool = Field(False, description="Whether configuration meets SLO targets")
    reasoning: str = Field(..., description="Explanation of recommendation choice or error message")
    alternative_options: list[dict] | None = Field(
        default=None, description="Alternative configurations with trade-offs"
    )

    # Multi-criteria scores (added for Solution Ranking feature)
    scores: ConfigurationScores | None = Field(
        default=None, description="Multi-criteria scores for ranking"
    )

    def to_alternative_dict(self) -> dict:
        """
        Convert recommendation to alternative option format.

        This is used when building the alternative_options list to avoid
        code duplication across config_finder.py and workflow.py.

        Returns:
            Dictionary with all fields needed for alternative comparison
        """
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "model_uri": self.model_uri,
            "gpu_config": self.gpu_config.model_dump() if self.gpu_config else None,
            "predicted_ttft_p95_ms": self.predicted_ttft_p95_ms,
            "predicted_itl_p95_ms": self.predicted_itl_p95_ms,
            "predicted_e2e_p95_ms": self.predicted_e2e_p95_ms,
            "predicted_throughput_qps": self.predicted_throughput_qps,
            "cost_per_hour_usd": self.cost_per_hour_usd,
            "cost_per_month_usd": self.cost_per_month_usd,
            "reasoning": self.reasoning,
            "scores": self.scores.model_dump() if self.scores else None,
            "benchmark_metrics": self.benchmark_metrics,
        }


class RankedRecommendationsResponse(BaseModel):
    """Response containing multiple ranked recommendation lists.

    Provides 5 different views of the same configurations, each sorted
    by a different criterion to help users explore trade-offs.
    """

    # Filters applied
    min_accuracy_threshold: int | None = Field(
        default=None, description="Minimum accuracy score filter applied"
    )
    max_cost_ceiling: float | None = Field(
        default=None, description="Maximum monthly cost filter applied (USD)"
    )
    include_near_miss: bool = Field(
        default=True, description="Whether near-SLO configurations are included"
    )

    # Original specification
    specification: DeploymentSpecification | None = Field(
        default=None, description="The generated deployment specification"
    )

    # Ranked lists (top 5 each, sorted by respective criterion)
    best_accuracy: list[DeploymentRecommendation] = Field(
        default_factory=list, description="Top configs sorted by accuracy score"
    )
    lowest_cost: list[DeploymentRecommendation] = Field(
        default_factory=list, description="Top configs sorted by price score"
    )
    lowest_latency: list[DeploymentRecommendation] = Field(
        default_factory=list, description="Top configs sorted by latency score"
    )
    simplest: list[DeploymentRecommendation] = Field(
        default_factory=list, description="Top configs sorted by complexity score"
    )
    balanced: list[DeploymentRecommendation] = Field(
        default_factory=list, description="Top configs sorted by weighted composite score"
    )

    # Statistics
    total_configs_evaluated: int = Field(
        default=0, description="Total number of configurations evaluated"
    )
    configs_after_filters: int = Field(
        default=0, description="Number of configurations after applying filters"
    )
