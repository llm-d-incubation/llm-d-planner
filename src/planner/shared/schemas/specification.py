"""Specification-related schemas for traffic profiles and SLO targets."""

from typing import Literal

from pydantic import BaseModel, Field

from .intent import DeploymentIntent


class TrafficProfile(BaseModel):
    """GuideLLM traffic profile for the deployment."""

    prompt_tokens: int = Field(..., description="Target prompt length in tokens (GuideLLM config)")
    output_tokens: int = Field(..., description="Target output length in tokens (GuideLLM config)")
    expected_qps: float | None = Field(None, description="Expected queries per second")


class SLOTargets(BaseModel):
    """Service Level Objective targets for the deployment."""

    ttft_p95_target_ms: int = Field(..., description="Time to First Token target (ms)")
    itl_p95_target_ms: int = Field(..., description="Inter-Token Latency target (ms/token)")
    e2e_p95_target_ms: int = Field(..., description="End-to-end latency target (ms)")
    percentile: Literal["mean", "p90", "p95", "p99"] = Field(
        default="p95", description="Percentile for SLO comparison (mean, p90, p95, p99)"
    )


class DeploymentSpecification(BaseModel):
    """
    Deployment specification generated from user intent.

    This is always generated successfully, even if no viable configurations exist.
    Contains the extracted intent, traffic profile, and SLO targets.
    """

    # User intent
    intent: DeploymentIntent

    # Generated specifications
    traffic_profile: TrafficProfile
    slo_targets: SLOTargets
