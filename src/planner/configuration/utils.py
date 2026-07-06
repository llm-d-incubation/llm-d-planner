"""Shared utilities for deployment configuration generators."""

from __future__ import annotations

import re
from datetime import datetime

from planner.shared.schemas import DeploymentRecommendation

_MODEL_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/-]*$")
_NAMESPACE_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


def validate_model_id(model_id: str) -> None:
    """Validate model_id to prevent YAML injection."""
    if not _MODEL_ID_RE.match(model_id):
        raise ValueError(f"Invalid model_id format: {model_id}")


def validate_namespace(namespace: str) -> None:
    """Validate Kubernetes namespace name."""
    if not _NAMESPACE_RE.match(namespace):
        raise ValueError(f"Invalid namespace format: {namespace}")


def generate_deployment_id(recommendation: DeploymentRecommendation) -> str:
    """Generate a Kubernetes-safe deployment ID.

    Must start with a letter, only lowercase alphanumeric and hyphens,
    max 44 characters (KServe adds "-predictor-default" suffix, total must be <= 63).
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    use_case = recommendation.intent.use_case.replace("_", "-")

    model_name = (recommendation.model_id or "unknown").split("/")[-1].lower()
    model_name = re.sub(r"[^a-z0-9-]", "-", model_name)
    model_name = re.sub(r"-+", "-", model_name).strip("-")

    deployment_id = f"{use_case}-{model_name}-{timestamp}"

    max_len = 44
    if len(deployment_id) > max_len:
        max_model_len = max_len - len(use_case) - len(timestamp) - 2
        model_name = model_name[:max_model_len].rstrip("-")
        deployment_id = f"{use_case}-{model_name}-{timestamp}"

    return deployment_id
