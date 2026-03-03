"""Quality/accuracy scorer using RHOAI Model Catalog accuracy-metrics artifacts."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuralnav.knowledge_base.model_catalog_client import ModelCatalogClient

logger = logging.getLogger(__name__)

_CACHE_TTL = 3600


class ModelCatalogQualityScorer:
    """Score model quality using accuracy-metrics from Model Catalog.

    Drop-in replacement for UseCaseQualityScorer when running with RHOAI.
    Uses overall_average from accuracy-metrics artifacts.
    """

    def __init__(self, client: ModelCatalogClient) -> None:
        self._client = client
        self._scores: dict[str, float] = {}  # model_name (lowercase) -> overall_average
        self._loaded_at: float = 0

    def preload(self) -> None:
        """Eagerly load quality scores cache (call during app startup)."""
        self._load_all()

    def _ensure_loaded(self) -> None:
        if self._scores and (time.time() - self._loaded_at) < _CACHE_TTL:
            return
        self._load_all()

    def _load_all(self) -> None:
        scores: dict[str, float] = {}
        models = self._client.list_models()
        for model in models:
            model_name = model.get("name", "")
            if not model_name:
                continue
            source_id = model.get("source_id")
            try:
                artifacts = self._client.get_model_artifacts(model_name, source_id=source_id)
            except Exception:
                logger.warning("Failed to fetch artifacts for %s, skipping", model_name)
                continue
            for artifact in artifacts:
                if (
                    artifact.get("artifactType") == "metrics-artifact"
                    and artifact.get("metricsType") == "accuracy-metrics"
                ):
                    props = artifact.get("customProperties", {})
                    avg_entry = props.get("overall_average")
                    if avg_entry:
                        scores[model_name.lower()] = float(avg_entry.get("double_value", 0))
                    break  # One accuracy artifact per model
        self._scores = scores
        self._loaded_at = time.time()
        logger.info("Loaded accuracy scores for %d models from Model Catalog", len(scores))

    def get_quality_score(self, model_name: str, use_case: str) -> float:
        """Get quality score for a model.

        Args:
            model_name: Full model name (e.g. "RedHatAI/granite-3.1-8b-instruct").
            use_case: Accepted for API compat but not used for weighting.

        Returns:
            Quality score (overall_average), or 0.0 if model not found.
        """
        self._ensure_loaded()
        return self._scores.get(model_name.lower(), 0.0)

    def get_available_use_cases(self) -> list[str]:
        """Return available use cases (for API compat with UseCaseQualityScorer)."""
        return ["chatbot_conversational"]
