"""Shared dependencies for API routes.

This module provides singleton instances via FastAPI's app.state and
dependency injection via Depends(). All shared state is initialized
during the application lifespan in init_app_state().
"""

import logging
import os
import threading

from fastapi import FastAPI, HTTPException, Request, status

from neuralnav.cluster import KubernetesClusterManager, KubernetesDeploymentError
from neuralnav.configuration import DeploymentGenerator, YAMLValidator
from neuralnav.knowledge_base.model_catalog import ModelCatalog
from neuralnav.knowledge_base.slo_templates import SLOTemplateRepository
from neuralnav.orchestration.workflow import RecommendationWorkflow
from neuralnav.shared.schemas import DeploymentMode

# Configure logging
debug_mode = os.getenv("NEURALNAV_DEBUG", "false").lower() == "true"
log_level = logging.DEBUG if debug_mode else logging.INFO
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_VALID_BENCHMARK_SOURCES = {"postgresql", "model_catalog"}


def _get_benchmark_source_type() -> str:
    """Get configured benchmark source type."""
    source = os.getenv("NEURALNAV_BENCHMARK_SOURCE", "postgresql").strip().lower()
    if source not in _VALID_BENCHMARK_SOURCES:
        logger.warning(
            "Unknown NEURALNAV_BENCHMARK_SOURCE='%s'; defaulting to 'postgresql'",
            source,
        )
        return "postgresql"
    return source


def _preload_model_catalog_async(benchmark_source, catalog, quality_scorer) -> None:
    """Preload Model Catalog caches in a background thread.

    Runs in a daemon thread so the app starts serving immediately
    (health probes, etc.) while caches warm in the background.
    """

    def _preload():
        try:
            logger.info("Background preload: loading Model Catalog data...")
            benchmark_source.preload()
            catalog.preload()
            quality_scorer.preload()
            logger.info("Background preload: Model Catalog data ready")
        except Exception:
            logger.exception("Background preload failed; data will load on first request")

    thread = threading.Thread(target=_preload, name="model-catalog-preload", daemon=True)
    thread.start()


# ---------------------------------------------------------------------------
# Lifespan: initialize / close all singletons on app.state
# ---------------------------------------------------------------------------


def init_app_state(app: FastAPI) -> None:
    """Initialize all singletons on app.state during lifespan startup."""
    source_type = _get_benchmark_source_type()
    if source_type == "model_catalog":
        from neuralnav.knowledge_base.model_catalog_benchmarks import (
            ModelCatalogBenchmarkSource,
        )
        from neuralnav.knowledge_base.model_catalog_client import (
            ModelCatalogClient,
        )
        from neuralnav.knowledge_base.model_catalog_models import (
            ModelCatalogModelSource,
        )
        from neuralnav.knowledge_base.model_catalog_quality import (
            ModelCatalogQualityScorer,
        )
        from neuralnav.recommendation.config_finder import ConfigFinder

        client = ModelCatalogClient()
        app.state.model_catalog_client = client
        benchmark_source = ModelCatalogBenchmarkSource(client)
        catalog = ModelCatalogModelSource(client)
        quality_scorer = ModelCatalogQualityScorer(client)
        config_finder = ConfigFinder(
            benchmark_repo=benchmark_source,
            catalog=catalog,
            quality_scorer=quality_scorer,
        )
        logger.info("Using Model Catalog as benchmark source")
        app.state.workflow = RecommendationWorkflow(config_finder=config_finder)
        _preload_model_catalog_async(benchmark_source, catalog, quality_scorer)
    else:
        app.state.model_catalog_client = None
        logger.info("Using PostgreSQL as benchmark source")
        app.state.workflow = RecommendationWorkflow()

    app.state.model_catalog = ModelCatalog()
    app.state.slo_repo = SLOTemplateRepository()
    app.state.deployment_generator = DeploymentGenerator(simulator_mode=False)
    app.state.yaml_validator = YAMLValidator()
    app.state.cluster_manager = None  # Lazy — created on first request


def close_app_state(app: FastAPI) -> None:
    """Close resources and clear state."""
    client = getattr(app.state, "model_catalog_client", None)
    if client is not None and hasattr(client, "close"):
        try:
            client.close()
        except Exception:
            logger.exception("Error closing Model Catalog client")


# ---------------------------------------------------------------------------
# Depends() providers — read from request.app.state
# ---------------------------------------------------------------------------


def get_workflow(request: Request) -> RecommendationWorkflow:
    """Get the recommendation workflow singleton."""
    return request.app.state.workflow


def get_model_catalog(request: Request) -> ModelCatalog:
    """Get the model catalog singleton."""
    return request.app.state.model_catalog


def get_slo_repo(request: Request) -> SLOTemplateRepository:
    """Get the SLO template repository singleton."""
    return request.app.state.slo_repo


def get_deployment_generator(request: Request) -> DeploymentGenerator:
    """Get the deployment generator singleton."""
    return request.app.state.deployment_generator


def get_yaml_validator(request: Request) -> YAMLValidator:
    """Get the YAML validator singleton."""
    return request.app.state.yaml_validator


def get_deployment_mode(request: Request) -> DeploymentMode:
    """Return the current deployment mode."""
    gen = request.app.state.deployment_generator
    return DeploymentMode.SIMULATOR if gen.simulator_mode else DeploymentMode.PRODUCTION


def set_deployment_mode(request: Request, mode: DeploymentMode) -> DeploymentMode:
    """Set the deployment mode and return the new mode."""
    gen = request.app.state.deployment_generator
    gen.simulator_mode = mode == DeploymentMode.SIMULATOR
    logger.info(f"Deployment mode changed to: {mode.value}")
    return mode


def get_cluster_manager_or_raise(
    request: Request, namespace: str = "default"
) -> KubernetesClusterManager:
    """Get or create a cluster manager, raising an exception if not accessible."""
    if request.app.state.cluster_manager is None:
        try:
            request.app.state.cluster_manager = KubernetesClusterManager(namespace=namespace)
            logger.info("Kubernetes cluster manager initialized successfully")
        except KubernetesDeploymentError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Kubernetes cluster not accessible: {e}",
            ) from e
    return request.app.state.cluster_manager
