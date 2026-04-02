"""Shared dependencies for API routes.

This module provides singleton instances via FastAPI's app.state and
dependency injection via Depends(). All shared state is initialized
during the application lifespan in init_app_state().
"""

import asyncio
import logging
import os
import threading
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from planner.cluster import KubernetesClusterManager, KubernetesDeploymentError
from planner.configuration import DeploymentGenerator, YAMLValidator
from planner.knowledge_base.model_catalog import ModelCatalog
from planner.knowledge_base.slo_templates import SLOTemplateRepository
from planner.orchestration.workflow import RecommendationWorkflow

# Configure logging
debug_mode = os.getenv("PLANNER_DEBUG", "false").lower() == "true"
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
    source = os.getenv("PLANNER_BENCHMARK_SOURCE", "postgresql").strip().lower()
    if source not in _VALID_BENCHMARK_SOURCES:
        logger.warning(
            "Unknown PLANNER_BENCHMARK_SOURCE='%s'; defaulting to 'postgresql'",
            source,
        )
        return "postgresql"
    return source


def _sync_model_catalog_async(
    client: Any,
    database_url: str,
    model_catalog: ModelCatalog,
    quality_scorer: Any,
) -> threading.Thread:
    """Run Model Catalog sync in a background thread.

    The app starts serving immediately (health probes, etc.)
    while catalog data syncs in the background.
    """

    def _sync() -> None:
        try:
            import psycopg2

            from planner.knowledge_base.model_catalog_sync import sync_model_catalog

            logger.info("Background sync: loading Model Catalog data into PostgreSQL...")
            conn = psycopg2.connect(database_url)
            try:
                result = sync_model_catalog(
                    client=client,
                    conn=conn,
                    model_catalog=model_catalog,
                    quality_scorer=quality_scorer,
                )
                if result.errors:
                    logger.warning(
                        "Model Catalog sync completed with %d errors", len(result.errors)
                    )
                else:
                    logger.info("Background sync: Model Catalog data ready")
            finally:
                conn.close()
        except Exception:
            logger.exception("Background Model Catalog sync failed")

    thread = threading.Thread(target=_sync, name="model-catalog-sync", daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Lifespan: initialize all singletons on app.state
# ---------------------------------------------------------------------------


def init_app_state(app: FastAPI) -> None:
    """Initialize all singletons on app.state during lifespan startup."""
    source_type = _get_benchmark_source_type()

    # Always create the same components — single code path
    app.state.model_catalog = ModelCatalog()
    app.state.slo_repo = SLOTemplateRepository()
    app.state.deployment_generator = DeploymentGenerator(simulator_mode=False)
    app.state.yaml_validator = YAMLValidator()
    app.state.cluster_managers = {}  # dict[str, KubernetesClusterManager]

    if source_type == "model_catalog":
        from planner.knowledge_base.model_catalog_client import ModelCatalogClient
        from planner.recommendation.config_finder import ConfigFinder
        from planner.recommendation.quality.usecase_scorer import UseCaseQualityScorer

        client = ModelCatalogClient()
        app.state.model_catalog_client = client
        quality_scorer = UseCaseQualityScorer()

        # Wire shared instances so sync updates propagate to recommendations
        config_finder = ConfigFinder(catalog=app.state.model_catalog, quality_scorer=quality_scorer)
        app.state.workflow = RecommendationWorkflow(config_finder=config_finder)

        database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:planner@localhost:5432/planner",
        )

        logger.info("Using Model Catalog as benchmark source (syncing to PostgreSQL)")
        app.state.model_catalog_sync_thread = _sync_model_catalog_async(
            client, database_url, app.state.model_catalog, quality_scorer
        )
    else:
        app.state.model_catalog_client = None
        app.state.model_catalog_sync_thread = None
        app.state.workflow = RecommendationWorkflow()
        logger.info("Using PostgreSQL as benchmark source")


# ---------------------------------------------------------------------------
# Depends() providers — read from request.app.state
# ---------------------------------------------------------------------------


def get_workflow(request: Request) -> RecommendationWorkflow:
    """Get the recommendation workflow singleton."""
    return cast(RecommendationWorkflow, request.app.state.workflow)


def get_model_catalog(request: Request) -> ModelCatalog:
    """Get the model catalog singleton."""
    return cast(ModelCatalog, request.app.state.model_catalog)


def get_slo_repo(request: Request) -> SLOTemplateRepository:
    """Get the SLO template repository singleton."""
    return cast(SLOTemplateRepository, request.app.state.slo_repo)


def get_deployment_generator(request: Request) -> DeploymentGenerator:
    """Get the deployment generator singleton."""
    return cast(DeploymentGenerator, request.app.state.deployment_generator)


def get_yaml_validator(request: Request) -> YAMLValidator:
    """Get the YAML validator singleton."""
    return cast(YAMLValidator, request.app.state.yaml_validator)


_MAX_CACHED_NAMESPACES = 32


async def get_cluster_manager_or_raise(
    request: Request, namespace: str = "default"
) -> KubernetesClusterManager:
    """Get or create a cluster manager, raising an exception if not accessible."""
    managers: dict[str, KubernetesClusterManager] = request.app.state.cluster_managers
    if namespace not in managers:
        lock = cast(asyncio.Lock, request.app.state.cluster_manager_lock)
        async with lock:
            if namespace not in managers:
                if len(managers) >= _MAX_CACHED_NAMESPACES:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Too many namespaces (limit {_MAX_CACHED_NAMESPACES})",
                    )
                try:
                    managers[namespace] = await run_in_threadpool(
                        KubernetesClusterManager, namespace=namespace
                    )
                    logger.info(
                        "Kubernetes cluster manager initialized for namespace=%s",
                        namespace,
                    )
                except KubernetesDeploymentError as e:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Kubernetes cluster not accessible: {e}",
                    ) from e
    return managers[namespace]
