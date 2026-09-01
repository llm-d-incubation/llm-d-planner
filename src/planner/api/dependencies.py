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
from planner.configuration import DeploymentGenerator, LlmdDeploymentGenerator, YAMLValidator
from planner.knowledge_base.model_catalog import ModelCatalog
from planner.knowledge_base.use_cases import UseCaseRepository
from planner.orchestration.workflow import RecommendationWorkflow
from planner.specification.traffic_profile import TrafficProfileGenerator

# Configure logging
debug_mode = os.getenv("PLANNER_DEBUG", "false").lower() == "true"
log_level = logging.DEBUG if debug_mode else logging.INFO
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_VALID_BENCHMARK_SOURCES = {"database", "model_catalog"}


def _get_benchmark_source_type() -> str:
    """Get configured benchmark source type."""
    source = os.getenv("PLANNER_BENCHMARK_SOURCE", "database").strip().lower()
    if source not in _VALID_BENCHMARK_SOURCES:
        logger.warning(
            "Unknown PLANNER_BENCHMARK_SOURCE='%s'; defaulting to 'database'",
            source,
        )
        return "database"
    return source


def _sync_model_catalog_async(
    client: Any,
    benchmark_repo: Any,
    model_catalog: ModelCatalog,
) -> threading.Thread:
    """Run Model Catalog sync in a background thread.

    The app starts serving immediately (health probes, etc.)
    while catalog data syncs in the background.
    """

    def _sync() -> None:
        try:
            from planner.knowledge_base.model_catalog_sync import sync_model_catalog

            logger.info("Background sync: loading Model Catalog data into database...")
            result = sync_model_catalog(
                client=client,
                benchmark_repo=benchmark_repo,
                model_catalog=model_catalog,
            )
            if result.errors:
                logger.warning("Model Catalog sync completed with %d errors", len(result.errors))
            else:
                logger.info("Background sync: Model Catalog data ready")
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

    from planner.knowledge_base.benchmarks import BenchmarkRepository
    from planner.recommendation.config_finder import ConfigFinder
    from planner.recommendation.quality.scoring import (
        build_scoring_engine,
        load_quality_weights,
        validate_quality_weights,
    )

    source_type = _get_benchmark_source_type()

    # Always create the same components — single code path
    app.state.benchmark_repo = BenchmarkRepository()
    app.state.model_catalog = ModelCatalog()
    app.state.use_case_repo = UseCaseRepository()
    app.state.traffic_generator = TrafficProfileGenerator()
    app.state.deployment_generator = DeploymentGenerator(simulator_mode=False)
    app.state.llmd_deployment_generator = LlmdDeploymentGenerator()
    app.state.yaml_validator = YAMLValidator()
    app.state.cluster_managers = {}  # dict[str, KubernetesClusterManager]

    # Build quality scoring engine (shared across both code paths)
    from planner.data._resolver import data_path as resolve_data_path

    engine, quality_metadata = build_scoring_engine()
    app.state.scoring_engine = engine
    app.state.quality_metadata = quality_metadata
    weights_path = resolve_data_path("configuration/quality_weights.json")
    quality_weights = load_quality_weights(weights_path)
    validate_quality_weights(quality_weights)

    config_finder = ConfigFinder(
        benchmark_repo=app.state.benchmark_repo,
        catalog=app.state.model_catalog,
        engine=engine,
        quality_weights=quality_weights,
    )
    app.state.workflow = RecommendationWorkflow(config_finder=config_finder)

    if source_type == "model_catalog":
        from planner.knowledge_base.model_catalog_client import ModelCatalogClient

        client = ModelCatalogClient()
        app.state.model_catalog_client = client

        logger.info("Using Model Catalog as benchmark source (syncing to database)")
        app.state.model_catalog_sync_thread = _sync_model_catalog_async(
            client, app.state.benchmark_repo, app.state.model_catalog
        )
    else:
        app.state.model_catalog_client = None
        app.state.model_catalog_sync_thread = None
        logger.info("Using database as benchmark source")


# ---------------------------------------------------------------------------
# Depends() providers — read from request.app.state
# ---------------------------------------------------------------------------


def get_benchmark_repo(request: Request):
    """Get the benchmark repository singleton."""
    return request.app.state.benchmark_repo


def get_workflow(request: Request) -> RecommendationWorkflow:
    """Get the recommendation workflow singleton."""
    return cast(RecommendationWorkflow, request.app.state.workflow)


def get_model_catalog(request: Request) -> ModelCatalog:
    """Get the model catalog singleton."""
    return cast(ModelCatalog, request.app.state.model_catalog)


def get_use_case_repo(request: Request) -> UseCaseRepository:
    """Get the use case repository singleton."""
    return cast(UseCaseRepository, request.app.state.use_case_repo)


def get_traffic_generator(request: Request) -> TrafficProfileGenerator:
    """Get the traffic profile generator singleton."""
    return cast(TrafficProfileGenerator, request.app.state.traffic_generator)


def get_deployment_generator(request: Request) -> DeploymentGenerator:
    """Get the deployment generator singleton."""
    return cast(DeploymentGenerator, request.app.state.deployment_generator)


def get_llmd_deployment_generator(request: Request) -> LlmdDeploymentGenerator:
    """Get the llm-d deployment generator singleton."""
    return cast(LlmdDeploymentGenerator, request.app.state.llmd_deployment_generator)


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
