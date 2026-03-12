"""Shared dependencies for API routes.

This module provides singleton instances via FastAPI's app.state and
dependency injection via Depends(). All shared state is initialized
during the application lifespan in init_app_state().
"""

import asyncio
import logging
import os
from typing import cast

from fastapi import FastAPI, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from neuralnav.cluster import KubernetesClusterManager, KubernetesDeploymentError
from neuralnav.configuration import DeploymentGenerator, YAMLValidator
from neuralnav.knowledge_base.model_catalog import ModelCatalog
from neuralnav.knowledge_base.slo_templates import SLOTemplateRepository
from neuralnav.orchestration.workflow import RecommendationWorkflow

# Configure logging
debug_mode = os.getenv("NEURALNAV_DEBUG", "false").lower() == "true"
log_level = logging.DEBUG if debug_mode else logging.INFO
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: initialize all singletons on app.state
# ---------------------------------------------------------------------------


def init_app_state(app: FastAPI) -> None:
    """Initialize all singletons on app.state during lifespan startup."""
    app.state.model_catalog = ModelCatalog()
    app.state.slo_repo = SLOTemplateRepository()
    app.state.deployment_generator = DeploymentGenerator(simulator_mode=False)
    app.state.yaml_validator = YAMLValidator()
    app.state.cluster_managers = {}  # dict[str, KubernetesClusterManager]
    app.state.workflow = RecommendationWorkflow()


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
