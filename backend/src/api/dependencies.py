"""Shared dependencies for API routes.

This module provides singleton instances and dependency injection
for the API routes. All shared state is initialized here.
"""

import logging
import os

from src.cluster import KubernetesClusterManager, KubernetesDeploymentError
from src.configuration import DeploymentGenerator, YAMLValidator
from src.knowledge_base.model_catalog import ModelCatalog
from src.knowledge_base.slo_templates import SLOTemplateRepository
from src.orchestration.workflow import RecommendationWorkflow

# Configure logging
debug_mode = os.getenv("NEURALNAV_DEBUG", "false").lower() == "true"
log_level = logging.DEBUG if debug_mode else logging.INFO
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Singleton instances
_workflow: RecommendationWorkflow | None = None
_model_catalog: ModelCatalog | None = None
_slo_repo: SLOTemplateRepository | None = None
_deployment_generator: DeploymentGenerator | None = None
_yaml_validator: YAMLValidator | None = None
_cluster_manager: KubernetesClusterManager | None = None


def get_workflow() -> RecommendationWorkflow:
    """Get the recommendation workflow singleton."""
    global _workflow
    if _workflow is None:
        _workflow = RecommendationWorkflow()
    return _workflow


def get_model_catalog() -> ModelCatalog:
    """Get the model catalog singleton."""
    global _model_catalog
    if _model_catalog is None:
        _model_catalog = ModelCatalog()
    return _model_catalog


def get_slo_repo() -> SLOTemplateRepository:
    """Get the SLO template repository singleton."""
    global _slo_repo
    if _slo_repo is None:
        _slo_repo = SLOTemplateRepository()
    return _slo_repo


def get_deployment_generator() -> DeploymentGenerator:
    """Get the deployment generator singleton."""
    global _deployment_generator
    if _deployment_generator is None:
        # Use simulator mode by default (no GPU required for development)
        _deployment_generator = DeploymentGenerator(simulator_mode=True)
    return _deployment_generator


def get_yaml_validator() -> YAMLValidator:
    """Get the YAML validator singleton."""
    global _yaml_validator
    if _yaml_validator is None:
        _yaml_validator = YAMLValidator()
    return _yaml_validator


def get_cluster_manager(namespace: str = "default") -> KubernetesClusterManager | None:
    """Get or create a cluster manager.

    Returns None if cluster is not accessible.
    """
    global _cluster_manager
    if _cluster_manager is None:
        try:
            _cluster_manager = KubernetesClusterManager(namespace=namespace)
            logger.info("Kubernetes cluster manager initialized successfully")
        except KubernetesDeploymentError as e:
            logger.warning(f"Kubernetes cluster not accessible: {e}")
            return None
    return _cluster_manager


def get_cluster_manager_or_raise(namespace: str = "default") -> KubernetesClusterManager:
    """Get or create a cluster manager, raising an exception if not accessible."""
    manager = get_cluster_manager(namespace)
    if manager is None:
        try:
            return KubernetesClusterManager(namespace=namespace)
        except KubernetesDeploymentError as e:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=503, detail=f"Kubernetes cluster not accessible: {str(e)}"
            ) from e
    return manager
