"""Configuration and deployment endpoints."""

import logging
from datetime import datetime
from typing import Any, Literal

import yaml
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from planner.api.dependencies import (
    get_cluster_manager_or_raise,
    get_deployment_generator,
    get_llmd_deployment_generator,
    get_yaml_validator,
)
from planner.configuration import DeploymentGenerator, LlmdDeploymentGenerator, YAMLValidator
from planner.shared.schemas import DeploymentConfiguration, DeploymentMode
from planner.shared.schemas.recommendation import DeploymentBundle

StackType = Literal["vllm", "llm-d"]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["configuration"])


class GenerateDeploymentRequest(BaseModel):
    """Request to generate deployment YAML from configuration."""

    configuration: DeploymentConfiguration
    namespace: str = "default"
    stack: StackType = "vllm"
    pd_enabled: bool = False
    prefill_replicas: int = Field(1, ge=1, le=32)
    decode_replicas: int = Field(1, ge=1, le=32)


class DeploymentModeRequest(BaseModel):
    """Request to set deployment mode."""

    mode: DeploymentMode


class DeployBundleRequest(BaseModel):
    """Request to deploy a DeploymentBundle to cluster."""

    bundle: DeploymentBundle


def _generate_yaml_from_config(
    config: DeploymentConfiguration,
    namespace: str,
    stack: StackType,
    deployment_generator: DeploymentGenerator,
    llmd_generator: LlmdDeploymentGenerator,
    yaml_validator: YAMLValidator,
    pd_enabled: bool = False,
    prefill_replicas: int = 1,
    decode_replicas: int = 1,
) -> dict[str, Any]:
    """Generate YAML files from a deployment configuration."""
    logger.info(f"Generating deployment for model: {config.model_name} (stack={stack})")

    if stack == "llm-d":
        result = llmd_generator.generate_all(
            config=config,
            namespace=namespace,
            pd_enabled=pd_enabled,
            prefill_replicas=prefill_replicas,
            decode_replicas=decode_replicas,
        )
    elif stack == "vllm":
        result = deployment_generator.generate_all(config=config, namespace=namespace)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown stack: {stack}",
        )

    try:
        yaml_validator.validate_all(result["files"])
        logger.info(f"All YAML files validated for deployment: {result['deployment_id']}")
    except Exception as e:
        logger.error(f"YAML validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generated YAML validation failed: {str(e)}",
        ) from e

    return result


@router.get("/deployment-mode")
async def get_mode(http_request: Request):
    """Return the current deployment mode ('production' or 'simulator')."""
    gen = http_request.app.state.deployment_generator
    mode = DeploymentMode.SIMULATOR if gen.simulator_mode else DeploymentMode.PRODUCTION
    return {"mode": mode}


@router.put("/deployment-mode")
async def set_mode(request: DeploymentModeRequest, http_request: Request):
    """Set the deployment mode."""
    gen = http_request.app.state.deployment_generator
    gen.simulator_mode = request.mode == DeploymentMode.SIMULATOR
    logger.info(f"Deployment mode changed to: {request.mode.value}")
    return {"mode": request.mode}


@router.post("/generate-deployment", response_model=DeploymentBundle)
async def generate_deployment(
    request: GenerateDeploymentRequest,
    deployment_generator: DeploymentGenerator = Depends(get_deployment_generator),
    llmd_generator: LlmdDeploymentGenerator = Depends(get_llmd_deployment_generator),
    yaml_validator: YAMLValidator = Depends(get_yaml_validator),
):
    """Generate deployment files and return as a DeploymentBundle.

    This endpoint generates YAML files but does NOT deploy them to the cluster.
    Use /deploy-bundle-to-cluster to apply the bundle to a cluster.
    """
    try:
        result = _generate_yaml_from_config(
            request.configuration,
            request.namespace,
            request.stack,
            deployment_generator,
            llmd_generator,
            yaml_validator,
            pd_enabled=request.pd_enabled,
            prefill_replicas=request.prefill_replicas,
            decode_replicas=request.decode_replicas,
        )

        return DeploymentBundle(
            deployment_id=result["deployment_id"],
            namespace=request.namespace,
            stack=request.stack,
            configuration=request.configuration,
            files=result["contents"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate deployment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate deployment: {str(e)}",
        ) from e


@router.post("/deploy-bundle-to-cluster")
async def deploy_bundle_to_cluster(
    request: DeployBundleRequest,
    http_request: Request,
):
    """
    Deploy a DeploymentBundle to Kubernetes cluster.

    Applies YAML content directly via kubectl stdin — no intermediate
    files are written to disk.

    Args:
        request: Request containing a DeploymentBundle

    Returns:
        Deployment result with status

    Raises:
        HTTPException: If cluster not accessible or deployment fails
    """
    bundle = request.bundle
    manager = await get_cluster_manager_or_raise(http_request, bundle.namespace)

    try:
        logger.info(f"Deploying bundle {bundle.deployment_id} to cluster")

        # Step 1: Create namespace if it doesn't exist
        await run_in_threadpool(manager.create_namespace_if_not_exists)

        # Step 2: Validate YAML content before applying
        for name, yaml_content in bundle.files.items():
            try:
                list(yaml.safe_load_all(yaml_content))
                logger.info(f"YAML validation passed for {name}")
            except yaml.YAMLError as e:
                logger.error(f"YAML validation failed for {name}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid YAML syntax in {name}: {str(e)}",
                ) from e

        # Step 3: Apply each YAML document via kubectl apply -f - (stdin)
        applied_files = []
        for name, yaml_content in bundle.files.items():
            result = await run_in_threadpool(manager.apply_yaml_content, yaml_content)
            if not result["success"]:
                logger.error(f"Failed to apply {name}: {result.get('error')}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to apply {name}: {result.get('error')}",
                )
            applied_files.append(name)

        logger.info(
            f"Successfully deployed bundle {bundle.deployment_id} to cluster ({len(applied_files)} files)"
        )

        return {
            "success": True,
            "deployment_id": bundle.deployment_id,
            "namespace": bundle.namespace,
            "files_applied": applied_files,
            "message": f"Successfully deployed {bundle.deployment_id} to Kubernetes cluster",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to deploy bundle to cluster: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deploy bundle to cluster: {str(e)}",
        ) from e


@router.get("/cluster-status")
async def get_cluster_status(http_request: Request, namespace: str = "default"):
    """
    Get Kubernetes cluster status.

    Returns:
        Cluster accessibility and basic info
    """
    try:
        manager = await get_cluster_manager_or_raise(http_request, namespace)
        deployments = await run_in_threadpool(manager.list_inferenceservices)

        return {
            "accessible": True,
            "namespace": manager.namespace,
            "inference_services": deployments,
            "count": len(deployments),
            "message": "Cluster accessible",
        }
    except HTTPException as e:
        logger.error("Failed to query cluster status: %s", e.detail)
        return {"accessible": False, "error": e.detail}
    except Exception as e:
        logger.error(f"Failed to query cluster status: {e}")
        return {"accessible": False, "error": str(e)}


@router.get("/deployments/{deployment_id}/k8s-status")
async def get_k8s_deployment_status(
    deployment_id: str, http_request: Request, namespace: str = "default"
):
    """
    Get actual Kubernetes deployment status (not mock data).

    Args:
        deployment_id: InferenceService name
        namespace: Kubernetes namespace

    Returns:
        Real deployment status from cluster

    Raises:
        HTTPException: If cluster not accessible
    """
    manager = await get_cluster_manager_or_raise(http_request, namespace)

    try:
        isvc_status = await run_in_threadpool(manager.get_inferenceservice_status, deployment_id)
        pods = await run_in_threadpool(manager.get_deployment_pods, deployment_id)

        return {
            "deployment_id": deployment_id,
            "inferenceservice": isvc_status,
            "pods": pods,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get K8s deployment status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get deployment status: {str(e)}",
        ) from e


@router.delete("/deployments/{deployment_id}")
async def delete_deployment(deployment_id: str, http_request: Request, namespace: str = "default"):
    """
    Delete a deployment from the cluster.

    Args:
        deployment_id: InferenceService name to delete
        namespace: Kubernetes namespace

    Returns:
        Deletion result

    Raises:
        HTTPException: If cluster not accessible or deletion fails
    """
    manager = await get_cluster_manager_or_raise(http_request, namespace)

    try:
        result = await run_in_threadpool(manager.delete_inferenceservice, deployment_id)

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete deployment: {result.get('error', 'Unknown error')}",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete deployment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete deployment: {str(e)}",
        ) from e


@router.get("/deployments")
async def list_all_deployments(http_request: Request, namespace: str = "default"):
    """
    List all InferenceServices in the cluster with their detailed status.

    Args:
        namespace: Kubernetes namespace

    Returns:
        List of deployments with status information

    Raises:
        HTTPException: If cluster not accessible
    """
    manager = await get_cluster_manager_or_raise(http_request, namespace)

    try:
        deployment_ids = await run_in_threadpool(manager.list_inferenceservices)

        deployments = []
        for deployment_id in deployment_ids:
            svc_status = await run_in_threadpool(manager.get_inferenceservice_status, deployment_id)
            pods = await run_in_threadpool(manager.get_deployment_pods, deployment_id)

            deployments.append({"deployment_id": deployment_id, "status": svc_status, "pods": pods})

        return {
            "success": True,
            "count": len(deployments),
            "deployments": deployments,
            "namespace": manager.namespace,
        }

    except Exception as e:
        logger.error(f"Failed to list deployments: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list deployments: {str(e)}",
        ) from e
