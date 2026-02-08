"""Configuration and deployment endpoints."""

import logging
import random
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.dependencies import (
    get_cluster_manager_or_raise,
    get_deployment_generator,
    get_yaml_validator,
)
from src.cluster import KubernetesClusterManager
from src.shared.schemas import DeploymentRecommendation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["configuration"])


class DeploymentRequest(BaseModel):
    """Request to generate deployment YAML files."""

    recommendation: DeploymentRecommendation
    namespace: str = "default"


class DeploymentResponse(BaseModel):
    """Response with generated deployment files."""

    deployment_id: str
    namespace: str
    files: dict
    success: bool = True
    message: str | None = None


class DeploymentStatusResponse(BaseModel):
    """Mock deployment status response."""

    deployment_id: str
    status: str
    slo_compliance: dict
    resource_utilization: dict
    cost_analysis: dict
    traffic_patterns: dict
    recommendations: list[str] | None = None


@router.post("/deploy", response_model=DeploymentResponse)
async def deploy_model(request: DeploymentRequest):
    """
    Generate deployment YAML files from recommendation.

    Args:
        request: Deployment request with recommendation

    Returns:
        Deployment response with file paths

    Raises:
        HTTPException: If deployment generation fails
    """
    try:
        deployment_generator = get_deployment_generator()
        yaml_validator = get_yaml_validator()

        logger.info(f"Generating deployment for model: {request.recommendation.model_name}")

        # Generate all YAML files
        result = deployment_generator.generate_all(
            recommendation=request.recommendation, namespace=request.namespace
        )

        # Validate generated files
        try:
            yaml_validator.validate_all(result["files"])
            logger.info(f"All YAML files validated for deployment: {result['deployment_id']}")
        except Exception as e:
            logger.error(f"YAML validation failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Generated YAML validation failed: {str(e)}"
            ) from e

        return DeploymentResponse(
            deployment_id=result["deployment_id"],
            namespace=result["namespace"],
            files=result["files"],
            success=True,
            message=f"Deployment files generated successfully for {result['deployment_id']}",
        )

    except Exception as e:
        logger.error(f"Failed to generate deployment: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate deployment: {str(e)}"
        ) from e


@router.get("/deployments/{deployment_id}/status", response_model=DeploymentStatusResponse)
async def get_deployment_status(deployment_id: str):
    """
    Get mock deployment status for observability demonstration.

    This endpoint returns simulated observability data to demonstrate
    Component 9 (Inference Observability & SLO Monitoring).

    Args:
        deployment_id: Deployment identifier

    Returns:
        Mock deployment status with observability metrics

    Raises:
        HTTPException: If deployment not found
    """
    try:
        logger.info(f"Fetching mock status for deployment: {deployment_id}")

        # Mock observability data (in production, this would query Prometheus/Grafana)
        base_ttft = 185
        base_tpot = 48
        base_e2e = 1850

        mock_status = DeploymentStatusResponse(
            deployment_id=deployment_id,
            status="running",
            slo_compliance={
                "ttft_p90_ms": base_ttft + random.randint(-10, 15),
                "ttft_target_ms": 200,
                "ttft_compliant": True,
                "tpot_p90_ms": base_tpot + random.randint(-3, 5),
                "tpot_target_ms": 50,
                "tpot_compliant": True,
                "e2e_p90_ms": base_e2e + random.randint(-50, 100),
                "e2e_target_ms": 2000,
                "e2e_compliant": True,
                "throughput_qps": 122 + random.randint(-5, 10),
                "throughput_target_qps": 100,
                "throughput_compliant": True,
                "uptime_pct": 99.94 + random.uniform(-0.05, 0.05),
                "uptime_target_pct": 99.9,
                "uptime_compliant": True,
            },
            resource_utilization={
                "gpu_utilization_pct": 78 + random.randint(-5, 10),
                "gpu_memory_used_gb": 14.2 + random.uniform(-1, 2),
                "gpu_memory_total_gb": 24,
                "avg_batch_size": 18 + random.randint(-3, 5),
                "queue_depth": random.randint(0, 5),
                "token_throughput_per_gpu": 3500 + random.randint(-200, 300),
            },
            cost_analysis={
                "actual_cost_per_hour_usd": 0.95 + random.uniform(-0.05, 0.1),
                "predicted_cost_per_hour_usd": 1.00,
                "actual_cost_per_month_usd": 812 + random.randint(-30, 50),
                "predicted_cost_per_month_usd": 800,
                "cost_per_1k_tokens_usd": 0.042 + random.uniform(-0.002, 0.005),
                "predicted_cost_per_1k_tokens_usd": 0.040,
            },
            traffic_patterns={
                "avg_prompt_tokens": 165 + random.randint(-10, 20),
                "predicted_prompt_tokens": 150,
                "avg_generation_tokens": 220 + random.randint(-15, 25),
                "predicted_generation_tokens": 200,
                "peak_qps": 95 + random.randint(-5, 10),
                "predicted_peak_qps": 100,
                "requests_last_hour": 7200 + random.randint(-500, 800),
                "requests_last_24h": 172800 + random.randint(-5000, 10000),
            },
            recommendations=[
                "GPU utilization is 78%, below the 80% efficiency target. Consider downsizing to reduce cost.",
                "Actual traffic is 10% higher than predicted. Monitor for potential capacity constraints.",
                "All SLO targets are being met with headroom. Configuration is performing well.",
            ],
        )

        return mock_status

    except Exception as e:
        logger.error(f"Failed to get deployment status: {e}")
        raise HTTPException(status_code=404, detail=f"Deployment not found: {deployment_id}") from e


@router.post("/deploy-to-cluster")
async def deploy_to_cluster(request: DeploymentRequest):
    """
    Deploy model to Kubernetes cluster.

    This endpoint generates YAML files AND applies them to the cluster.

    Args:
        request: Deployment request with recommendation and namespace

    Returns:
        Deployment result with status

    Raises:
        HTTPException: If cluster not accessible or deployment fails
    """
    manager = get_cluster_manager_or_raise(request.namespace)
    deployment_generator = get_deployment_generator()
    yaml_validator = get_yaml_validator()

    try:
        logger.info(f"Deploying model to cluster: {request.recommendation.model_name}")

        # Step 1: Generate YAML files
        result = deployment_generator.generate_all(
            recommendation=request.recommendation, namespace=request.namespace
        )

        deployment_id = result["deployment_id"]
        files = result["files"]

        # Step 2: Validate generated files
        try:
            yaml_validator.validate_all(files)
            logger.info(f"YAML validation passed for: {deployment_id}")
        except Exception as e:
            logger.error(f"YAML validation failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Generated YAML validation failed: {str(e)}"
            ) from e

        # Step 3: Deploy to cluster
        yaml_file_paths = [files["inferenceservice"], files["autoscaling"]]

        deployment_result = manager.deploy_all(yaml_file_paths)

        if not deployment_result["success"]:
            logger.error(f"Deployment failed: {deployment_result['errors']}")
            raise HTTPException(
                status_code=500, detail=f"Deployment failed: {deployment_result['errors']}"
            )

        logger.info(f"Successfully deployed {deployment_id} to cluster")

        return {
            "success": True,
            "deployment_id": deployment_id,
            "namespace": request.namespace,
            "files": files,
            "deployment_result": deployment_result,
            "message": f"Successfully deployed {deployment_id} to Kubernetes cluster",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to deploy to cluster: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to deploy to cluster: {str(e)}") from e


@router.get("/cluster-status")
async def get_cluster_status():
    """
    Get Kubernetes cluster status.

    Returns:
        Cluster accessibility and basic info
    """
    try:
        temp_manager = KubernetesClusterManager(namespace="default")
        deployments = temp_manager.list_inferenceservices()

        return {
            "accessible": True,
            "namespace": temp_manager.namespace,
            "inference_services": deployments,
            "count": len(deployments),
            "message": "Cluster accessible",
        }
    except Exception as e:
        logger.error(f"Failed to query cluster status: {e}")
        return {"accessible": False, "error": str(e)}


@router.get("/deployments/{deployment_id}/k8s-status")
async def get_k8s_deployment_status(deployment_id: str):
    """
    Get actual Kubernetes deployment status (not mock data).

    Args:
        deployment_id: InferenceService name

    Returns:
        Real deployment status from cluster

    Raises:
        HTTPException: If cluster not accessible
    """
    manager = get_cluster_manager_or_raise("default")

    try:
        isvc_status = manager.get_inferenceservice_status(deployment_id)
        pods = manager.get_deployment_pods(deployment_id)

        return {
            "deployment_id": deployment_id,
            "inferenceservice": isvc_status,
            "pods": pods,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get K8s deployment status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get deployment status: {str(e)}"
        ) from e


@router.get("/deployments/{deployment_id}/yaml")
async def get_deployment_yaml(deployment_id: str):
    """
    Retrieve generated YAML files for a deployment.

    Args:
        deployment_id: Deployment identifier

    Returns:
        Dictionary with YAML file contents

    Raises:
        HTTPException: If YAML files not found
    """
    try:
        deployment_generator = get_deployment_generator()
        output_dir = deployment_generator.output_dir

        yaml_files = {}
        for file_path in output_dir.glob(f"{deployment_id}*.yaml"):
            with open(file_path) as f:
                yaml_files[file_path.name] = f.read()

        if not yaml_files:
            raise HTTPException(
                status_code=404, detail=f"No YAML files found for deployment {deployment_id}"
            )

        return {"deployment_id": deployment_id, "files": yaml_files, "count": len(yaml_files)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve YAML files: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve YAML files: {str(e)}"
        ) from e


@router.delete("/deployments/{deployment_id}")
async def delete_deployment(deployment_id: str):
    """
    Delete a deployment from the cluster.

    Args:
        deployment_id: InferenceService name to delete

    Returns:
        Deletion result

    Raises:
        HTTPException: If cluster not accessible or deletion fails
    """
    manager = get_cluster_manager_or_raise("default")

    try:
        result = manager.delete_inferenceservice(deployment_id)

        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete deployment: {result.get('error', 'Unknown error')}",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete deployment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete deployment: {str(e)}") from e


@router.get("/deployments")
async def list_all_deployments():
    """
    List all InferenceServices in the cluster with their detailed status.

    Returns:
        List of deployments with status information

    Raises:
        HTTPException: If cluster not accessible
    """
    manager = get_cluster_manager_or_raise("default")

    try:
        deployment_ids = manager.list_inferenceservices()

        deployments = []
        for deployment_id in deployment_ids:
            status = manager.get_inferenceservice_status(deployment_id)
            pods = manager.get_deployment_pods(deployment_id)

            deployments.append({"deployment_id": deployment_id, "status": status, "pods": pods})

        return {
            "success": True,
            "count": len(deployments),
            "deployments": deployments,
            "namespace": manager.namespace,
        }

    except Exception as e:
        logger.error(f"Failed to list deployments: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list deployments: {str(e)}") from e
