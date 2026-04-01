"""Kubernetes Cluster Management for Deployments.

This module handles actual deployment to Kubernetes clusters using kubectl
and the Kubernetes Python client.
"""

import logging
import os
import subprocess
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class KubernetesDeploymentError(Exception):
    """Raised when deployment to Kubernetes fails."""

    pass


class KubernetesClusterManager:
    """Manage deployments to Kubernetes cluster."""

    def __init__(self, namespace: str = "default", use_kubectl: bool = True):
        """
        Initialize cluster manager.

        Args:
            namespace: Kubernetes namespace for deployments
            use_kubectl: If True, use kubectl CLI; if False, use Python K8s client
        """
        self.namespace = namespace
        self.use_kubectl = use_kubectl
        self._verify_cluster_access()

    def _verify_cluster_access(self):
        """Verify we can access the Kubernetes cluster."""
        try:
            result = subprocess.run(
                ["kubectl", "cluster-info"], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                raise KubernetesDeploymentError(
                    f"Cannot access Kubernetes cluster: {result.stderr}"
                )
            logger.info("Kubernetes cluster access verified")
        except subprocess.TimeoutExpired as e:
            raise KubernetesDeploymentError("kubectl cluster-info timed out") from e
        except FileNotFoundError as e:
            raise KubernetesDeploymentError("kubectl not found in PATH") from e

    def create_namespace_if_not_exists(self) -> bool:
        """Create namespace if it doesn't exist."""
        try:
            # Check if namespace exists
            result = subprocess.run(
                ["kubectl", "get", "namespace", self.namespace],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                logger.info(f"Namespace {self.namespace} already exists")
                return False

            # Create namespace
            result = subprocess.run(
                ["kubectl", "create", "namespace", self.namespace],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                raise KubernetesDeploymentError(f"Failed to create namespace: {result.stderr}")

            logger.info(f"Created namespace: {self.namespace}")
            return True

        except subprocess.TimeoutExpired as e:
            raise KubernetesDeploymentError("Namespace operation timed out") from e

    def apply_yaml(self, yaml_path: str) -> dict[str, Any]:
        """
        Apply a YAML file to the cluster.

        Args:
            yaml_path: Path to YAML file

        Returns:
            Dict with status and output
        """
        if not os.path.exists(yaml_path):
            raise KubernetesDeploymentError(f"YAML file not found: {yaml_path}")

        try:
            result = subprocess.run(
                ["kubectl", "apply", "-f", yaml_path, "-n", self.namespace],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                raise KubernetesDeploymentError(f"Failed to apply {yaml_path}: {result.stderr}")

            logger.info(f"Applied {yaml_path} to namespace {self.namespace}")
            return {
                "success": True,
                "file": yaml_path,
                "output": result.stdout,
                "timestamp": datetime.now().isoformat(),
            }

        except subprocess.TimeoutExpired as e:
            raise KubernetesDeploymentError(f"Timeout applying {yaml_path}") from e

    def deploy_all(self, yaml_files: list[str]) -> dict[str, Any]:
        """
        Deploy all YAML files to the cluster.

        Args:
            yaml_files: List of paths to YAML files

        Returns:
            Dict with deployment results
        """
        # Ensure namespace exists
        self.create_namespace_if_not_exists()

        results = []
        errors = []

        for yaml_file in yaml_files:
            try:
                result = self.apply_yaml(yaml_file)
                results.append(result)
            except KubernetesDeploymentError as e:
                error_info = {
                    "file": yaml_file,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
                errors.append(error_info)
                logger.error(f"Failed to apply {yaml_file}: {e}")

        # Overall success if all files applied
        success = len(errors) == 0

        return {
            "success": success,
            "namespace": self.namespace,
            "applied_files": results,
            "errors": errors,
            "timestamp": datetime.now().isoformat(),
        }

    def get_inferenceservice_status(self, deployment_id: str) -> dict[str, Any]:
        """
        Get status of an InferenceService.

        Args:
            deployment_id: Name of the InferenceService

        Returns:
            Dict with status information
        """
        try:
            # Get InferenceService resource
            result = subprocess.run(
                [
                    "kubectl",
                    "get",
                    "inferenceservice",
                    deployment_id,
                    "-n",
                    self.namespace,
                    "-o",
                    "json",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return {
                    "exists": False,
                    "deployment_id": deployment_id,
                    "error": result.stderr.strip(),
                }

            # Parse JSON output
            import json

            isvc = json.loads(result.stdout)

            # Extract status information
            status = isvc.get("status", {})
            conditions = status.get("conditions", [])

            # Determine overall status
            ready_condition = next((c for c in conditions if c.get("type") == "Ready"), None)

            is_ready = ready_condition.get("status") == "True" if ready_condition else False

            return {
                "exists": True,
                "deployment_id": deployment_id,
                "ready": is_ready,
                "conditions": conditions,
                "url": status.get("url"),
                "address": status.get("address", {}).get("url"),
                "components": status.get("components", {}),
                "raw_status": status,
            }

        except subprocess.TimeoutExpired:
            return {
                "exists": False,
                "deployment_id": deployment_id,
                "error": "Timeout querying InferenceService",
            }
        except Exception as e:
            return {"exists": False, "deployment_id": deployment_id, "error": str(e)}

    def get_deployment_pods(self, deployment_id: str) -> list[dict[str, Any]]:
        """
        Get pods associated with a deployment.

        Args:
            deployment_id: Deployment ID to query

        Returns:
            List of pod information dicts
        """
        try:
            result = subprocess.run(
                [
                    "kubectl",
                    "get",
                    "pods",
                    "-n",
                    self.namespace,
                    "-l",
                    f"serving.kserve.io/inferenceservice={deployment_id}",
                    "-o",
                    "json",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                logger.error(f"Failed to get pods: {result.stderr}")
                return []

            import json

            pod_list = json.loads(result.stdout)

            pods = []
            for pod in pod_list.get("items", []):
                metadata = pod.get("metadata", {})
                status = pod.get("status", {})
                spec = pod.get("spec", {})

                pods.append(
                    {
                        "name": metadata.get("name"),
                        "phase": status.get("phase"),
                        "conditions": status.get("conditions", []),
                        "container_statuses": status.get("containerStatuses", []),
                        "node_name": spec.get("nodeName"),
                        "start_time": status.get("startTime"),
                    }
                )

            return pods

        except Exception as e:
            logger.error(f"Error getting pods: {e}")
            return []

    def delete_inferenceservice(self, deployment_id: str) -> dict[str, Any]:
        """
        Delete an InferenceService.

        Args:
            deployment_id: Name of the InferenceService to delete

        Returns:
            Dict with deletion status
        """
        try:
            result = subprocess.run(
                ["kubectl", "delete", "inferenceservice", deployment_id, "-n", self.namespace],
                capture_output=True,
                text=True,
                timeout=30,
            )

            success = result.returncode == 0

            return {
                "success": success,
                "deployment_id": deployment_id,
                "output": result.stdout if success else result.stderr,
                "timestamp": datetime.now().isoformat(),
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "deployment_id": deployment_id,
                "error": "Timeout deleting InferenceService",
            }
        except Exception as e:
            return {"success": False, "deployment_id": deployment_id, "error": str(e)}

    def list_inferenceservices(self) -> list[str]:
        """
        List all InferenceServices in namespace.

        Returns:
            List of InferenceService names
        """
        try:
            result = subprocess.run(
                [
                    "kubectl",
                    "get",
                    "inferenceservices",
                    "-n",
                    self.namespace,
                    "-o",
                    "jsonpath={.items[*].metadata.name}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                logger.error(f"Failed to list InferenceServices: {result.stderr}")
                return []

            names = result.stdout.strip().split()
            return names

        except Exception as e:
            logger.error(f"Error listing InferenceServices: {e}")
            return []
