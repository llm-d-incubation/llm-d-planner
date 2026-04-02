"""Detect GPU types available on the Kubernetes/OpenShift cluster.

Reads nvidia.com/gpu.product labels from cluster nodes and maps them
to Planner canonical GPU names used in the benchmark database.

The kubernetes package is an optional dependency. When not installed
or when no cluster is accessible, returns an empty list gracefully.
"""

import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# TTL cache for detected GPU types (avoids K8s API call on every request)
_GPU_CACHE_TTL = int(os.getenv("PLANNER_CLUSTER_GPU_CACHE_TTL", "300"))


class _GPUCache:
    """Mutable container for GPU detection cache state."""

    def __init__(self) -> None:
        self.result: list[str] | None = None
        self.expires_at: float = 0.0
        self.lock = threading.Lock()

    def reset(self) -> None:
        with self.lock:
            self.result = None
            self.expires_at = 0.0


_cache = _GPUCache()

# GPU label used by NVIDIA GPU Operator / device plugin
_GPU_LABEL = "nvidia.com/gpu.product"

# Guard kubernetes import — optional dependency
try:
    from kubernetes import client, config  # noqa: F401
    from kubernetes.config import ConfigException  # noqa: F401

    _HAS_KUBERNETES = True
except ImportError:
    _HAS_KUBERNETES = False

# Map nvidia.com/gpu.product label values to Planner canonical names.
# All values MUST be members of CANONICAL_GPUS in gpu_normalizer.py.
# Keys are stored lowercase for case-insensitive matching.
GPU_PRODUCT_MAP: dict[str, str] = {
    "nvidia-l4": "L4",
    "nvidia-a100-sxm4-40gb": "A100-40",
    "nvidia-a100-40gb-pcie": "A100-40",
    "nvidia-a100-sxm4-80gb": "A100-80",
    "nvidia-a100-80gb-pcie": "A100-80",
    "nvidia-h100-80gb-hbm3": "H100",
    "nvidia-h100-sxm5-80gb": "H100",
    "nvidia-h100-pcie-80gb": "H100",
    "nvidia-h200": "H200",
    "nvidia-b200": "B200",
}


def _load_k8s_config() -> None:
    """Load Kubernetes configuration (in-cluster first, then kubeconfig)."""
    try:
        config.load_incluster_config()
        logger.debug("Loaded in-cluster Kubernetes config")
    except ConfigException:
        config.load_kube_config()
        logger.debug("Loaded kubeconfig")


def _list_nodes() -> list[Any]:
    """List all cluster nodes with a 10-second timeout."""
    v1 = client.CoreV1Api()
    response = v1.list_node(_request_timeout=10)
    return list(response.items)


def reset_gpu_cache() -> None:
    """Reset the GPU detection cache (for testing)."""
    _cache.reset()


def detect_cluster_gpus() -> list[str]:
    """Detect GPU types available on the current cluster.

    Reads nvidia.com/gpu.product labels from all cluster nodes,
    maps them to Planner canonical GPU names. Results are cached
    for _GPU_CACHE_TTL seconds to avoid repeated K8s API calls.

    Returns:
        Sorted list of canonical GPU type names (e.g., ["A100-80", "H100"]).
        Empty list if no cluster is available, no GPUs found, or detection fails.
    """
    with _cache.lock:
        # Return cached result if still valid
        if _cache.result is not None and time.monotonic() < _cache.expires_at:
            return _cache.result

        # Check env var toggle
        if os.environ.get("PLANNER_DETECT_CLUSTER_GPUS", "true").lower() == "false":
            logger.debug("Cluster GPU detection disabled via PLANNER_DETECT_CLUSTER_GPUS=false")
            return []

        # Check if kubernetes package is available
        if not _HAS_KUBERNETES:
            logger.info("kubernetes package not installed — skipping cluster GPU detection")
            return []

        try:
            _load_k8s_config()
        except Exception as e:
            logger.info(f"No Kubernetes config available — skipping GPU detection: {e}")
            return []

        try:
            nodes = _list_nodes()
        except Exception as e:
            logger.warning(f"Failed to list cluster nodes — skipping GPU detection: {e}")
            return []

        if not nodes:
            logger.info("No nodes found on cluster")
            return []

        detected: set[str] = set()
        for node in nodes:
            labels = node.metadata.labels or {}
            gpu_label = labels.get(_GPU_LABEL)
            if not gpu_label:
                continue

            canonical = GPU_PRODUCT_MAP.get(gpu_label.lower())
            if canonical:
                detected.add(canonical)
            else:
                logger.warning(
                    f"Unknown GPU label '{gpu_label}' on node '{node.metadata.name}' "
                    f"— not in GPU_PRODUCT_MAP, skipping"
                )

        result = sorted(detected)
        if result:
            logger.info(f"Detected {len(result)} GPU type(s) on cluster: {result}")
        else:
            logger.info("No GPU labels found on any cluster node")

        # Cache the result
        _cache.result = result
        _cache.expires_at = time.monotonic() + _GPU_CACHE_TTL

        return result
