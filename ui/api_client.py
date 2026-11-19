"""Backend API client for the Planner UI.

All HTTP communication with the backend lives here.
"""

import contextlib
import logging
import os
from typing import Any, cast

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = 10  # seconds; override per-call for heavier APIs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@st.cache_data(ttl=300)
def fetch_slo_defaults(use_case: str) -> dict | None:
    """Fetch default SLO values for a use case from the backend API.

    Returns dict with ttft_ms, itl_ms, e2e_ms each containing min, max, default.
    Cached for 5 minutes.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/slo-defaults/{use_case}",
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            return cast(dict[Any, Any] | None, data.get("slo_defaults"))
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch SLO defaults for {use_case}: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_expected_rps(use_case: str, user_count: int) -> dict | None:
    """Fetch expected RPS for a use case from the backend API.

    Uses research-backed workload patterns to calculate:
    - expected_rps: average requests per second
    - peak_rps: peak capacity needed
    - workload_params: active_fraction, requests_per_min, etc.

    Cached for 5 minutes.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/expected-rps/{use_case}",
            params={"user_count": user_count},
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            return cast(dict[Any, Any], data)
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch expected RPS for {use_case}: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_workload_profile(use_case: str) -> dict | None:
    """Fetch workload profile for a use case from the backend API.

    Returns dict with workload_profile containing:
    - prompt_tokens: mean input token count
    - output_tokens: mean output token count

    Cached for 5 minutes.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/workload-profile/{use_case}",
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            return cast(dict[Any, Any] | None, data.get("workload_profile"))
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch workload profile for {use_case}: {e}")
        return None


@st.cache_data(ttl=3600)
def fetch_priority_weights() -> dict | None:
    """Fetch priority weights configuration from the backend API.

    Returns dict with priority_weights mapping and defaults.
    Cached for 1 hour (config rarely changes).
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/priority-weights",
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            return cast(dict[Any, Any] | None, data.get("priority_weights"))
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch priority weights: {e}")
        return None


@st.cache_data(ttl=3600)
def fetch_use_cases() -> dict[str, dict]:
    """Fetch use case definitions from the API, keyed by use case ID."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/use-cases", timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        result: dict[str, dict] = data.get("use_cases", {})
        return result
    except Exception as e:
        logger.warning(f"Failed to fetch use cases: {e}")
        return {}


@st.cache_data(ttl=3600)
def fetch_catalog_model_ids() -> list[str]:
    """Fetch model IDs from the model catalog API.

    Returns sorted list of model_id strings (e.g., "meta-llama/Llama-3.1-8B-Instruct").
    """
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/models", timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return sorted(m["model_id"] for m in data.get("models", []) if m.get("model_id"))
    except Exception as e:
        logger.warning(f"Failed to fetch catalog models: {e}")
        return []


@st.cache_data(ttl=3600)
def fetch_gpu_types() -> dict[str, dict]:
    """Fetch all GPU types from the API, keyed by canonical gpu_type name.

    Returns a dict where each key is the short gpu_type identifier from ModelCatalog
    (e.g. "H100", "L4", "A100-40") and the value is the full GPU info dict.

    Example return value:
        {
            "H100": {"gpu_type": "H100", "memory_gb": 80, "cost_per_hour_usd": 3.0, ...},
            "L4":   {"gpu_type": "L4",   "memory_gb": 24, "cost_per_hour_usd": 0.8, ...},
        }

    The keys are the same short names used in benchmark data and recommendation results,
    not longer alias strings like "NVIDIA-H100-SXM5-80GB".
    """
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/gpu-types", timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return {g["gpu_type"]: g for g in data.get("gpu_types", [])}
    except Exception as e:
        logger.warning(f"Failed to fetch GPU types: {e}")
        return {}


def fetch_capacity_planner_model_info(model_id: str) -> dict | None:
    """Fetch model metadata from HuggingFace via the backend.

    Returns the full model-info response dict, or None on error.
    Error detail (e.g. gated model message) is surfaced via st.error in the caller.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/model-info",
            json={"model_id": model_id},
            timeout=60,
        )
        if response.status_code == 200:
            return cast(dict[Any, Any], response.json())
        elif response.status_code == 403:
            st.error(
                "This is a gated model. Set the `HF_TOKEN` environment variable "
                "on the backend to access it."
            )
        else:
            st.error(f"Failed to fetch model info: {response.text[:200]}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend API.")
    except requests.exceptions.Timeout:
        st.error("Request to fetch model info timed out.")
    except Exception as e:
        st.error(f"Error fetching model info: {e}")
    return None


def fetch_capacity_planner_calculate(
    model_id: str,
    max_model_len: int | None = None,
    batch_size: int = 1,
    gpu_memory: float | None = None,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    gpu_mem_util: float = 0.9,
    block_size: int = 16,
) -> dict | None:
    """Run capacity planning calculations via the backend.

    Returns the calculate response dict, or None on error.
    """
    payload: dict[str, Any] = {
        "model_id": model_id,
        "batch_size": batch_size,
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "gpu_mem_util": gpu_mem_util,
        "block_size": block_size,
    }
    if max_model_len is not None:
        payload["max_model_len"] = max_model_len
    if gpu_memory is not None:
        payload["gpu_memory"] = gpu_memory

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/calculate",
            json=payload,
            timeout=60,
        )
        if response.status_code == 200:
            return cast(dict[Any, Any], response.json())
        else:
            try:
                error_detail = response.json().get("detail", response.text[:200])
            except Exception:
                error_detail = response.text[:200]
            st.warning(f"Calculation error: {error_detail}")
    except requests.exceptions.ConnectionError:
        st.warning("Cannot connect to backend API.")
    except Exception as e:
        st.warning(f"Calculation failed: {e}")
    return None


def fetch_gpu_recommender_estimate(
    model_id: str,
    input_len: int,
    output_len: int,
    max_gpus: int = 1,
    max_gpus_per_type: dict[str, int] | None = None,
    gpu_list: list[str] | None = None,
    max_ttft: float | None = None,
    max_itl: float | None = None,
    max_latency: float | None = None,
    custom_gpu_costs: dict[str, float] | None = None,
) -> dict | None:
    """Run GPU performance estimation via the backend.

    Returns the estimate response dict, or None on error.
    """
    payload: dict[str, Any] = {
        "model_id": model_id,
        "input_len": input_len,
        "output_len": output_len,
        "max_gpus": max_gpus,
    }
    if max_gpus_per_type:
        payload["max_gpus_per_type"] = max_gpus_per_type
    if gpu_list:
        payload["gpu_list"] = gpu_list
    if max_ttft is not None:
        payload["max_ttft"] = max_ttft
    if max_itl is not None:
        payload["max_itl"] = max_itl
    if max_latency is not None:
        payload["max_latency"] = max_latency
    if custom_gpu_costs:
        payload["custom_gpu_costs"] = custom_gpu_costs

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/estimate",
            json=payload,
            timeout=120,  # GPU estimation can be slow
        )
        if response.status_code == 200:
            return cast(dict[Any, Any], response.json())
        else:
            try:
                error_detail = response.json().get("detail", response.text[:200])
            except Exception:
                error_detail = response.text[:200]
            st.error(f"Estimation error: {error_detail}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend API.")
    except requests.exceptions.Timeout:
        st.error("GPU estimation timed out.")
    except Exception as e:
        st.error(f"Estimation error: {e}")
    return None


def generate_specification(intent: dict) -> dict | None:
    """Generate deployment specification from intent.

    Args:
        intent: DeploymentIntent dict with use_case, user_count, priorities, etc.

    Returns:
        DeploymentSpecification dict, or None on error
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/generate-specification",
            json=intent,
            timeout=30,
        )
        response.raise_for_status()
        return cast(dict[Any, Any], response.json())
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to generate specification: {e}")
        return None


def fetch_ranked_recommendations(
    use_case: str,
    user_count: int,
    prompt_tokens: int,
    output_tokens: int,
    expected_qps: float,
    ttft_target_ms: int,
    itl_target_ms: int,
    e2e_target_ms: int,
    weights: dict[Any, Any] | None = None,
    include_near_miss: bool = False,
    percentile: str = "p95",
    preferred_gpu_types: list[str] | None = None,
    preferred_models: list[str] | None = None,
    enable_estimated: bool = True,
    quality_priority: str = "medium",
    cost_priority: str = "medium",
    latency_priority: str = "medium",
) -> dict | None:
    """Fetch ranked recommendations from the backend API.

    Args:
        use_case: Use case identifier (e.g., "chatbot_conversational")
        user_count: Number of concurrent users
        prompt_tokens: Input prompt token count
        output_tokens: Output generation token count
        expected_qps: Queries per second
        ttft_target_ms: TTFT SLO target
        itl_target_ms: ITL SLO target
        e2e_target_ms: E2E SLO target
        weights: Optional dict with quality, price, latency weights (0-10)
        include_near_miss: Whether to include near-SLO configurations
        percentile: Which percentile to use for SLO comparison (mean, p90, p95, p99)
        preferred_gpu_types: Optional list of GPU types to filter by (empty = any GPU)
        preferred_models: Optional list of HuggingFace model IDs to include via estimation
        enable_estimated: Whether to run roofline estimation for missing benchmarks
        quality_priority: Quality priority level (low/medium/high)
        cost_priority: Cost priority level (low/medium/high)
        latency_priority: Latency priority level (low/medium/high)

    Returns:
        RankedRecommendationsResponse as dict, or None on error
    """
    # Build DeploymentSpecification payload
    specification = {
        "intent": {
            "use_case": use_case,
            "user_count": user_count,
            "quality_priority": quality_priority,
            "cost_priority": cost_priority,
            "latency_priority": latency_priority,
            "preferred_gpu_types": preferred_gpu_types or [],
            "preferred_models": preferred_models or [],
        },
        "slo_targets": {
            "ttft_target_ms": ttft_target_ms,
            "itl_target_ms": itl_target_ms,
            "e2e_target_ms": e2e_target_ms,
            "percentile": percentile,
        },
        "workload_profile": {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "expected_qps": expected_qps,
        },
        "priorities": {
            "quality": {
                "priority": quality_priority,
                "weight": weights.get("quality", 4) if weights else 4,
            },
            "cost": {
                "priority": cost_priority,
                "weight": weights.get("price", 4) if weights else 4,
            },
            "latency": {
                "priority": latency_priority,
                "weight": weights.get("latency", 1) if weights else 1,
            },
        },
    }

    payload = {
        "specification": specification,
        "enable_estimated": enable_estimated,
        "min_quality": None,
        "max_cost": None,
        "include_near_miss": include_near_miss,
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/generate-recommendations",
            json=payload,
            timeout=90,  # Backend estimation timeout (60s default) + buffer
        )
        response.raise_for_status()
        return cast(dict[Any, Any], response.json())
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch ranked recommendations: {e}")
        return None


def check_llm_available() -> bool:
    """Check whether the backend's LLM provider is reachable."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/llm-status", timeout=5)
        if response.status_code == 200:
            return bool(response.json().get("available", False))
    except Exception:
        pass
    return False


def extract_business_context(user_input: str) -> dict | None:
    """Extract business context using the backend LLM extraction API.

    Returns None on failure — caller is responsible for showing errors.
    """
    try:
        logger.info(f"Calling LLM extraction API at {API_BASE_URL}/api/v1/extract-intent")
        response = requests.post(
            f"{API_BASE_URL}/api/v1/extract-intent",
            json={"text": user_input},
            timeout=300,
        )
        if response.status_code == 200:
            result = response.json()
            # Map preferred_gpu_types (list) to hardware for UI compatibility
            if "preferred_gpu_types" in result:
                gpu_list = result["preferred_gpu_types"]
                gpu_names = [
                    g["gpu_type"] if isinstance(g, dict) and "gpu_type" in g else str(g)
                    for g in gpu_list
                ]
                result["hardware"] = ", ".join(gpu_names) if gpu_names else None
            logger.info(f"LLM extraction successful: {result.get('use_case')}")
            return cast(dict[Any, Any], result)
        else:
            logger.warning(
                f"LLM extraction API returned status {response.status_code}: {response.text[:200]}"
            )
    except requests.exceptions.ConnectionError as e:
        logger.warning(f"Backend not reachable: {e}")
    except requests.exceptions.Timeout as e:
        logger.warning(f"LLM extraction timed out: {e}")
    except Exception as e:
        logger.warning(f"LLM extraction failed: {type(e).__name__}: {e}")

    return None


def deploy_and_generate_yaml(recommendation: dict, stack: str = "vllm") -> dict | None:
    """Generate deployment YAML from a recommendation's configuration.

    Returns dict with deployment_id, yaml_contents, and success status, or None on error.
    """
    try:
        configuration = recommendation.get("configuration")
        if not configuration:
            return {"success": False, "message": "Recommendation missing configuration field"}
        response = requests.post(
            f"{API_BASE_URL}/api/v1/generate-deployment",
            json={"configuration": configuration, "namespace": "default", "stack": stack},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        # New endpoint returns DeploymentBundle with 'files' instead of 'yaml_contents'
        if result.get("deployment_id"):
            return {
                "success": True,
                "deployment_id": result.get("deployment_id"),
                "yaml_contents": result.get(
                    "files", {}
                ),  # Map 'files' to 'yaml_contents' for UI compatibility
            }
        else:
            return {"success": False, "message": result.get("message", "Unknown error")}
    except Exception as e:
        return {"success": False, "message": str(e)}


# =============================================================================
# CLUSTER & DEPLOYMENT MANAGEMENT
# =============================================================================


def check_cluster_status() -> dict:
    """Check if Kubernetes cluster is accessible.

    Returns dict with 'accessible' bool and optional 'inference_services' list.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/cluster-status",
            timeout=5,
        )
        if response.status_code == 200:
            return cast(dict[Any, Any], response.json())
        return {"accessible": False}
    except Exception:
        return {"accessible": False}


def load_all_deployments() -> list | None:
    """Load all InferenceServices from the cluster.

    Returns list of deployment dicts, or None if cluster is not accessible.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/deployments",
            timeout=DEFAULT_TIMEOUT,
        )
        if response.status_code == 200:
            data = response.json()
            return cast(list[Any], data.get("deployments", []))
        elif response.status_code == 503:
            return None
        else:
            logger.error(f"Failed to load deployments: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        return None
    except Exception:
        logger.exception("Error loading deployments")
        return None


def deploy_to_cluster(recommendation: dict, namespace: str = "default") -> dict:
    """Deploy a recommendation to the Kubernetes cluster.

    Two-step flow: generate-deployment → deploy-bundle-to-cluster.
    Returns dict with deployment_id, yaml_contents, and success status.
    """
    try:
        config = recommendation.get("configuration")
        if not config:
            return {"success": False, "message": "Recommendation missing configuration field"}

        stack = recommendation.get("_stack", "vllm")

        gen_response = requests.post(
            f"{API_BASE_URL}/api/v1/generate-deployment",
            json={"configuration": config, "namespace": namespace, "stack": stack},
            timeout=30,
        )
        if gen_response.status_code != 200:
            return {
                "success": False,
                "message": f"Failed to generate deployment: {gen_response.text}",
            }
        bundle = gen_response.json()

        deploy_response = requests.post(
            f"{API_BASE_URL}/api/v1/deploy-bundle-to-cluster",
            json={"bundle": bundle},
            timeout=60,
        )
        if deploy_response.status_code == 200:
            result = deploy_response.json()
            result["yaml_contents"] = bundle.get("files", {})
            result["deployment_id"] = bundle.get("deployment_id")
            return cast(dict[Any, Any], result)
        elif deploy_response.status_code == 503:
            return {"success": False, "message": "Kubernetes cluster not accessible"}
        else:
            return {"success": False, "message": deploy_response.text}
    except requests.exceptions.ConnectionError:
        return {"success": False, "message": "Cannot connect to backend API"}
    except Exception as e:
        return {"success": False, "message": str(e)}


def delete_deployment(deployment_id: str) -> dict:
    """Delete a deployment from the cluster.

    Returns dict with 'success' bool and 'message'.
    """
    try:
        response = requests.delete(
            f"{API_BASE_URL}/api/v1/deployments/{deployment_id}",
            timeout=30,
        )
        if response.status_code == 200:
            return cast(dict[Any, Any], response.json())
        else:
            return {"success": False, "message": response.text}
    except Exception as e:
        return {"success": False, "message": str(e)}


def get_k8s_status(deployment_id: str) -> dict | None:
    """Get real Kubernetes status for a deployment.

    Returns dict with inferenceservice status, pods, etc., or None on error.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/deployments/{deployment_id}/k8s-status",
            timeout=DEFAULT_TIMEOUT,
        )
        if response.status_code == 200:
            return cast(dict[Any, Any], response.json())
        return None
    except Exception as e:
        logger.error(f"Failed to get K8s status for {deployment_id}: {e}")
        return None


# =============================================================================
# DEPLOYMENT MODE
# =============================================================================


def fetch_deployment_mode() -> str | None:
    """Fetch the current deployment mode from the backend.

    Returns 'production' or 'simulator', or None on error.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/deployment-mode", timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        return cast(str | None, response.json().get("mode"))
    except Exception as e:
        logger.error(f"Failed to fetch deployment mode: {e}")
        return None


def update_deployment_mode(mode: str) -> dict | None:
    """Set the deployment mode ('production' or 'simulator').

    Returns response dict with 'mode' key, or None on error.
    """
    try:
        response = requests.put(
            f"{API_BASE_URL}/api/v1/deployment-mode",
            json={"mode": mode},
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        return cast(dict[Any, Any], response.json())
    except Exception as e:
        logger.error(f"Failed to set deployment mode: {e}")
        return None


# =============================================================================
# DATABASE MANAGEMENT
# =============================================================================


def fetch_db_status() -> dict | None:
    """Fetch current benchmark database statistics.

    Returns dict with total_benchmarks, num_models, num_hardware_types, etc.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/db/status",
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        return cast(dict[Any, Any], response.json())
    except Exception as e:
        logger.error(f"Failed to fetch DB status: {e}")
        return None


def is_db_admin_required() -> bool:
    """Check whether the backend requires a password for DB admin operations."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/db/admin-required",
            timeout=5,
        )
        response.raise_for_status()
        return bool(response.json().get("required", False))
    except Exception:
        return False


def verify_db_admin_password(password: str) -> bool:
    """Verify the admin password against the backend."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/db/verify-admin",
            headers={"X-Admin-Password": password},
            timeout=5,
        )
        return response.status_code == 200
    except Exception:
        return False


def upload_benchmarks(
    file_bytes: bytes,
    filename: str,
    password: str | None = None,
) -> dict | None:
    """Upload a benchmark JSON file to the backend.

    Args:
        file_bytes: Raw bytes of the JSON file
        filename: Original filename
        password: Optional admin password

    Returns:
        Response dict with success status and stats, or None on error.
    """
    headers: dict[str, str] = {}
    if password:
        headers["X-Admin-Password"] = password
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/db/upload-benchmarks",
            files={"file": (filename, file_bytes, "application/json")},
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        return cast(dict[Any, Any], response.json())
    except requests.exceptions.HTTPError as e:
        detail = ""
        with contextlib.suppress(Exception):
            detail = e.response.json().get("detail", "")
        logger.error(f"Upload failed: {detail or e}")
        return {"success": False, "message": detail or str(e)}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return {"success": False, "message": str(e)}


def reset_database(password: str | None = None) -> dict | None:
    """Reset the benchmark database (removes all benchmark data).

    Args:
        password: Optional admin password

    Returns:
        Response dict with success status, or None on error.
    """
    headers: dict[str, str] = {}
    if password:
        headers["X-Admin-Password"] = password
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/db/reset",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        return cast(dict[Any, Any], response.json())
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        return {"success": False, "message": str(e)}
