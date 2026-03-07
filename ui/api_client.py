"""Backend API client for the NeuralNav UI.

All HTTP communication with the backend lives here.
"""

import contextlib
import logging
import os

import pandas as pd
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


@st.cache_data
def load_206_models() -> pd.DataFrame:
    """Load all 206 models from backend API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/benchmarks",
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("success") and data.get("benchmarks"):
            df = pd.DataFrame(data["benchmarks"])
            if "Model Name" in df.columns:
                df = df.dropna(subset=["Model Name"])
                df = df[df["Model Name"].str.strip() != ""]
            return df
        else:
            logger.warning("No benchmark data returned from API")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to load benchmarks from API: {e}")
        return pd.DataFrame()


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
            return data.get("slo_defaults")
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
            return data
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch expected RPS for {use_case}: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_workload_profile(use_case: str) -> dict | None:
    """Fetch workload profile for a use case from the backend API.

    Returns dict with workload_profile containing:
    - prompt_tokens: average input token count
    - output_tokens: average output token count
    - peak_multiplier: peak traffic multiplier
    - distribution: workload distribution type

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
            return data.get("workload_profile")
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
            return data.get("priority_weights")
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch priority weights: {e}")
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
    weights: dict = None,
    include_near_miss: bool = False,
    percentile: str = "p95",
    preferred_gpu_types: list[str] = None,
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
        weights: Optional dict with accuracy, price, latency, complexity weights (0-10)
        include_near_miss: Whether to include near-SLO configurations
        percentile: Which percentile to use for SLO comparison (mean, p90, p95, p99)
        preferred_gpu_types: Optional list of GPU types to filter by (empty = any GPU)

    Returns:
        RankedRecommendationsResponse as dict, or None on error
    """
    # Build request payload
    # min_accuracy=35 filters out models with 30% fallback (no AA data)
    payload = {
        "use_case": use_case,
        "user_count": user_count,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "expected_qps": expected_qps,
        "ttft_target_ms": ttft_target_ms,
        "itl_target_ms": itl_target_ms,
        "e2e_target_ms": e2e_target_ms,
        "percentile": percentile,
        "include_near_miss": include_near_miss,
        "min_accuracy": 35,
        "preferred_gpu_types": preferred_gpu_types or [],
    }

    if weights:
        payload["weights"] = weights

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/ranked-recommend-from-spec",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch ranked recommendations: {e}")
        return None


def extract_business_context(user_input: str) -> dict | None:
    """Extract business context using the backend LLM extraction API.

    Returns None on failure — caller is responsible for showing errors.
    """
    try:
        logger.info(f"Calling LLM extraction API at {API_BASE_URL}/api/v1/extract")
        response = requests.post(
            f"{API_BASE_URL}/api/v1/extract",
            json={"text": user_input},
            timeout=300,
        )
        if response.status_code == 200:
            result = response.json()
            # Map preferred_gpu_types (list) to hardware for UI compatibility
            if "preferred_gpu_types" in result:
                gpu_list = result["preferred_gpu_types"]
                result["hardware"] = ", ".join(gpu_list) if gpu_list else None
            logger.info(f"LLM extraction successful: {result.get('use_case')}")
            return result
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


def deploy_and_generate_yaml(recommendation: dict) -> dict | None:
    """Deploy a recommendation and fetch generated YAML files.

    Returns dict with deployment_id, files, and success status, or None on error.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/deploy",
            json={"recommendation": recommendation, "namespace": "default"},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("success"):
            deployment_id = result.get("deployment_id")
            yaml_response = requests.get(
                f"{API_BASE_URL}/api/v1/deployments/{deployment_id}/yaml",
                timeout=DEFAULT_TIMEOUT,
            )
            yaml_response.raise_for_status()
            yaml_data = yaml_response.json()
            return {
                "success": True,
                "deployment_id": deployment_id,
                "files": yaml_data.get("files", {}),
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
            return response.json()
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
            return data.get("deployments", [])
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

    Returns dict with deployment_id, files, deployment_result, and success status.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/deploy-to-cluster",
            json={"recommendation": recommendation, "namespace": namespace},
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            return {"success": False, "message": "Kubernetes cluster not accessible"}
        else:
            return {"success": False, "message": response.text}
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
            return response.json()
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
            return response.json()
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
        return response.json().get("mode")
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
        return response.json()
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
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch DB status: {e}")
        return None


def upload_benchmarks(file_bytes: bytes, filename: str) -> dict | None:
    """Upload a benchmark JSON file to the backend.

    Args:
        file_bytes: Raw bytes of the JSON file
        filename: Original filename

    Returns:
        Response dict with success status and stats, or None on error.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/db/upload-benchmarks",
            files={"file": (filename, file_bytes, "application/json")},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        detail = ""
        with contextlib.suppress(Exception):
            detail = e.response.json().get("detail", "")
        logger.error(f"Upload failed: {detail or e}")
        return {"success": False, "message": detail or str(e)}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return {"success": False, "message": str(e)}


def reset_database() -> dict | None:
    """Reset the benchmark database (removes all benchmark data).

    Returns:
        Response dict with success status, or None on error.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/db/reset",
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        return {"success": False, "message": str(e)}
