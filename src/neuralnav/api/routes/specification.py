"""Specification-related endpoints (SLO defaults, workload profiles)."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["specification"])


def _round_to_nearest(value: float, nearest: int = 5) -> int:
    """Round a value to the nearest multiple of `nearest`."""
    return int(round(value / nearest) * nearest)


def _calculate_percentile_value(min_val: int, max_val: int, percentile: float = 0.75) -> int:
    """Calculate value at given percentile between min and max, rounded to nearest 5."""
    value = min_val + (max_val - min_val) * percentile
    return _round_to_nearest(value, 5)


def _get_slo_workload_path() -> Path:
    """Get path to the SLO workload config file."""
    return (
        Path(__file__).parent.parent.parent.parent.parent
        / "data"
        / "configuration"
        / "usecase_slo_workload.json"
    )


@router.get("/slo-defaults/{use_case}")
async def get_slo_defaults(use_case: str):
    """Get default SLO values for a use case.

    Returns SLO targets at the 75th percentile between min and max,
    rounded to the nearest 5.
    """
    try:
        json_path = _get_slo_workload_path()

        if not json_path.exists():
            logger.error(f"SLO workload config not found at: {json_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="SLO workload configuration not found"
            )

        with open(json_path) as f:
            data = json.load(f)

        use_case_data = data.get("use_case_slo_workload", {}).get(use_case)
        if not use_case_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Use case '{use_case}' not found"
            )

        slo_targets = use_case_data.get("slo_targets", {})

        # Get SLO ranges - fail if data is missing
        ttft = slo_targets["ttft_ms"]
        itl = slo_targets["itl_ms"]
        e2e = slo_targets["e2e_ms"]

        defaults = {
            "use_case": use_case,
            "description": use_case_data.get("description", ""),
            "ttft_ms": {
                "min": ttft["min"],
                "max": ttft["max"],
                "default": _calculate_percentile_value(ttft["min"], ttft["max"], 0.75),
            },
            "itl_ms": {
                "min": itl["min"],
                "max": itl["max"],
                "default": _calculate_percentile_value(itl["min"], itl["max"], 0.75),
            },
            "e2e_ms": {
                "min": e2e["min"],
                "max": e2e["max"],
                "default": _calculate_percentile_value(e2e["min"], e2e["max"], 0.75),
            },
        }

        return {"success": True, "slo_defaults": defaults}

    except HTTPException:
        raise
    except KeyError as e:
        logger.error(f"Missing SLO data for {use_case}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Missing SLO data: {e}"
        ) from e
    except Exception as e:
        logger.error(f"Failed to get SLO defaults for {use_case}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.get("/workload-profile/{use_case}")
async def get_workload_profile(use_case: str):
    """Get workload profile for a use case.

    Returns the token configuration and peak multiplier used for
    capacity planning and recommendation generation.
    """
    try:
        json_path = _get_slo_workload_path()

        if not json_path.exists():
            logger.error(f"SLO workload config not found at: {json_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="SLO workload configuration not found"
            )

        with open(json_path) as f:
            data = json.load(f)

        use_case_data = data.get("use_case_slo_workload", {}).get(use_case)
        if not use_case_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Use case '{use_case}' not found"
            )

        workload = use_case_data.get("workload", {})

        # Require all workload fields - fail if missing
        prompt_tokens = workload["prompt_tokens"]
        output_tokens = workload["output_tokens"]
        peak_multiplier = workload["peak_multiplier"]
        distribution = workload["distribution"]
        active_fraction = workload["active_fraction"]
        requests_per_active_user_per_min = workload["requests_per_active_user_per_min"]

        return {
            "success": True,
            "use_case": use_case,
            "description": use_case_data.get("description", ""),
            "workload_profile": {
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "peak_multiplier": peak_multiplier,
                "distribution": distribution,
                "active_fraction": active_fraction,
                "requests_per_active_user_per_min": requests_per_active_user_per_min,
            },
        }

    except HTTPException:
        raise
    except KeyError as e:
        logger.error(f"Missing workload data for {use_case}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Missing workload data: {e}"
        ) from e
    except Exception as e:
        logger.error(f"Failed to get workload profile for {use_case}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.get("/expected-rps/{use_case}")
async def get_expected_rps(use_case: str, user_count: int = 1000):
    """Calculate expected RPS for a use case based on workload patterns.

    Uses research-backed workload distribution parameters:
    - active_fraction: percentage of users active at any time
    - requests_per_active_user_per_min: request rate per active user

    Formula: expected_rps = (user_count * active_fraction * requests_per_min) / 60
    """
    try:
        json_path = _get_slo_workload_path()

        if not json_path.exists():
            logger.error(f"SLO workload config not found at: {json_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="SLO workload configuration not found"
            )

        with open(json_path) as f:
            data = json.load(f)

        use_case_data = data.get("use_case_slo_workload", {}).get(use_case)
        if not use_case_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Use case '{use_case}' not found"
            )

        workload = use_case_data.get("workload", {})

        # Get workload parameters - fail if missing
        active_fraction = workload["active_fraction"]
        requests_per_min = workload["requests_per_active_user_per_min"]
        peak_multiplier = workload.get("peak_multiplier", 2.0)
        distribution = workload.get("distribution", "poisson")

        # Calculate expected RPS using research-based formula
        expected_concurrent = int(user_count * active_fraction)
        expected_rps = (expected_concurrent * requests_per_min) / 60
        expected_rps = max(1, round(expected_rps, 2))  # Minimum 1 RPS, round to 2 decimals

        # Calculate peak RPS for capacity planning
        peak_rps = expected_rps * peak_multiplier

        return {
            "success": True,
            "use_case": use_case,
            "user_count": user_count,
            "workload_params": {
                "active_fraction": active_fraction,
                "requests_per_active_user_per_min": requests_per_min,
                "peak_multiplier": peak_multiplier,
                "distribution": distribution,
            },
            "expected_rps": expected_rps,
            "expected_concurrent_users": expected_concurrent,
            "peak_rps": round(peak_rps, 2),
        }

    except HTTPException:
        raise
    except KeyError as e:
        logger.error(f"Missing workload data for {use_case}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Missing workload data: {e}"
        ) from e
    except Exception as e:
        logger.error(f"Failed to calculate expected RPS for {use_case}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
