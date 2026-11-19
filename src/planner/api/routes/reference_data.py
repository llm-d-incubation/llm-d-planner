"""Reference data endpoints (models, GPU types, benchmarks, etc.)."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from planner.api.dependencies import get_model_catalog, get_use_case_repo
from planner.data._resolver import data_path
from planner.knowledge_base.model_catalog import ModelCatalog
from planner.knowledge_base.use_cases import UseCaseRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["reference-data"])


def _get_data_path() -> Path:
    """Get the base data directory path."""
    return data_path("")


@router.get("/models")
async def list_models(model_catalog: ModelCatalog = Depends(get_model_catalog)):
    """Get list of available models."""
    try:
        models = model_catalog.get_all_models()
        return {"models": [model.to_dict() for model in models], "count": len(models)}
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.get("/gpu-types")
async def list_gpu_types(model_catalog: ModelCatalog = Depends(get_model_catalog)):
    """Get list of available GPU types."""
    try:
        gpu_types = model_catalog.get_all_gpu_types()
        return {"gpu_types": [gpu.to_dict() for gpu in gpu_types], "count": len(gpu_types)}
    except Exception as e:
        logger.error(f"Failed to list GPU types: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.get("/use-cases")
async def list_use_cases(use_case_repo: UseCaseRepository = Depends(get_use_case_repo)):
    """Get list of supported use cases with configuration."""
    try:
        use_cases = use_case_repo.get_all_use_cases()
        return {
            "use_cases": {uc_id: uc.to_dict() for uc_id, uc in use_cases.items()},
            "count": len(use_cases),
        }
    except Exception as e:
        logger.error(f"Failed to list use cases: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.get("/priority-weights")
async def get_priority_weights():
    """Get priority to weight mapping configuration.

    Returns the priority_weights.json data for UI to use
    when setting initial weights based on priority dropdowns.
    """
    try:
        json_path = _get_data_path() / "configuration" / "priority_weights.json"

        if not json_path.exists():
            logger.error(f"Priority weights config not found at: {json_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Priority weights configuration not found",
            )

        with open(json_path) as f:
            data = json.load(f)

        return {"success": True, "priority_weights": data}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load priority weights: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
