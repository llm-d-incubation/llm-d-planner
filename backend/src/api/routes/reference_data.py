"""Reference data endpoints (models, GPU types, benchmarks, etc.)."""

import csv
import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.api.dependencies import get_model_catalog, get_slo_repo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["reference-data"])


def _get_data_path() -> Path:
    """Get the base data directory path."""
    return Path(__file__).parent.parent.parent.parent.parent / "data"


@router.get("/models")
async def list_models():
    """Get list of available models."""
    try:
        model_catalog = get_model_catalog()
        models = model_catalog.get_all_models()
        return {"models": [model.to_dict() for model in models], "count": len(models)}
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/gpu-types")
async def list_gpu_types():
    """Get list of available GPU types."""
    try:
        model_catalog = get_model_catalog()
        gpu_types = model_catalog.get_all_gpu_types()
        return {"gpu_types": [gpu.to_dict() for gpu in gpu_types], "count": len(gpu_types)}
    except Exception as e:
        logger.error(f"Failed to list GPU types: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/use-cases")
async def list_use_cases():
    """Get list of supported use cases with SLO templates."""
    try:
        slo_repo = get_slo_repo()
        templates = slo_repo.get_all_templates()
        return {
            "use_cases": {use_case: template.to_dict() for use_case, template in templates.items()},
            "count": len(templates),
        }
    except Exception as e:
        logger.error(f"Failed to list use cases: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/benchmarks")
async def get_benchmarks():
    """Get all 206 models benchmark data from opensource_all_benchmarks.csv."""
    try:
        csv_path = _get_data_path() / "benchmarks" / "accuracy" / "opensource_all_benchmarks.csv"

        if not csv_path.exists():
            logger.error(f"Benchmark CSV not found at: {csv_path}")
            raise HTTPException(status_code=404, detail="Benchmark data file not found")

        # Read CSV using built-in csv module
        records = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter out rows with empty/missing Model Name
                if row.get("Model Name") and row["Model Name"].strip():
                    records.append(row)

        logger.info(f"Loaded {len(records)} benchmark records from CSV")

        return {"success": True, "count": len(records), "benchmarks": records}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load benchmarks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load benchmarks: {str(e)}") from e


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
            raise HTTPException(status_code=404, detail="Priority weights configuration not found")

        with open(json_path) as f:
            data = json.load(f)

        return {"success": True, "priority_weights": data}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load priority weights: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/weighted-scores/{use_case}")
async def get_weighted_scores(use_case: str):
    """Get use-case-specific weighted scores from CSV."""
    try:
        # Map use case to CSV filename
        use_case_to_file = {
            "chatbot_conversational": "opensource_chatbot_conversational.csv",
            "code_completion": "opensource_code_completion.csv",
            "code_generation_detailed": "opensource_code_generation_detailed.csv",
            "document_analysis_rag": "opensource_document_analysis_rag.csv",
            "summarization_short": "opensource_summarization_short.csv",
            "long_document_summarization": "opensource_long_document_summarization.csv",
            "translation": "opensource_translation.csv",
            "content_generation": "opensource_content_generation.csv",
            "research_legal_analysis": "opensource_research_legal_analysis.csv",
        }

        filename = use_case_to_file.get(use_case)
        if not filename:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid use case: {use_case}. Valid options: {list(use_case_to_file.keys())}",
            )

        csv_path = _get_data_path() / "benchmarks" / "accuracy" / "weighted_scores" / filename

        if not csv_path.exists():
            logger.error(f"Weighted scores CSV not found at: {csv_path}")
            raise HTTPException(
                status_code=404, detail=f"Weighted scores file not found for use case: {use_case}"
            )

        # Read CSV using built-in csv module
        records = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)

        logger.info(f"Loaded {len(records)} weighted score records for use case: {use_case}")

        return {"success": True, "use_case": use_case, "count": len(records), "scores": records}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load weighted scores: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to load weighted scores: {str(e)}"
        ) from e
