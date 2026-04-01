"""Intent extraction endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from planner.api.dependencies import get_workflow
from planner.intent_extraction import IntentExtractor
from planner.orchestration.workflow import RecommendationWorkflow

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["intent"])


class ExtractRequest(BaseModel):
    """Request for intent extraction from natural language."""

    text: str


@router.post("/extract")
async def extract_intent(
    request: ExtractRequest,
    workflow: RecommendationWorkflow = Depends(get_workflow),
):
    """Extract business context from natural language using LLM.

    Takes a user's natural language description of their deployment needs
    and extracts structured intent using Ollama (Qwen 2.5 7B).

    Args:
        request: ExtractRequest with 'text' field containing user input

    Returns:
        Structured intent with use_case, user_count, priority, etc.
    """
    logger.info("=" * 60)
    logger.info("EXTRACT INTENT REQUEST")
    logger.info("=" * 60)
    logger.info(f"  Input text: {request.text[:200]}{'...' if len(request.text) > 200 else ''}")

    try:
        # Create intent extractor (uses workflow's LLM client)
        intent_extractor = IntentExtractor(workflow.llm_client)

        # Extract intent from natural language
        intent = intent_extractor.extract_intent(request.text)

        # Infer any missing fields based on use case
        intent = intent_extractor.infer_missing_fields(intent)

        logger.info(f"  Extracted use_case: {intent.use_case}")
        logger.info(f"  Extracted user_count: {intent.user_count}")
        logger.info(f"  Extracted latency_priority: {intent.latency_priority}")
        logger.info("=" * 60)

        # Return as dict for JSON serialization
        result = intent.model_dump()
        result["priority"] = intent.latency_priority  # UI compatibility
        return result

    except ValueError as e:
        logger.error(f"Intent extraction failed: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Unexpected error during intent extraction: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
