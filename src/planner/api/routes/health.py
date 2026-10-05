"""Health check endpoint."""

import logging

from fastapi import APIRouter, Request, Response, status

from planner.api.dependencies import get_benchmark_repo
from planner.llm.factory import create_llm_client

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health")
def health_check(request: Request, response: Response) -> dict[str, str]:
    """Check that the active benchmark database is usable."""
    try:
        get_benchmark_repo(request).check_health()
    except Exception:
        logger.exception("Benchmark database health check failed")
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unhealthy", "service": "planner"}
    return {"status": "healthy", "service": "planner"}


@router.get("/api/v1/llm-status")
async def llm_status():
    """Check whether the configured LLM provider is reachable."""
    try:
        client = create_llm_client()
        available = client.is_available()
    except Exception as e:
        logger.warning("LLM status check failed: %s", e)
        available = False
    return {"available": available}
