"""FastAPI application factory for Planner API."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from planner.api.routes import (
    configuration_router,
    database_router,
    health_router,
    intent_router,
    recommendation_router,
    reference_data_router,
    specification_router,
)

# Configure logging
debug_mode = os.getenv("PLANNER_DEBUG", "false").lower() == "true"
log_level = logging.DEBUG if debug_mode else logging.INFO
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all singletons on app.state during startup."""
    from planner.api.dependencies import init_app_state

    logger.info("Initializing app state...")
    try:
        await asyncio.to_thread(init_app_state, app)
    except Exception:
        logger.exception("App state initialization failed during startup")
        raise
    # Create asyncio.Lock in the event loop thread (not in the worker thread
    # where init_app_state runs) to avoid cross-loop binding issues.
    app.state.cluster_manager_lock = asyncio.Lock()
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Planner API",
        description="API for LLM deployment recommendations",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify actual origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include all routers
    app.include_router(health_router)
    app.include_router(intent_router)
    app.include_router(specification_router)
    app.include_router(recommendation_router)
    app.include_router(configuration_router)
    app.include_router(reference_data_router)
    app.include_router(database_router)

    logger.info(f"Planner API starting with log level: {logging.getLevelName(log_level)}")

    return app


# Create the app instance for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
