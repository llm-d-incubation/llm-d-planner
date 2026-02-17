"""FastAPI application factory for NeuralNav API."""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import (
    configuration_router,
    database_router,
    health_router,
    intent_router,
    recommendation_router,
    reference_data_router,
    specification_router,
)

# Configure logging
debug_mode = os.getenv("NEURALNAV_DEBUG", "false").lower() == "true"
log_level = logging.DEBUG if debug_mode else logging.INFO
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="NeuralNav API",
        description="API for LLM deployment recommendations",
        version="0.1.0",
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

    logger.info(f"NeuralNav API starting with log level: {logging.getLevelName(log_level)}")

    return app


# Create the app instance for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
