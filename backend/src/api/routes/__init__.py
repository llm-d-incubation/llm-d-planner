"""API route modules for NeuralNav."""

from .configuration import router as configuration_router
from .health import router as health_router
from .intent import router as intent_router
from .recommendation import router as recommendation_router
from .reference_data import router as reference_data_router
from .specification import router as specification_router
