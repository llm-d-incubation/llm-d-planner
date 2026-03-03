"""Model metadata source from RHOAI Model Catalog."""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING

from neuralnav.knowledge_base.model_catalog import ModelInfo

if TYPE_CHECKING:
    from neuralnav.knowledge_base.model_catalog import ModelCatalog
    from neuralnav.knowledge_base.model_catalog_client import ModelCatalogClient

logger = logging.getLogger(__name__)

_CACHE_TTL = 3600

# Map Model Catalog tasks to NeuralNav supported_tasks
_TASK_MAP: dict[str, list[str]] = {
    "text-to-text": [
        "chatbot_conversational",
        "summarization_short",
        "content_generation",
        "translation",
    ],
    "text-generation": [
        "chatbot_conversational",
        "code_completion",
        "content_generation",
    ],
}

_SIZE_RE = re.compile(r"([\d.]+)\s*[Bb]\b")

_KNOWN_FAMILIES = [
    "granite",
    "llama",
    "mistral",
    "qwen",
    "gemma",
    "phi",
    "deepseek",
    "mixtral",
    "kimi",
    "nemotron",
]


def _parse_size(size_str: str) -> str:
    """Parse '8B params' or '70B params' into '8B' or '70B'."""
    match = _SIZE_RE.search(size_str)
    return match.group(0).upper().replace(" ", "") if match else size_str


def _extract_family(name: str) -> str:
    """Extract model family from name.

    E.g. 'RedHatAI/granite-3.1-8b-instruct' -> 'granite'.
    """
    base = name.split("/")[-1].lower()
    for family in _KNOWN_FAMILIES:
        if family in base:
            return family
    return base.split("-")[0]


def _catalog_model_to_model_info(model: dict) -> ModelInfo:
    """Map a Model Catalog model dict to a ModelInfo instance."""
    name = model.get("name", "")
    props = model.get("customProperties", {})

    size_str = props.get("size", {}).get("string_value", "")
    validated_on = props.get("validated_on", {}).get("string_value", "")
    has_validated = "validated" in props or bool(validated_on)

    # Map tasks
    catalog_tasks = model.get("tasks", [])
    supported_tasks: list[str] = []
    for t in catalog_tasks:
        supported_tasks.extend(_TASK_MAP.get(t, [t]))

    # Estimate min GPU memory from size
    size_parsed = _parse_size(size_str) if size_str else ""
    try:
        param_billions = float(size_parsed.rstrip("Bb"))
    except ValueError:
        param_billions = 7.0  # default
    # Rough heuristic: ~2 bytes per param for FP16, ~1 byte for FP8/INT8
    min_gpu_gb = max(8, int(param_billions * 1.5))

    data = {
        "model_id": name,
        "name": name.split("/")[-1],
        "provider": model.get("provider", "Unknown"),
        "family": _extract_family(name),
        "size_parameters": _parse_size(size_str) if size_str else f"{param_billions}B",
        "context_length": 128000,  # Default; not available in Model Catalog
        "supported_tasks": list(set(supported_tasks)),
        "domain_specialization": [],
        "license": model.get("license", "Unknown"),
        "license_type": (
            "permissive" if "apache" in model.get("license", "").lower() else "restricted"
        ),
        "min_gpu_memory_gb": min_gpu_gb,
        "recommended_for": supported_tasks,
        "approval_status": "approved" if has_validated else "pending",
    }
    return ModelInfo(data)


class ModelCatalogModelSource:
    """Model metadata from RHOAI Model Catalog.

    Drop-in for ModelCatalog when running with RHOAI.
    """

    def __init__(self, client: ModelCatalogClient) -> None:
        self._client = client
        self._models: dict[str, ModelInfo] = {}
        self._loaded_at: float = 0
        self._local_catalog: ModelCatalog | None = None

    def preload(self) -> None:
        """Eagerly load model cache (call during app startup)."""
        self._load_all()

    def _ensure_loaded(self) -> None:
        if self._models and (time.time() - self._loaded_at) < _CACHE_TTL:
            return
        self._load_all()

    def _load_all(self) -> None:
        models: dict[str, ModelInfo] = {}
        for raw in self._client.list_models():
            info = _catalog_model_to_model_info(raw)
            models[info.model_id] = info
        self._models = models
        self._loaded_at = time.time()
        logger.info("Loaded %d models from Model Catalog", len(models))

    def get_model(self, model_id: str) -> ModelInfo | None:
        """Get model by ID."""
        self._ensure_loaded()
        return self._models.get(model_id)

    def get_all_models(self) -> list[ModelInfo]:
        """Get all approved models."""
        self._ensure_loaded()
        return [m for m in self._models.values() if m.approval_status == "approved"]

    def find_models_for_use_case(self, use_case: str) -> list[ModelInfo]:
        """Find approved models recommended for a use case."""
        self._ensure_loaded()
        return [
            m
            for m in self._models.values()
            if use_case in m.recommended_for and m.approval_status == "approved"
        ]

    def find_models_by_task(self, task: str) -> list[ModelInfo]:
        """Find approved models supporting a specific task."""
        self._ensure_loaded()
        return [
            m
            for m in self._models.values()
            if task in m.supported_tasks and m.approval_status == "approved"
        ]

    def _get_local_catalog(self) -> ModelCatalog:
        """Lazily create and cache local ModelCatalog for GPU pricing."""
        if self._local_catalog is None:
            from neuralnav.knowledge_base.model_catalog import (
                ModelCatalog as _ModelCatalog,
            )

            self._local_catalog = _ModelCatalog()
        return self._local_catalog

    def get_gpu_type(self, gpu_type: str):
        """Delegate to local JSON catalog for GPU pricing."""
        return self._get_local_catalog().get_gpu_type(gpu_type)

    def calculate_gpu_cost(
        self,
        gpu_type: str,
        gpu_count: int,
        hours_per_month: float = 730,
        provider: str | None = None,
    ) -> float | None:
        """Delegate to local JSON catalog for GPU pricing."""
        return self._get_local_catalog().calculate_gpu_cost(gpu_type, gpu_count, hours_per_month, provider)

    def get_all_gpu_types(self):
        """Delegate to local JSON catalog."""
        return self._get_local_catalog().get_all_gpu_types()
