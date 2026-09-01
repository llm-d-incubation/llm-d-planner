"""Data access layer for use case definitions."""

import json
import logging
from pathlib import Path

from planner.data._resolver import data_path as resolve_data_path
from planner.shared.schemas.specification import SLORange

logger = logging.getLogger(__name__)


_ACRONYMS = {"rag": "RAG", "llm": "LLM", "ai": "AI"}


def _format_display_name(use_case_id: str) -> str:
    """Fallback: title-case the key with whole-word acronym fixes."""
    return " ".join(_ACRONYMS.get(w, w.title()) for w in use_case_id.split("_"))


class UseCaseConfig:
    """Configuration for a specific use case."""

    def __init__(self, use_case_id: str, data: dict):
        self.use_case_id = use_case_id
        self.display_name = data.get("display_name") or _format_display_name(use_case_id)
        self.description = data.get("description", "")

        workload = data["workload"]
        self.prompt_tokens: int = workload["prompt_tokens"]
        self.output_tokens: int = workload["output_tokens"]
        self.active_fraction: float = workload["active_fraction"]
        self.requests_per_active_user_per_min: float = workload["requests_per_active_user_per_min"]
        self.peak_multiplier: float = workload["peak_multiplier"]

        slo = data["slo_targets"]
        self.ttft_range = SLORange(**slo["ttft_ms"])
        self.itl_range = SLORange(**slo["itl_ms"])
        self.e2e_range = SLORange(**slo["e2e_ms"])

    def to_dict(self) -> dict:
        """Convert to dictionary matching the JSON input shape."""
        return {
            "use_case_id": self.use_case_id,
            "display_name": self.display_name,
            "description": self.description,
            "workload": {
                "prompt_tokens": self.prompt_tokens,
                "output_tokens": self.output_tokens,
                "active_fraction": self.active_fraction,
                "requests_per_active_user_per_min": self.requests_per_active_user_per_min,
                "peak_multiplier": self.peak_multiplier,
            },
            "slo_targets": {
                "ttft_ms": {"min": self.ttft_range.min, "max": self.ttft_range.max},
                "itl_ms": {"min": self.itl_range.min, "max": self.itl_range.max},
                "e2e_ms": {"min": self.e2e_range.min, "max": self.e2e_range.max},
            },
        }


class UseCaseRepository:
    """Repository for use case definitions."""

    def __init__(self, data_path: Path | None = None):
        if data_path is None:
            data_path = resolve_data_path("configuration/usecase_slo_workload.json")

        self.data_path = data_path
        self._use_cases: dict[str, UseCaseConfig] = {}
        self._load_data()

    def _load_data(self):
        """Load use case definitions from JSON file."""
        try:
            with open(self.data_path) as f:
                data = json.load(f)
                for use_case_id, config_data in data["use_case_slo_workload"].items():
                    self._use_cases[use_case_id] = UseCaseConfig(use_case_id, config_data)
                logger.info(f"Loaded {len(self._use_cases)} use case definitions")
        except Exception as e:
            logger.error(f"Failed to load use cases from {self.data_path}: {e}")
            raise

    def get_use_case(self, use_case_id: str) -> UseCaseConfig | None:
        """Get configuration for a specific use case."""
        return self._use_cases.get(use_case_id)

    def get_all_use_cases(self) -> dict[str, UseCaseConfig]:
        """Get all use case configurations."""
        return self._use_cases.copy()

    def list_use_cases(self) -> list[str]:
        """Get list of all supported use case IDs."""
        return list(self._use_cases.keys())
