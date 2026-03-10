"""Data access layer for use case SLO templates."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SLOTemplate:
    """SLO template for a specific use case."""

    def __init__(self, use_case_id: str, data: dict):
        self.use_case_id = use_case_id
        self.use_case = data["use_case"]
        self.description = data["description"]

        # Traffic profile (GuideLLM configuration)
        traffic = data["traffic_profile"]
        self.prompt_tokens = traffic["prompt_tokens"]
        self.output_tokens = traffic["output_tokens"]

        # Experience class
        self.experience_class = data["experience_class"]

        # SLO targets (p95)
        slo = data["slo_targets"]
        self.ttft_p95_target_ms = slo["ttft_p95_ms"]
        self.itl_p95_target_ms = slo["itl_p95_ms"]
        self.e2e_p95_target_ms = slo["e2e_p95_ms"]

        # Rationale and business context
        self.rationale = data.get("rationale", "")

        context = data.get("business_context", {})
        self.user_facing = context.get("user_facing", True)
        self.latency_sensitivity = context.get("latency_sensitivity", "medium")
        self.throughput_priority = context.get("throughput_priority", "medium")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "use_case_id": self.use_case_id,
            "use_case": self.use_case,
            "description": self.description,
            "traffic_profile": {
                "prompt_tokens": self.prompt_tokens,
                "output_tokens": self.output_tokens,
            },
            "experience_class": self.experience_class,
            "slo_targets": {
                "ttft_p95_ms": self.ttft_p95_target_ms,
                "itl_p95_ms": self.itl_p95_target_ms,
                "e2e_p95_ms": self.e2e_p95_target_ms,
            },
            "rationale": self.rationale,
            "business_context": {
                "user_facing": self.user_facing,
                "latency_sensitivity": self.latency_sensitivity,
                "throughput_priority": self.throughput_priority,
            },
        }


class SLOTemplateRepository:
    """Repository for use case SLO templates."""

    def __init__(self, data_path: Path | None = None):
        """
        Initialize SLO template repository.

        Args:
            data_path: Path to slo_templates.json
        """
        if data_path is None:
            data_path = (
                Path(__file__).parent.parent.parent.parent
                / "data"
                / "configuration"
                / "slo_templates.json"
            )

        self.data_path = data_path
        self._templates: dict[str, SLOTemplate] = {}
        self._load_data()

    def _load_data(self):
        """Load SLO templates from JSON file."""
        try:
            with open(self.data_path) as f:
                data = json.load(f)
                for use_case_id, template_data in data["use_cases"].items():
                    self._templates[use_case_id] = SLOTemplate(use_case_id, template_data)
                logger.info(f"Loaded {len(self._templates)} SLO templates")
        except Exception as e:
            logger.error(f"Failed to load SLO templates from {self.data_path}: {e}")
            raise

    def get_template(self, use_case_id: str) -> SLOTemplate | None:
        """
        Get SLO template for a specific use case.

        Args:
            use_case_id: Use case identifier (e.g., 'chatbot_conversational')

        Returns:
            SLOTemplate if found, None otherwise
        """
        return self._templates.get(use_case_id)

    def get_all_templates(self) -> dict[str, SLOTemplate]:
        """Get all SLO templates."""
        return self._templates.copy()

    def list_use_cases(self) -> list[str]:
        """Get list of all supported use case IDs."""
        return list(self._templates.keys())

    def get_templates_by_traffic_profile(
        self, prompt_tokens: int, output_tokens: int
    ) -> list[SLOTemplate]:
        """
        Get all templates that use a specific traffic profile.

        Args:
            prompt_tokens: Prompt token count
            output_tokens: Output token count

        Returns:
            List of templates using this traffic profile
        """
        return [
            template
            for template in self._templates.values()
            if template.prompt_tokens == prompt_tokens and template.output_tokens == output_tokens
        ]

    def get_templates_by_experience_class(self, experience_class: str) -> list[SLOTemplate]:
        """
        Get all templates for a specific experience class.

        Args:
            experience_class: Experience class (instant, conversational, interactive, deferred, batch)

        Returns:
            List of templates for this experience class
        """
        return [
            template
            for template in self._templates.values()
            if template.experience_class == experience_class
        ]
