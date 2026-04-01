"""YAML validation module for generated deployment configurations."""

import logging
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for YAML validation errors."""

    pass


class YAMLValidator:
    """Validate generated YAML configurations."""

    # Required fields for KServe InferenceService
    KSERVE_REQUIRED_FIELDS = [
        "apiVersion",
        "kind",
        "metadata.name",
        "spec.predictor",
    ]

    # Required fields for HPA
    HPA_REQUIRED_FIELDS = [
        "apiVersion",
        "kind",
        "metadata.name",
        "spec.scaleTargetRef",
        "spec.minReplicas",
        "spec.maxReplicas",
    ]

    # Required fields for ServiceMonitor
    SERVICEMONITOR_REQUIRED_FIELDS = [
        "apiVersion",
        "kind",
        "metadata.name",
        "spec.selector",
        "spec.endpoints",
    ]

    def __init__(self):
        """Initialize the validator."""
        pass

    def validate_yaml_syntax(self, file_path: str) -> bool:
        """
        Validate YAML syntax by attempting to parse the file.

        Supports both single-document and multi-document YAML files.

        Args:
            file_path: Path to YAML file

        Returns:
            True if valid YAML syntax

        Raises:
            ValidationError: If YAML syntax is invalid
        """
        try:
            with open(file_path) as f:
                # Use safe_load_all for multi-document YAML files
                docs = list(yaml.safe_load_all(f))
                if not docs or all(doc is None for doc in docs):
                    raise ValidationError(f"No valid YAML documents found in {file_path}")
            logger.info(f"YAML syntax valid: {file_path} ({len(docs)} document(s))")
            return True
        except yaml.YAMLError as e:
            raise ValidationError(f"Invalid YAML syntax in {file_path}: {e}") from e

    def _get_nested_field(self, data: dict[str, Any], field_path: str) -> Any | None:
        """
        Get nested field from dictionary using dot notation.

        Args:
            data: Dictionary to search
            field_path: Dot-separated field path (e.g., "metadata.name")

        Returns:
            Field value if found, None otherwise
        """
        parts = field_path.split(".")
        current = data

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def validate_required_fields(self, file_path: str, required_fields: list[str]) -> bool:
        """
        Validate that required fields are present in at least one document.

        Args:
            file_path: Path to YAML file
            required_fields: List of required field paths

        Returns:
            True if all required fields present

        Raises:
            ValidationError: If required fields are missing
        """
        with open(file_path) as f:
            docs = list(yaml.safe_load_all(f))

        # Check first non-empty document
        data = next((doc for doc in docs if doc is not None), None)
        if data is None:
            raise ValidationError(f"No valid YAML documents found in {file_path}")

        missing_fields = []
        for field in required_fields:
            if self._get_nested_field(data, field) is None:
                missing_fields.append(field)

        if missing_fields:
            raise ValidationError(
                f"Missing required fields in {file_path}: {', '.join(missing_fields)}"
            )

        logger.info(f"All required fields present: {file_path}")
        return True

    def validate_kserve_yaml(self, file_path: str) -> bool:
        """
        Validate KServe InferenceService YAML.

        Args:
            file_path: Path to KServe YAML file

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        # Check syntax
        self.validate_yaml_syntax(file_path)

        # Check required fields
        self.validate_required_fields(file_path, self.KSERVE_REQUIRED_FIELDS)

        # Additional KServe-specific checks
        with open(file_path) as f:
            data = yaml.safe_load(f)

        # Verify it's a KServe InferenceService
        if data.get("kind") != "InferenceService":
            raise ValidationError(f"Expected kind 'InferenceService', got '{data.get('kind')}'")

        if not data.get("apiVersion", "").startswith("serving.kserve.io"):
            raise ValidationError(
                f"Expected apiVersion 'serving.kserve.io/v1beta1', "
                f"got '{data.get('apiVersion')}'"
            )

        # Check if this is simulator mode (skip GPU validation)
        annotations = data.get("metadata", {}).get("annotations", {})
        simulator_mode = annotations.get("neuralnav/simulator-mode") == "true"

        # Verify GPU resources are specified (unless simulator mode)
        predictor = data.get("spec", {}).get("predictor", {})
        containers = predictor.get("containers", [])

        if not containers:
            raise ValidationError("No containers defined in predictor")

        if not simulator_mode:
            # Only validate GPU resources for real vLLM mode
            resources = containers[0].get("resources", {})
            gpu_requests = resources.get("requests", {}).get("nvidia.com/gpu")
            gpu_limits = resources.get("limits", {}).get("nvidia.com/gpu")

            if not gpu_requests or not gpu_limits:
                raise ValidationError("GPU resources not specified in container")
        else:
            logger.info("Simulator mode detected - skipping GPU validation")

        logger.info(f"KServe YAML validation passed: {file_path}")
        return True

    def validate_hpa_yaml(self, file_path: str) -> bool:
        """
        Validate HPA YAML (may contain multiple documents).

        Args:
            file_path: Path to HPA YAML file

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        # Check syntax
        self.validate_yaml_syntax(file_path)

        # Check required fields on the first document (HPA)
        self.validate_required_fields(file_path, self.HPA_REQUIRED_FIELDS)

        with open(file_path) as f:
            # Load all documents (in case of multi-document YAML)
            docs = list(yaml.safe_load_all(f))

        # Find the HPA document
        hpa_doc = None
        for doc in docs:
            if doc and doc.get("kind") == "HorizontalPodAutoscaler":
                hpa_doc = doc
                break

        if not hpa_doc:
            raise ValidationError(f"No HorizontalPodAutoscaler document found in {file_path}")

        # Verify min <= max replicas
        min_replicas = hpa_doc.get("spec", {}).get("minReplicas", 0)
        max_replicas = hpa_doc.get("spec", {}).get("maxReplicas", 0)

        if min_replicas > max_replicas:
            raise ValidationError(f"minReplicas ({min_replicas}) > maxReplicas ({max_replicas})")

        logger.info(f"HPA YAML validation passed: {file_path}")
        return True

    def validate_servicemonitor_yaml(self, file_path: str) -> bool:
        """
        Validate ServiceMonitor YAML (may contain multiple documents).

        Args:
            file_path: Path to ServiceMonitor YAML file

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        # Check syntax
        self.validate_yaml_syntax(file_path)

        # Check required fields on the first document
        self.validate_required_fields(file_path, self.SERVICEMONITOR_REQUIRED_FIELDS)

        with open(file_path) as f:
            docs = list(yaml.safe_load_all(f))

        # Find the ServiceMonitor document
        servicemonitor_doc = None
        for doc in docs:
            if doc and doc.get("kind") == "ServiceMonitor":
                servicemonitor_doc = doc
                break

        if not servicemonitor_doc:
            raise ValidationError(f"No ServiceMonitor document found in {file_path}")

        logger.info(f"ServiceMonitor YAML validation passed: {file_path} ({len(docs)} document(s))")
        return True

    def validate_all(self, files: dict[str, str]) -> dict[str, bool]:
        """
        Validate all generated YAML files.

        Args:
            files: Dictionary mapping config type to file path

        Returns:
            Dictionary mapping config type to validation result

        Raises:
            ValidationError: If any validation fails
        """
        results = {}

        for config_type, file_path in files.items():
            try:
                if "inferenceservice" in config_type.lower():
                    self.validate_kserve_yaml(file_path)
                elif "autoscaling" in config_type.lower() or "hpa" in config_type.lower():
                    self.validate_hpa_yaml(file_path)
                elif "servicemonitor" in config_type.lower():
                    self.validate_servicemonitor_yaml(file_path)
                else:
                    # Generic YAML syntax check
                    self.validate_yaml_syntax(file_path)

                results[config_type] = True

            except ValidationError as e:
                logger.error(f"Validation failed for {config_type}: {e}")
                raise

        logger.info(f"All YAML files validated successfully: {list(files.keys())}")
        return results
