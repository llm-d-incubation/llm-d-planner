"""GPU type normalization utility.

Normalizes user-specified GPU types to canonical names used in benchmark data.
Uses ModelCatalog as the single source of truth for GPU aliases.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..knowledge_base.model_catalog import ModelCatalog

logger = logging.getLogger(__name__)

# Canonical GPU names from benchmark data
CANONICAL_GPUS = {"L4", "A100-40", "A100-80", "H100", "H200", "B200"}

# Expansion map for shorthand/ambiguous names
# When user says "A100" without specifying variant, include both
GPU_EXPANSIONS = {
    "A100": ["A100-80", "A100-40"],
}

# Singleton catalog instance to avoid repeated loading
_catalog_instance: "ModelCatalog | None" = None


def _get_catalog() -> "ModelCatalog":
    """Get or create the ModelCatalog singleton."""
    global _catalog_instance
    if _catalog_instance is None:
        from ..knowledge_base.model_catalog import ModelCatalog
        _catalog_instance = ModelCatalog()
    return _catalog_instance


def normalize_gpu_types(gpu_types: list[str]) -> list[str]:
    """
    Normalize GPU types to canonical names using ModelCatalog aliases.

    - Case-insensitive matching
    - Uses ModelCatalog's alias lookup (from model_catalog.json)
    - Expands shorthand (A100 → [A100-80, A100-40])
    - Returns empty list for empty input

    Args:
        gpu_types: List of GPU type strings from user input or intent extraction

    Returns:
        List of canonical GPU names (uppercase), deduplicated and sorted
    """
    if not gpu_types:
        return []

    catalog = _get_catalog()
    normalized = set()

    for gpu in gpu_types:
        if not gpu or not isinstance(gpu, str):
            continue

        gpu_stripped = gpu.strip()
        gpu_upper = gpu_stripped.upper()

        # Skip empty or "any gpu" values
        if not gpu_upper or gpu_upper == "ANY GPU":
            continue

        # Check if it's an expansion case (e.g., A100 → both variants)
        if gpu_upper in GPU_EXPANSIONS:
            normalized.update(GPU_EXPANSIONS[gpu_upper])
            logger.debug(f"Expanded '{gpu}' to {GPU_EXPANSIONS[gpu_upper]}")
            continue

        # Use ModelCatalog's alias lookup (handles case-insensitivity)
        gpu_info = catalog.get_gpu_type(gpu_stripped)
        if gpu_info:
            normalized.add(gpu_info.gpu_type.upper())
            logger.debug(f"Resolved '{gpu}' to '{gpu_info.gpu_type}' via ModelCatalog")
            continue

        # Check if it's already a canonical name (direct match)
        if gpu_upper in CANONICAL_GPUS:
            normalized.add(gpu_upper)
            continue

        # Unknown GPU type - log warning and skip
        logger.warning(
            f"Unknown GPU type '{gpu}' - not found in ModelCatalog or canonical list. "
            "Skipping this GPU filter."
        )

    return sorted(normalized)  # Sorted for consistent ordering
