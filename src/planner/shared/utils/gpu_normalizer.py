"""GPU type normalization utility.

Normalizes user-specified GPU types to canonical names used in benchmark data.
Uses ModelCatalog as the single source of truth for GPU aliases.
"""

import logging
import re
from typing import TYPE_CHECKING

# Mapping from gpu_catalog.json canonical gpu_type to llm_optimizer GPU_SPECS keys.
# GPUs not in this map are not supported by the roofline model.
# Most names map 1:1; only the A100 variants differ.
CATALOG_TO_OPTIMIZER_GPU: dict[str, str] = {
    "H100": "H100",
    "H200": "H200",
    "A100-80": "A100",
    "A100-40": "A100-40GB",
    "L40": "L40",
    "L20": "L20",
    "B100": "B100",
    "B200": "B200",
}


def catalog_to_optimizer_gpu_name(catalog_name: str) -> str:
    """Translate a catalog GPU name to the llm_optimizer GPU_SPECS key.

    Returns the original name unchanged if not in the mapping.
    """
    return CATALOG_TO_OPTIMIZER_GPU.get(catalog_name, catalog_name)


if TYPE_CHECKING:
    from planner.knowledge_base.model_catalog import ModelCatalog

logger = logging.getLogger(__name__)

# Expansion map for shorthand/ambiguous names
# When user says "A100" without specifying variant, include both
GPU_EXPANSIONS = {
    "A100": ["A100-80", "A100-40"],
}

# Known GPU model tokens, ordered longest-first to prevent partial matches.
# E.g., "L40" must be checked before "L4", "B200" before "B100".
_KNOWN_GPU_TOKENS = (
    "MI300X",
    "A100",
    "A10G",
    "B200",
    "B100",
    "H100",
    "H200",
    "L40",
    "L20",
    "L4",
    "T4",
)

# Vendor prefixes to strip, ordered longest-first
_VENDOR_PREFIXES = ("NVIDIA-GEFORCE-", "NVIDIA-", "AMD-INSTINCT-", "AMD-", "TESLA-")

# Singleton catalog instance to avoid repeated loading
_catalog_instance: "ModelCatalog | None" = None


def _get_catalog() -> "ModelCatalog":
    """Get or create the ModelCatalog singleton."""
    global _catalog_instance
    if _catalog_instance is None:
        from planner.knowledge_base.model_catalog import ModelCatalog

        _catalog_instance = ModelCatalog()
    return _catalog_instance


def _disambiguate_expansion(key: str, original_upper: str) -> list[str]:
    """Disambiguate an expansion key using memory info from the original string.

    For "A100": if "40" appears as a word boundary in original -> A100-40 only.
    If "80" appears -> A100-80 only. Otherwise return both variants.

    Examples:
        >>> _disambiguate_expansion("A100", "NVIDIA-A100-SXM4-40GB")
        ["A100-40"]

        >>> _disambiguate_expansion("A100", "NVIDIA-A100-SXM4-80GB")
        ["A100-80"]

        >>> _disambiguate_expansion("A100", "NVIDIA-A100")
        ["A100-40", "A100-80"]

        >>> _disambiguate_expansion("A100", "A100-40GB-PCIE")
        ["A100-40"]
    """
    variants = GPU_EXPANSIONS[key]
    # Look for memory size indicators in the original string
    if re.search(r"(?:^|[-_])40(?:GB)?(?:$|[-_])", original_upper):
        return [v for v in variants if "40" in v]
    if re.search(r"(?:^|[-_])80(?:GB)?(?:$|[-_])", original_upper):
        return [v for v in variants if "80" in v]
    return list(variants)


def _fuzzy_resolve(raw: str, catalog: "ModelCatalog") -> list[str]:
    """Pattern-based fallback for GPU strings not found by exact/alias lookup.

    Called only after GPU_EXPANSIONS check and catalog.get_gpu_type() miss.
    Returns list of canonical uppercase GPU type names, or empty list.
    """
    upper = raw.strip().upper()
    if not upper:
        return []

    # Step 1: Strip vendor prefix
    stripped = upper
    for prefix in _VENDOR_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :]
            break

    # Step 2: Try catalog lookup on stripped string
    gpu_info = catalog.get_gpu_type(stripped)
    if gpu_info:
        return [gpu_info.gpu_type.upper()]

    # Step 3: Check GPU_EXPANSIONS on stripped string
    if stripped in GPU_EXPANSIONS:
        return _disambiguate_expansion(stripped, upper)

    # Step 4: Progressive suffix removal (split by "-")
    parts = stripped.split("-")
    for length in range(len(parts) - 1, 0, -1):
        candidate = "-".join(parts[:length])

        gpu_info = catalog.get_gpu_type(candidate)
        if gpu_info:
            return [gpu_info.gpu_type.upper()]
        if candidate in GPU_EXPANSIONS:
            return _disambiguate_expansion(candidate, upper)

    return []


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

        # Fallback: pattern-based fuzzy resolution
        resolved = _fuzzy_resolve(gpu_stripped, catalog)
        if resolved:
            normalized.update(resolved)
            logger.debug(f"Fuzzy-resolved '{gpu}' to {resolved}")
            continue

        # Truly unknown GPU type - log warning and skip
        logger.warning(
            f"Unknown GPU type '{gpu}' - not found in ModelCatalog or by pattern matching. "
            "Skipping this GPU filter."
        )

    return sorted(normalized)  # Sorted for consistent ordering
