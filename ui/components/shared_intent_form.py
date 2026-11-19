"""Shared intent form fields used by both Form mode and Modify Business Context.

Renders use case, user count, priorities, GPU preferences (with optional max_count),
and model preferences. Returns a DeploymentIntent dict when the caller submits.
"""

import streamlit as st
from api_client import fetch_catalog_model_ids, fetch_gpu_types, fetch_use_cases

PRIORITY_OPTIONS = ["low", "medium", "high"]

# Static fallback when the backend is unreachable.  Keeps the dropdown usable
# until fetch_use_cases() succeeds and caches a real response.
_FALLBACK_USE_CASE_KEYS = [
    "chatbot_conversational",
    "code_completion",
    "code_generation_detailed",
    "translation",
    "content_generation",
    "summarization_short",
    "document_analysis_rag",
    "long_document_summarization",
    "research_legal_analysis",
]


def render_intent_fields(
    defaults: dict | None = None,
    key_prefix: str = "intent",
) -> dict:
    """Render the shared intent form fields and return the current values.

    Args:
        defaults: Optional dict to pre-populate fields (e.g., from prior extraction).
        key_prefix: Streamlit widget key prefix to avoid collisions.

    Returns:
        Dict with all DeploymentIntent fields reflecting current widget values.
    """
    defaults = defaults or {}

    # Use case — display names from API, with static fallback if backend is down.
    # The fallback prevents fetch_use_cases()'s cached {} from disabling the dropdown.
    # See https://github.com/llm-d-incubation/llm-d-planner/issues/356 for the
    # systemic api_client.py caching problem.
    use_case_data = fetch_use_cases()
    if not use_case_data:
        use_case_data = {
            k: {"display_name": k.replace("_", " ").title()} for k in _FALLBACK_USE_CASE_KEYS
        }
    use_case_keys = list(use_case_data.keys())
    use_case_display = {k: v.get("display_name", k) for k, v in use_case_data.items()}

    current_use_case = defaults.get("use_case", "chatbot_conversational")
    current_idx = use_case_keys.index(current_use_case) if current_use_case in use_case_keys else 0

    use_case = st.selectbox(
        "Use Case",
        options=use_case_keys,
        index=current_idx,
        format_func=lambda x: use_case_display.get(x, x),
        key=f"{key_prefix}_use_case",
    )

    # User count
    user_count = st.number_input(
        "User Count",
        min_value=1,
        max_value=1000000,
        value=defaults.get("user_count", 1000),
        step=100,
        key=f"{key_prefix}_user_count",
    )

    # 3 priority selectors
    col1, col2, col3 = st.columns(3)
    with col1:
        quality_priority = st.selectbox(
            "Quality Priority",
            options=PRIORITY_OPTIONS,
            index=PRIORITY_OPTIONS.index(defaults.get("quality_priority", "medium")),
            help="How important is model quality/capability?",
            key=f"{key_prefix}_quality_priority",
        )
    with col2:
        cost_priority = st.selectbox(
            "Cost Priority",
            options=PRIORITY_OPTIONS,
            index=PRIORITY_OPTIONS.index(defaults.get("cost_priority", "medium")),
            help="How important is cost efficiency?",
            key=f"{key_prefix}_cost_priority",
        )
    with col3:
        latency_priority = st.selectbox(
            "Latency Priority",
            options=PRIORITY_OPTIONS,
            index=PRIORITY_OPTIONS.index(defaults.get("latency_priority", "medium")),
            help="How important is low latency?",
            key=f"{key_prefix}_latency_priority",
        )

    # GPU preferences with optional max_count
    gpu_types_data = fetch_gpu_types()
    available_gpus = sorted(gpu_types_data.keys()) if gpu_types_data else []
    current_gpus = _extract_gpu_names(defaults.get("preferred_gpu_types", []))
    valid_current_gpus = [g for g in current_gpus if g in available_gpus]

    selected_gpus = st.multiselect(
        "Preferred GPU Types (optional)",
        options=available_gpus,
        default=valid_current_gpus,
        help="Leave empty to consider all GPU types",
        key=f"{key_prefix}_gpu_types",
    )

    # Max GPU count inputs for each selected GPU
    gpu_max_counts = _extract_gpu_max_counts(defaults.get("preferred_gpu_types", []))
    preferred_gpu_types: list = []
    if selected_gpus:
        max_count_cols = st.columns(len(selected_gpus))
        for i, gpu in enumerate(selected_gpus):
            with max_count_cols[i]:
                default_max = gpu_max_counts.get(gpu)
                max_count = st.number_input(
                    f"{gpu} max GPUs",
                    min_value=0,
                    max_value=64,
                    value=default_max if default_max is not None else 0,
                    step=1,
                    help="0 = no limit",
                    key=f"{key_prefix}_gpu_max_{gpu}",
                )
                if max_count > 0:
                    preferred_gpu_types.append({"gpu_type": gpu, "max_count": max_count})
                else:
                    preferred_gpu_types.append(gpu)

    # Model preferences
    st.markdown("**Model Preferences** (optional)")
    col_models, col_custom = st.columns(2, gap="medium")
    with col_models:
        catalog_model_ids = fetch_catalog_model_ids()
        current_models = defaults.get("preferred_models", [])
        catalog_current = [m for m in current_models if m in catalog_model_ids]
        selected_catalog_models = st.multiselect(
            "Catalog Models",
            catalog_model_ids,
            default=catalog_current,
            key=f"{key_prefix}_catalog_models",
            help="Select from approved model catalog",
        )
    with col_custom:
        custom_current = [m for m in current_models if m not in catalog_model_ids]
        custom_models_str = st.text_input(
            "Custom HuggingFace Model IDs",
            value=", ".join(custom_current),
            key=f"{key_prefix}_custom_models",
            help="Comma-separated HuggingFace model IDs (e.g., meta-llama/Llama-3.3-70B-Instruct)",
        )

    custom_models_list = [m.strip() for m in custom_models_str.split(",") if m.strip()]
    all_preferred_models = list(dict.fromkeys(selected_catalog_models + custom_models_list))

    return {
        "use_case": use_case,
        "user_count": user_count,
        "quality_priority": quality_priority,
        "cost_priority": cost_priority,
        "latency_priority": latency_priority,
        "preferred_gpu_types": preferred_gpu_types,
        "preferred_models": all_preferred_models,
        "domain_specialization": defaults.get("domain_specialization", ["general"]),
    }


def _extract_gpu_names(gpu_types: list) -> list[str]:
    """Extract plain GPU type names from a mixed list of strings and GpuPreference dicts."""
    names = []
    for item in gpu_types:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and "gpu_type" in item:
            names.append(item["gpu_type"])
    return names


def _extract_gpu_max_counts(gpu_types: list) -> dict[str, int]:
    """Extract max_count values from GpuPreference dicts."""
    counts = {}
    for item in gpu_types:
        if isinstance(item, dict) and "gpu_type" in item and item.get("max_count"):
            counts[item["gpu_type"]] = item["max_count"]
    return counts
