"""SLO specification components for the Planner UI.

SLO targets, workload profile, quality benchmarks, priorities, and approval.
"""

import json
import logging

import streamlit as st
from api_client import (
    fetch_expected_rps,
    fetch_priority_weights,
    fetch_slo_defaults,
    fetch_workload_profile,
    generate_specification,
)

from planner.data._resolver import data_path as _data_path

logger = logging.getLogger(__name__)

_WEIGHTS_PATH = _data_path("configuration/quality_weights.json")


def _load_quality_categories() -> dict:
    """Load quality weight categories from config."""
    if _WEIGHTS_PATH.is_file():
        with open(_WEIGHTS_PATH) as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}
    return {}


def _populate_specification_from_api(extraction: dict) -> dict | None:
    """Generate specification from extraction and populate session state.

    This replaces multiple GET calls (slo-defaults, workload-profile, expected-rps)
    with a single POST to generate-specification.

    Args:
        extraction: Extraction result dict with use_case, user_count, priorities, etc.

    Returns:
        The full specification dict, or None on error.
    """
    # Build intent from extraction
    intent = {
        "use_case": extraction.get("use_case", "chatbot_conversational"),
        "user_count": extraction.get("user_count", 1000),
        "quality_priority": extraction.get("quality_priority", "medium"),
        "cost_priority": extraction.get("cost_priority", "medium"),
        "latency_priority": extraction.get("latency_priority", "medium"),
        "preferred_gpu_types": extraction.get("preferred_gpu_types", []),
        "preferred_models": extraction.get("preferred_models", []),
        "domain_specialization": extraction.get("domain_specialization", ["general"]),
    }

    specification = generate_specification(intent)
    if not specification:
        return None

    # Populate session state with specification data
    slo_targets = specification["slo_targets"]
    st.session_state.input_ttft = slo_targets["ttft_target_ms"]
    st.session_state.input_itl = slo_targets["itl_target_ms"]
    st.session_state.input_e2e = slo_targets["e2e_target_ms"]
    st.session_state.slo_percentile = slo_targets["percentile"]

    # Initialize custom values if not already set
    if "custom_ttft" not in st.session_state:
        st.session_state.custom_ttft = slo_targets["ttft_target_ms"]
    if "custom_itl" not in st.session_state:
        st.session_state.custom_itl = slo_targets["itl_target_ms"]
    if "custom_e2e" not in st.session_state:
        st.session_state.custom_e2e = slo_targets["e2e_target_ms"]

    workload = specification["workload_profile"]
    st.session_state.spec_prompt_tokens = workload["prompt_tokens"]
    st.session_state.spec_output_tokens = workload["output_tokens"]
    st.session_state.spec_expected_qps = workload["expected_qps"]

    priorities = specification["priorities"]
    st.session_state.weight_quality = priorities["quality"]["weight"]
    st.session_state.weight_cost = priorities["cost"]["weight"]
    st.session_state.weight_latency = priorities["latency"]["weight"]

    return specification


def get_workload_insights(use_case: str, qps: int, user_count: int) -> list:
    """Get workload pattern insights based on API data.

    Returns list of tuples: (icon, color, message, severity)
    """
    messages: list[tuple[str, str, str, str]] = []

    workload_profile = fetch_workload_profile(use_case)
    if not workload_profile:
        return messages

    return messages


def _render_slo_targets(slo_defaults):
    """Render SLO inputs with one percentile shared by all latency targets."""
    st.write("**SLO Targets**")

    prev_percentile = st.session_state.get("_last_slo_percentile")
    percentile_options = ["p90", "p95", "p99"]
    st.selectbox(
        "SLO percentile",
        percentile_options,
        index=percentile_options.index(st.session_state.get("slo_percentile", "p95")),
        key="slo_percentile",
        format_func=str.upper,
        help="Use this percentile for TTFT, ITL, and E2E latency comparisons.",
    )

    prev_ttft = st.session_state.get("_last_ttft")
    prev_itl = st.session_state.get("_last_itl")
    prev_e2e = st.session_state.get("_last_e2e")

    def render_slo_input(metric: str, label: str, step: int):
        """Render a single latency target."""
        metric_key = {"ttft": "ttft_ms", "itl": "itl_ms", "e2e": "e2e_ms"}[metric]
        val_min = int(slo_defaults[metric_key]["min"])
        val_max = int(slo_defaults[metric_key]["max"])
        input_key = f"input_{metric}"
        custom_key = f"custom_{metric}"
        if input_key not in st.session_state:
            st.session_state[input_key] = slo_defaults[metric_key]["default"]
        st.write(f"**{label}**")
        st.number_input(
            f"{metric.upper()} value",
            min_value=1,
            step=step,
            key=input_key,
            label_visibility="collapsed",
        )
        st.session_state[custom_key] = st.session_state[input_key]
        st.caption(f"Recommended Range: {val_min:,} - {val_max:,} ms")

    render_slo_input("ttft", "TTFT (Time to First Token)", step=100)
    render_slo_input("itl", "ITL (Inter-Token Latency)", step=10)
    render_slo_input("e2e", "E2E (End-to-End Latency)", step=1000)

    curr_ttft = st.session_state.get("custom_ttft")
    curr_itl = st.session_state.get("custom_itl")
    curr_e2e = st.session_state.get("custom_e2e")

    if (
        curr_ttft != prev_ttft
        or curr_itl != prev_itl
        or curr_e2e != prev_e2e
        or st.session_state.slo_percentile != prev_percentile
    ):
        if st.session_state.get("recommendation_result"):
            st.session_state.recommendation_result = None
        st.session_state._last_ttft = curr_ttft
        st.session_state._last_itl = curr_itl
        st.session_state._last_e2e = curr_e2e
        st.session_state._last_slo_percentile = st.session_state.slo_percentile


def _render_workload_profile(use_case, workload_profile, estimated_qps, qps, user_count):
    """Render workload profile: QPS input, token counts, insights."""
    st.write("**Workload Profile**")

    prompt_tokens = workload_profile["prompt_tokens"]
    output_tokens = workload_profile["output_tokens"]

    st.session_state.spec_prompt_tokens = prompt_tokens
    st.session_state.spec_output_tokens = output_tokens

    default_qps = estimated_qps
    st.write("**Request Rate (Requests Per Second)**")
    new_qps = st.number_input(
        "Expected RPS",
        value=min(qps, 10000000),
        min_value=1,
        max_value=10000000,
        step=1,
        key="edit_qps",
        label_visibility="collapsed",
    )
    st.caption(f"Recommended RPS: {default_qps}")

    st.session_state.spec_expected_qps = new_qps

    if new_qps != qps:
        st.session_state.custom_qps = new_qps

    # Fixed workload values
    st.markdown(
        f"""
    <div style="margin-top: 0; padding: 0; border-radius: 8px;">
        <div style="padding: 0.5rem 0; ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.95rem; font-weight: 500;">Mean Input Tokens</span>
                <span style="font-weight: 700; font-size: 1.1rem; padding: 3px 10px; border-radius: 4px;">{prompt_tokens}</span>
        </div>
            <div style="font-size: 0.75rem; margin-top: 0.25rem; padding-left: 0;">Mean input token length per request</div>
        </div>
        <div style="padding: 0.5rem 0; ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.95rem; font-weight: 500;">Mean Output Tokens</span>
                <span style="font-weight: 700; font-size: 1.1rem; padding: 3px 10px; border-radius: 4px;">{output_tokens}</span>
            </div>
            <div style="font-size: 0.75rem; margin-top: 0.25rem; padding-left: 0;">Mean output token length per request</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Workload insights
    workload_messages = get_workload_insights(use_case, new_qps, user_count)
    for _, _, text, _ in workload_messages[:3]:
        st.markdown(
            f'<div style="font-size: 0.85rem; padding: 0.4rem 0.5rem; line-height: 1.4; border-radius: 6px; margin: 4px 0;">{text}</div>',
            unsafe_allow_html=True,
        )


def _render_quality_benchmarks(use_case: str) -> None:
    """Render the quality benchmark weights for the current use case."""
    st.write("**Quality Benchmarks**")

    weights_data = _load_quality_categories()
    use_case_key = use_case.lower().replace(" ", "_").replace("-", "_")
    config = weights_data.get(use_case_key, {})
    categories = config.get("categories", {})

    if not categories:
        st.info(f"No quality benchmarks configured for use case: {use_case}")
        return

    # Normalize weights to 0-1 range for progress bars
    total_weight = sum(categories.values())
    for cat_name, weight in sorted(categories.items(), key=lambda x: -x[1]):
        normalized = weight / total_weight if total_weight > 0 else 0
        display_name = cat_name.replace("_", " ").title()
        pct_label = f"{int(normalized * 100)}%"
        st.progress(normalized, text=f"{display_name} — {pct_label}")


def _render_priorities():
    """Render priority dropdowns and weight inputs for quality, cost, latency."""
    st.write("**Priorities**")

    priority_config = fetch_priority_weights()
    priority_weights_map = priority_config.get("priority_weights", {}) if priority_config else {}
    defaults_config = priority_config.get("defaults", {}) if priority_config else {}
    logger.debug(
        f"Priority config loaded: weights_map keys={list(priority_weights_map.keys())}, defaults={defaults_config}"
    )

    priority_options = ["Low", "Medium", "High"]
    priority_to_value = {"Low": "low", "Medium": "medium", "High": "high"}
    value_to_priority = {"low": "Low", "medium": "Medium", "high": "High"}

    def get_weight_for_priority(dimension: str, priority_level: str) -> int:
        pw = priority_weights_map.get(dimension, {})
        default_priority = defaults_config.get("priority", "medium")
        fallback = {"quality": 4, "cost": 4, "latency": 1}
        weight = pw.get(priority_level, pw.get(default_priority, fallback.get(dimension, 4)))
        logger.info(
            f"get_weight_for_priority({dimension}, {priority_level}): pw={pw}, weight={weight}"
        )
        return int(weight)

    extraction = st.session_state.get("extraction_result", {})

    # Update priorities from new LLM extraction
    if st.session_state.get("new_extraction_available", False):
        st.session_state.quality_priority = extraction.get("quality_priority", "medium")
        st.session_state.cost_priority = extraction.get("cost_priority", "medium")
        st.session_state.latency_priority = extraction.get("latency_priority", "medium")
        st.session_state.weight_quality = get_weight_for_priority(
            "quality", st.session_state.quality_priority
        )
        st.session_state.weight_cost = get_weight_for_priority(
            "cost", st.session_state.cost_priority
        )
        st.session_state.weight_latency = get_weight_for_priority(
            "latency", st.session_state.latency_priority
        )
        st.session_state.new_extraction_available = False
        logger.info(
            f"Updated priorities from new extraction: quality={st.session_state.quality_priority}, "
            f"cost={st.session_state.cost_priority}, latency={st.session_state.latency_priority}"
        )

    # Initialize session state for priorities if not set (first load)
    if "quality_priority" not in st.session_state:
        st.session_state.quality_priority = extraction.get("quality_priority", "medium")
    if "cost_priority" not in st.session_state:
        st.session_state.cost_priority = extraction.get("cost_priority", "medium")
    if "latency_priority" not in st.session_state:
        st.session_state.latency_priority = extraction.get("latency_priority", "medium")

    # Initialize weights based on priorities
    if "weight_quality" not in st.session_state:
        st.session_state.weight_quality = get_weight_for_priority(
            "quality", st.session_state.quality_priority
        )
    if "weight_cost" not in st.session_state:
        st.session_state.weight_cost = get_weight_for_priority(
            "cost", st.session_state.cost_priority
        )
    if "weight_latency" not in st.session_state:
        st.session_state.weight_latency = get_weight_for_priority(
            "latency", st.session_state.latency_priority
        )

    # Column headers
    header_label, header_priority, header_weight = st.columns([1.2, 2, 1])
    with header_label:
        st.caption("Metric")
    with header_priority:
        st.caption("Priority")
    with header_weight:
        st.caption("Weight")

    # Priority rows: label | dropdown | weight input
    for dimension, state_key in [
        ("Quality", "quality"),
        ("Cost", "cost"),
        ("Latency", "latency"),
    ]:
        label_col, dropdown_col, weight_col = st.columns([1.2, 2, 1])
        with label_col:
            st.markdown(
                f'<span style="font-size: 0.9rem; font-weight: 500; line-height: 2.5;">{dimension}</span>',
                unsafe_allow_html=True,
            )
        with dropdown_col:
            current_priority = st.session_state.get(f"{state_key}_priority", "medium")
            display = value_to_priority.get(current_priority, "Medium")
            new_priority = st.selectbox(
                dimension,
                priority_options,
                index=priority_options.index(display),
                key=f"{state_key}_priority_select",
                label_visibility="collapsed",
            )
            new_priority_val = priority_to_value[new_priority]
            if new_priority_val != current_priority:
                st.session_state[f"{state_key}_priority"] = new_priority_val
                new_weight_val = get_weight_for_priority(state_key, new_priority_val)
                st.session_state[f"weight_{state_key}"] = new_weight_val
                st.session_state[f"{state_key}_weight_input"] = new_weight_val
        with weight_col:
            current_weight = st.session_state.get(f"weight_{state_key}", 5)
            new_weight = st.number_input(
                f"{dimension} weight",
                min_value=0,
                max_value=10,
                value=current_weight,
                key=f"{state_key}_weight_input",
                label_visibility="collapsed",
            )
            if new_weight != current_weight:
                st.session_state[f"weight_{state_key}"] = new_weight


def render_slo_cards(use_case: str, user_count: int):
    """Render SLO and workload specification cards in a 4-column layout."""
    slo_defaults = fetch_slo_defaults(use_case)
    if not slo_defaults:
        st.error(f"Failed to fetch SLO defaults for use case: {use_case}")
        return

    rps_data = fetch_expected_rps(use_case, user_count)
    if not rps_data:
        st.error("Failed to fetch expected RPS from backend. Please ensure the backend is running.")
        return
    estimated_qps = int(rps_data.get("expected_rps", 1))

    workload_profile = fetch_workload_profile(use_case)
    if not workload_profile:
        st.error(f"Failed to fetch workload profile for use case: {use_case}")
        return

    # Track if use_case or user_count changed - reset custom_qps to new default
    last_use_case = st.session_state.get("_last_rps_use_case")
    last_user_count = st.session_state.get("_last_rps_user_count")
    if last_use_case != use_case or last_user_count != user_count:
        st.session_state.custom_qps = None
        st.session_state._last_rps_use_case = use_case
        st.session_state._last_rps_user_count = user_count
        if "edit_qps" in st.session_state:
            del st.session_state["edit_qps"]

    qps = int(st.session_state.custom_qps) if st.session_state.custom_qps else estimated_qps

    use_case_display = use_case.replace("_", " ").title()
    st.subheader(f"Technical Specification (Use Case: {use_case_display})")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _render_slo_targets(slo_defaults)
    with col2:
        _render_workload_profile(use_case, workload_profile, estimated_qps, qps, user_count)
    with col3:
        _render_quality_benchmarks(use_case)
    with col4:
        _render_priorities()


def _render_constraints(extraction: dict):
    """Render GPU and model constraints if any are set."""
    gpu_types = extraction.get("preferred_gpu_types", [])
    models = extraction.get("preferred_models", []) or st.session_state.get("preferred_models", [])

    if not gpu_types and not models:
        return

    parts = []
    if gpu_types:
        gpu_labels = []
        for g in gpu_types:
            if isinstance(g, dict) and "gpu_type" in g:
                label = g["gpu_type"]
                if g.get("max_count"):
                    label += f" (max {g['max_count']})"
                gpu_labels.append(label)
            else:
                gpu_labels.append(str(g))
        parts.append(f"**GPUs:** {', '.join(gpu_labels)}")
    else:
        parts.append("**GPUs:** Any")
    if models:
        parts.append(f"**Models:** {', '.join(models)}")
    else:
        parts.append("**Models:** Any")

    st.markdown(" &nbsp;|&nbsp; ".join(parts))


def render_slo_with_approval(extraction: dict):
    """Render SLO section with approval to proceed to recommendations."""
    use_case = extraction.get("use_case", "chatbot_conversational")
    user_count = extraction.get("user_count", 1000)

    # Check if we need to populate specification data from the API
    # This replaces the piecemeal GET calls in render_slo_cards
    if "_specification_populated" not in st.session_state:
        with st.spinner("Loading specification data..."):
            spec = _populate_specification_from_api(extraction)
            if spec:
                st.session_state._specification_populated = True
            else:
                st.error("Failed to generate specification. Using fallback values.")
                # Fall through to render_slo_cards which will use old GET endpoints as fallback

    render_slo_cards(use_case, user_count)
    _render_constraints(extraction)

    # No validation needed - allow any positive values
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "Generate Recommendations",
            type="primary",
            width="stretch",
            key="generate_recs",
        ):
            st.session_state.slo_approved = True
            st.session_state.recommendation_result = None
            st.session_state.pop("_last_spec_fingerprint", None)
            st.session_state._pending_tab = 2
            for _cat in ("balanced", "quality", "latency", "cost"):
                st.session_state[f"cat_idx_{_cat}"] = 0
            st.rerun()
