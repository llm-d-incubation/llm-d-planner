"""SLO specification components for the Planner UI.

SLO targets, workload profile, accuracy benchmarks, priorities, and approval.
"""

import logging

import streamlit as st
from api_client import (
    fetch_expected_rps,
    fetch_priority_weights,
    fetch_slo_defaults,
    fetch_workload_profile,
)

logger = logging.getLogger(__name__)

# Benchmark weights per use case (from USE_CASE_METHODOLOGY.md)
TASK_DATASETS = {
    "chatbot_conversational": [
        ("τ²-Bench", 45, "Conversational AI agentic workflow (114 tasks)"),
        ("MMLU-Pro", 35, "General knowledge (12,032 questions)"),
        ("GPQA", 20, "Scientific reasoning (198 questions)"),
    ],
    "code_completion": [
        ("LiveCodeBench", 45, "Code benchmark (315 questions)"),
        ("τ²-Bench", 35, "Agentic code assistance (114 tasks)"),
        ("MMLU-Pro", 20, "Knowledge for context"),
    ],
    "code_generation_detailed": [
        ("LiveCodeBench", 40, "Code generation (315 questions)"),
        ("τ²-Bench", 40, "Agentic reasoning (114 tasks)"),
        ("GPQA", 20, "Scientific reasoning for explanations"),
    ],
    "translation": [
        ("τ²-Bench", 45, "Language agentic tasks (114 tasks)"),
        ("MMLU-Pro", 35, "Language understanding (12,032 questions)"),
        ("GPQA", 20, "Reasoning"),
    ],
    "content_generation": [
        ("τ²-Bench", 45, "Creative agentic workflow (114 tasks)"),
        ("MMLU-Pro", 35, "General knowledge for facts"),
        ("GPQA", 20, "Reasoning"),
    ],
    "summarization_short": [
        ("τ²-Bench", 45, "Summarization agentic (114 tasks)"),
        ("MMLU-Pro", 35, "Comprehension (12,032 questions)"),
        ("GPQA", 20, "Reasoning"),
    ],
    "document_analysis_rag": [
        ("τ²-Bench", 50, "RAG is agentic workflow - DOMINANT (114 tasks)"),
        ("GPQA", 30, "Scientific reasoning for factual answers"),
        ("MMLU-Pro", 20, "Knowledge retrieval"),
    ],
    "long_document_summarization": [
        ("τ²-Bench", 50, "Long doc handling is agentic (114 tasks)"),
        ("MMLU-Pro", 30, "Knowledge for understanding"),
        ("GPQA", 20, "Reasoning"),
    ],
    "research_legal_analysis": [
        ("τ²-Bench", 55, "Research analysis is agentic reasoning - CRITICAL"),
        ("GPQA", 25, "Scientific reasoning (198 questions)"),
        ("MMLU-Pro", 20, "Knowledge (12,032 questions)"),
    ],
}


def get_workload_insights(use_case: str, qps: int, user_count: int) -> list:
    """Get workload pattern insights based on API data.

    Returns list of tuples: (icon, color, message, severity)
    """
    messages: list[tuple[str, str, str, str]] = []

    workload_profile = fetch_workload_profile(use_case)
    if not workload_profile:
        return messages

    distribution = workload_profile.get("distribution", "poisson")
    active_fraction = workload_profile.get("active_fraction", 0.2)

    messages.append(
        (
            "",
            "",
            f"Pattern: {distribution.replace('_', ' ').title()} | {int(active_fraction*100)}% concurrent users",
            "info",
        )
    )

    return messages


def _render_slo_targets(slo_defaults):
    """Render the SLO target inputs (TTFT, ITL, E2E) with percentile selectors."""
    st.write("**SLO Targets**")

    percentile_options = ["P50", "P90", "P95", "P99"]
    percentile_map = {"P50": "p50", "P90": "p90", "P95": "p95", "P99": "p99"}
    reverse_map = {"p50": "P50", "p90": "P90", "p95": "P95", "p99": "P99"}

    prev_ttft = st.session_state.get("_last_ttft")
    prev_itl = st.session_state.get("_last_itl")
    prev_e2e = st.session_state.get("_last_e2e")

    def render_slo_input(metric: str, label: str, step: int):
        """Render a single SLO input (value + percentile selector)."""
        metric_key = {"ttft": "ttft_ms", "itl": "itl_ms", "e2e": "e2e_ms"}[metric]
        val_min = int(slo_defaults[metric_key]["min"])
        val_max = int(slo_defaults[metric_key]["max"])
        input_key = f"input_{metric}"
        custom_key = f"custom_{metric}"
        pct_key = f"{metric}_percentile"
        if input_key not in st.session_state:
            st.session_state[input_key] = slo_defaults[metric_key]["default"]
        st.write(f"**{label}**")
        val_col, pct_col = st.columns([2, 1])
        with val_col:
            st.number_input(
                f"{metric.upper()} value",
                min_value=val_min,
                max_value=val_max,
                step=step,
                key=input_key,
                label_visibility="collapsed",
            )
            st.session_state[custom_key] = st.session_state[input_key]
        with pct_col:
            pct_display = reverse_map.get(st.session_state[pct_key], "P95")
            selected_pct = st.selectbox(
                f"{metric.upper()} percentile",
                percentile_options,
                index=percentile_options.index(pct_display),
                key=f"{metric}_pct_selector",
                label_visibility="collapsed",
            )
            st.session_state[pct_key] = percentile_map[selected_pct]
        st.caption(f"Recommended Range: {val_min:,} - {val_max:,} ms")

    render_slo_input("ttft", "TTFT (Time to First Token)", step=100)
    render_slo_input("itl", "ITL (Inter-Token Latency)", step=10)
    render_slo_input("e2e", "E2E (End-to-End Latency)", step=1000)

    curr_ttft = st.session_state.get("custom_ttft")
    curr_itl = st.session_state.get("custom_itl")
    curr_e2e = st.session_state.get("custom_e2e")

    if curr_ttft != prev_ttft or curr_itl != prev_itl or curr_e2e != prev_e2e:
        if st.session_state.get("recommendation_result"):
            st.session_state.recommendation_result = None
        st.session_state._last_ttft = curr_ttft
        st.session_state._last_itl = curr_itl
        st.session_state._last_e2e = curr_e2e


def _render_workload_profile(use_case, workload_profile, estimated_qps, qps, user_count):
    """Render workload profile: QPS input, token counts, peak multiplier, insights."""
    st.write("**Workload Profile**")

    prompt_tokens = workload_profile["prompt_tokens"]
    output_tokens = workload_profile["output_tokens"]
    peak_mult = workload_profile["peak_multiplier"]

    st.session_state.spec_prompt_tokens = prompt_tokens
    st.session_state.spec_output_tokens = output_tokens
    st.session_state.spec_peak_multiplier = peak_mult

    default_qps = estimated_qps
    st.write("**Throughput (Requests Per Second)**")
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
            <div style="font-size: 0.75rem; margin-top: 0.25rem; padding-left: 0;">Average output token length generated per request</div>
        </div>
        <div style="padding: 0.5rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.95rem; font-weight: 500;">Peak Multiplier</span>
                <span style="font-weight: 700; font-size: 1.1rem; padding: 3px 10px; border-radius: 4px;">{peak_mult}x</span>
            </div>
            <div style="font-size: 0.75rem; margin-top: 0.25rem; padding-left: 0;">Capacity buffer for traffic spikes</div>
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


def _render_accuracy_benchmarks(use_case):
    """Render the accuracy benchmark weights for the current use case."""
    st.write("**Accuracy Benchmarks**")

    datasets = TASK_DATASETS.get(use_case, TASK_DATASETS["chatbot_conversational"])

    datasets_html = '<div style="padding: 0.5rem;">'
    for name, weight, tooltip in datasets:
        datasets_html += f"""<div style="padding: 0.5rem 0; ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.95rem; font-weight: 500;">{name}</span>
                <span style="font-weight: 700; font-size: 0.95rem; padding: 3px 10px; border-radius: 4px;">{weight}%</span>
        </div>
            <div style="font-size: 0.75rem; margin-top: 0.25rem; padding-left: 0;">{tooltip}</div>
        </div>"""
    datasets_html += "</div>"
    datasets_html += '<div style="font-size: 0.75rem; font-weight: 600; margin-top: 0.5rem;">Weights from Artificial Analysis Intelligence Index</div>'
    st.markdown(datasets_html, unsafe_allow_html=True)


def _render_priorities():
    """Render priority dropdowns and weight inputs for accuracy, cost, latency."""
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
        default_weights = defaults_config.get(
            "weights", {"accuracy": 5, "cost": 4, "latency": 2, "complexity": 2}
        )
        weight = pw.get(priority_level, default_weights.get(dimension, 5))
        logger.info(
            f"get_weight_for_priority({dimension}, {priority_level}): pw={pw}, weight={weight}"
        )
        return int(weight)

    extraction = st.session_state.get("extraction_result", {})

    # Update priorities from new LLM extraction
    if st.session_state.get("new_extraction_available", False):
        st.session_state.accuracy_priority = extraction.get("accuracy_priority", "medium")
        st.session_state.cost_priority = extraction.get("cost_priority", "medium")
        st.session_state.latency_priority = extraction.get("latency_priority", "medium")
        st.session_state.weight_accuracy = get_weight_for_priority(
            "accuracy", st.session_state.accuracy_priority
        )
        st.session_state.weight_cost = get_weight_for_priority(
            "cost", st.session_state.cost_priority
        )
        st.session_state.weight_latency = get_weight_for_priority(
            "latency", st.session_state.latency_priority
        )
        st.session_state.new_extraction_available = False
        logger.info(
            f"Updated priorities from new extraction: accuracy={st.session_state.accuracy_priority}, "
            f"cost={st.session_state.cost_priority}, latency={st.session_state.latency_priority}"
        )

    # Initialize session state for priorities if not set (first load)
    if "accuracy_priority" not in st.session_state:
        st.session_state.accuracy_priority = extraction.get("accuracy_priority", "medium")
    if "cost_priority" not in st.session_state:
        st.session_state.cost_priority = extraction.get("cost_priority", "medium")
    if "latency_priority" not in st.session_state:
        st.session_state.latency_priority = extraction.get("latency_priority", "medium")

    # Initialize weights based on priorities
    if "weight_accuracy" not in st.session_state:
        st.session_state.weight_accuracy = get_weight_for_priority(
            "accuracy", st.session_state.accuracy_priority
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
        ("Accuracy", "accuracy"),
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
                st.session_state[f"weight_{state_key}"] = get_weight_for_priority(
                    state_key, new_priority_val
                )
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
        _render_accuracy_benchmarks(use_case)
    with col4:
        _render_priorities()


def render_slo_with_approval(extraction: dict):
    """Render SLO section with approval to proceed to recommendations."""
    use_case = extraction.get("use_case", "chatbot_conversational")
    user_count = extraction.get("user_count", 1000)

    render_slo_cards(use_case, user_count)

    # Validate SLO values
    slo_defaults = fetch_slo_defaults(use_case)
    if not slo_defaults:
        st.error(f"Failed to fetch SLO defaults for use case: {use_case}")
        return

    ttft_min = int(slo_defaults["ttft_ms"]["min"])
    ttft_max = int(slo_defaults["ttft_ms"]["max"])
    itl_min = int(slo_defaults["itl_ms"]["min"])
    itl_max = int(slo_defaults["itl_ms"]["max"])
    e2e_min = int(slo_defaults["e2e_ms"]["min"])
    e2e_max = int(slo_defaults["e2e_ms"]["max"])

    current_ttft = int(st.session_state.get("edit_ttft", ttft_max))
    current_itl = int(st.session_state.get("edit_itl", itl_max))
    current_e2e = int(st.session_state.get("edit_e2e", e2e_max))

    validation_errors = []
    if current_ttft < ttft_min or current_ttft > ttft_max:
        validation_errors.append(f"TTFT must be between {ttft_min:,}-{ttft_max:,}ms")
    if current_itl < itl_min or current_itl > itl_max:
        validation_errors.append(f"ITL must be between {itl_min}-{itl_max}ms")
    if current_e2e < e2e_min or current_e2e > e2e_max:
        validation_errors.append(f"E2E must be between {e2e_min:,}-{e2e_max:,}ms")

    is_valid = len(validation_errors) == 0

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not is_valid:
            st.markdown(
                f"""
            <div style="border: 1px solid rgba(239, 68, 68, 0.5);
                        border-radius: 8px; padding: 0.75rem; margin-bottom: 0.75rem; text-align: center;">
                <div style="color: #ef4444; font-weight: 600; font-size: 0.9rem;">⚠️ Invalid SLO Values</div>
                <div style="font-size: 0.8rem; margin-top: 0.25rem;">
                    {' • '.join(validation_errors)}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        if st.button(
            "Generate Recommendations",
            type="primary",
            use_container_width=True,
            key="generate_recs",
            disabled=not is_valid,
        ):
            st.session_state.slo_approved = True
            st.session_state._pending_tab = 2
            for _cat in ("balanced", "accuracy", "latency", "cost"):
                st.session_state[f"cat_idx_{_cat}"] = 0
            st.rerun()
