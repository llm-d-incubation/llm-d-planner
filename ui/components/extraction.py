"""Extraction display components for the Planner UI.

Extraction result display, approval workflow, and edit form.
"""

import streamlit as st
from api_client import fetch_catalog_model_ids, fetch_gpu_types


def _format_priorities(extraction: dict) -> str:
    """Format priority display from extraction data."""
    accuracy = extraction.get("accuracy_priority", "medium")
    cost = extraction.get("cost_priority", "medium")
    latency = extraction.get("latency_priority", "medium")

    parts = []
    if accuracy != "medium":
        parts.append(f"Accuracy: {accuracy.title()}")
    if cost != "medium":
        parts.append(f"Cost: {cost.title()}")
    if latency != "medium":
        parts.append(f"Latency: {latency.title()}")

    return ", ".join(parts) if parts else "Default"


def _format_models(extraction: dict) -> str:
    """Format preferred models display."""
    models = extraction.get("preferred_models", []) or st.session_state.get("preferred_models", [])
    if not models:
        return "Any"
    return ", ".join(models)


def render_extraction_result(extraction: dict, priority: str):
    """Render extraction results (read-only, after approval) with modify button."""
    st.subheader("Extracted Business Context")

    use_case = extraction.get("use_case", "unknown")
    use_case_display = use_case.replace("_", " ").title() if use_case else "Unknown"
    user_count = extraction.get("user_count", 0)
    hardware = extraction.get("hardware") or "Any GPU"
    priorities = _format_priorities(extraction)
    models = _format_models(extraction)

    st.markdown(
        f"**Use Case:** {use_case_display}  \n"
        f"**Expected Users:** {user_count:,}  \n"
        f"**Hardware:** {hardware}  \n"
        f"**Models:** {models}  \n"
        f"**Priorities:** {priorities}"
    )

    if st.button("Modify Business Context", use_container_width=False, key="modify_after_approve"):
        st.session_state.extraction_approved = False
        st.session_state.slo_approved = None
        st.session_state.recommendation_result = None
        st.rerun()


def render_extraction_with_approval(extraction: dict, models_df):
    """Render extraction results with YES/NO approval buttons."""
    st.subheader("Extracted Business Context")

    use_case = extraction.get("use_case", "unknown")
    use_case_display = use_case.replace("_", " ").title() if use_case else "Unknown"
    user_count = extraction.get("user_count", 0)
    hardware = extraction.get("hardware") or "Any GPU"
    priorities = _format_priorities(extraction)
    models = _format_models(extraction)

    st.markdown(
        f"**Use Case:** {use_case_display}  \n"
        f"**Expected Users:** {user_count:,}  \n"
        f"**Hardware:** {hardware}  \n"
        f"**Models:** {models}  \n"
        f"**Priorities:** {priorities}"
    )

    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
    with col1:
        if st.button(
            "Generate Specification",
            type="primary",
            width="stretch",
            key="approve_extraction",
        ):
            st.session_state.extraction_approved = True
            st.session_state._pending_tab = 1
            st.rerun()
    with col2:
        if st.button("Modify Extracted Context", width="stretch", key="edit_extraction"):
            st.session_state.extraction_approved = False
            st.rerun()
    with col3:
        if st.button("Start Over", width="stretch", key="restart"):
            st.session_state.extraction_result = None
            st.session_state.extraction_approved = None
            st.session_state.recommendation_result = None
            st.session_state.user_input = ""
            for key in [
                "accuracy_priority",
                "cost_priority",
                "latency_priority",
                "weight_accuracy",
                "weight_cost",
                "weight_latency",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


def render_extraction_edit_form(extraction: dict, models_df):
    """Render editable form for extraction correction."""
    st.subheader("Edit Business Context")
    st.info('Review and adjust the extracted values below, then click "Apply Changes" to continue.')

    use_cases = [
        "chatbot_conversational",
        "code_completion",
        "code_generation_detailed",
        "document_analysis_rag",
        "summarization_short",
        "long_document_summarization",
        "translation",
        "content_generation",
        "research_legal_analysis",
    ]
    use_case_labels = {
        "chatbot_conversational": "Chatbot / Conversational AI",
        "code_completion": "Code Completion (IDE autocomplete)",
        "code_generation_detailed": "Code Generation (full implementations)",
        "document_analysis_rag": "Document RAG / Q&A",
        "summarization_short": "Short Summarization (<10 pages)",
        "long_document_summarization": "Long Document Summarization (10+ pages)",
        "translation": "Translation",
        "content_generation": "Content Generation",
        "research_legal_analysis": "Research / Legal Analysis",
    }

    current_use_case = extraction.get("use_case", "chatbot_conversational")
    current_idx = use_cases.index(current_use_case) if current_use_case in use_cases else 0

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        new_use_case = st.selectbox(
            "Use Case",
            use_cases,
            index=current_idx,
            format_func=lambda x: use_case_labels.get(x, x),
            key="edit_use_case",
        )

        new_user_count = st.number_input(
            "User Count",
            min_value=1,
            max_value=1000000,
            value=extraction.get("user_count", 1000),
            step=100,
            key="edit_user_count",
        )

    with col2:
        priorities = ["balanced", "low_latency", "cost_saving", "high_accuracy", "high_throughput"]
        priority_labels = {
            "balanced": "Balanced",
            "low_latency": "Low Latency",
            "cost_saving": "Cost Saving",
            "high_accuracy": "High Accuracy",
            "high_throughput": "High Throughput",
        }
        current_priority = extraction.get("priority", "balanced")
        priority_idx = priorities.index(current_priority) if current_priority in priorities else 0

        new_priority = st.selectbox(
            "Priority",
            priorities,
            index=priority_idx,
            format_func=lambda x: priority_labels.get(x, x),
            key="edit_priority",
        )

        # GPU multi-select from catalog
        gpu_catalog = fetch_gpu_types()
        gpu_options = sorted(gpu_catalog.keys()) if gpu_catalog else []
        current_gpus = extraction.get("preferred_gpu_types", [])
        # Ensure current values are valid options
        valid_current_gpus = [g for g in current_gpus if g in gpu_options]

        new_gpu_types = st.multiselect(
            "Hardware (GPUs)",
            gpu_options,
            default=valid_current_gpus,
            key="edit_gpu_types",
            help="Select one or more GPU types, or leave empty for any GPU",
        )

    # Model selection
    st.markdown("**Model Preferences** (optional)")
    col_models, col_custom = st.columns(2, gap="medium")
    with col_models:
        # Get catalog models for multiselect
        catalog_model_ids = fetch_catalog_model_ids()
        current_models = extraction.get("preferred_models", [])
        catalog_current = [m for m in current_models if m in catalog_model_ids]

        new_catalog_models = st.multiselect(
            "Catalog Models",
            catalog_model_ids,
            default=catalog_current,
            key="edit_catalog_models",
            help="Select from approved model catalog",
        )
    with col_custom:
        custom_current = [m for m in current_models if m not in catalog_model_ids]
        new_custom_models_str = st.text_input(
            "Custom HuggingFace Model IDs",
            value=", ".join(custom_current),
            key="edit_custom_models",
            help="Comma-separated HuggingFace model IDs (e.g., meta-llama/Llama-3.3-70B-Instruct)",
        )

    # Merge catalog + custom models
    custom_models_list = [m.strip() for m in new_custom_models_str.split(",") if m.strip()]
    all_preferred_models = list(dict.fromkeys(new_catalog_models + custom_models_list))

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        if st.button("Apply Changes", type="primary", width="stretch", key="apply_edit"):
            edited = {
                "use_case": new_use_case,
                "user_count": new_user_count,
                "priority": new_priority,
                "hardware": ", ".join(new_gpu_types) if new_gpu_types else None,
                "preferred_gpu_types": new_gpu_types,
                "preferred_models": all_preferred_models,
            }
            st.session_state.preferred_models = all_preferred_models
            st.session_state.edited_extraction = edited
            # Apply edits back to extraction_result so the approval view shows updated values
            st.session_state.extraction_result.update(edited)
            st.session_state.used_priority = new_priority
            # Return to approval view (not approved yet) so buttons remain visible
            st.session_state.extraction_approved = None
            st.rerun()
    with col2:
        if st.button("🔙 Cancel", width="stretch", key="cancel_edit"):
            st.session_state.extraction_approved = None
            st.rerun()
