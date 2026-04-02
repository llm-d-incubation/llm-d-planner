"""Extraction display components for the Planner UI.

Extraction result display, approval workflow, and edit form.
"""

import streamlit as st


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


def render_extraction_result(extraction: dict, priority: str):
    """Render extraction results (read-only, after approval)."""
    st.subheader("Extracted Business Context")

    use_case = extraction.get("use_case", "unknown")
    use_case_display = use_case.replace("_", " ").title() if use_case else "Unknown"
    user_count = extraction.get("user_count", 0)
    hardware = extraction.get("hardware") or "Any GPU"
    priorities = _format_priorities(extraction)

    st.markdown(
        f"**Use Case:** {use_case_display}  \n"
        f"**Expected Users:** {user_count:,}  \n"
        f"**Hardware:** {hardware}  \n"
        f"**Priorities:** {priorities}"
    )


def render_extraction_with_approval(extraction: dict, models_df):
    """Render extraction results with YES/NO approval buttons."""
    st.subheader("Extracted Business Context")

    use_case = extraction.get("use_case", "unknown")
    use_case_display = use_case.replace("_", " ").title() if use_case else "Unknown"
    user_count = extraction.get("user_count", 0)
    hardware = extraction.get("hardware") or "Any GPU"
    priorities = _format_priorities(extraction)

    st.markdown(
        f"**Use Case:** {use_case_display}  \n"
        f"**Expected Users:** {user_count:,}  \n"
        f"**Hardware:** {hardware}  \n"
        f"**Priorities:** {priorities}"
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button(
            "Generate Specification",
            type="primary",
            use_container_width=True,
            key="approve_extraction",
        ):
            st.session_state.extraction_approved = True
            st.session_state._pending_tab = 1
            st.rerun()
    with col2:
        if st.button("Modify Extracted Context", use_container_width=True, key="edit_extraction"):
            st.session_state.extraction_approved = False
            st.rerun()
    with col3:
        if st.button("Start Over", use_container_width=True, key="restart"):
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

    col1, col2 = st.columns(2)
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

        hardware_options = ["Any GPU", "H100", "A100", "A10G", "L4", "T4"]
        current_hardware = extraction.get("hardware") or "Any GPU"
        hw_idx = (
            hardware_options.index(current_hardware) if current_hardware in hardware_options else 0
        )

        new_hardware = st.selectbox(
            "Hardware",
            hardware_options,
            index=hw_idx,
            key="edit_hardware",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Apply Changes", type="primary", use_container_width=True, key="apply_edit"):
            edited = {
                "use_case": new_use_case,
                "user_count": new_user_count,
                "priority": new_priority,
                "hardware": new_hardware if new_hardware != "Any GPU" else None,
            }
            st.session_state.edited_extraction = edited
            # Apply edits back to extraction_result so the approval view shows updated values
            st.session_state.extraction_result.update(edited)
            st.session_state.used_priority = new_priority
            # Return to approval view (not approved yet) so buttons remain visible
            st.session_state.extraction_approved = None
            st.rerun()
    with col2:
        if st.button("🔙 Cancel", use_container_width=True, key="cancel_edit"):
            st.session_state.extraction_approved = None
            st.rerun()
