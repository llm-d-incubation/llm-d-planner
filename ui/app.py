"""
Planner - E2E LLM Deployment Recommendation System

A Streamlit application for AI-powered LLM deployment recommendations.

Usage:
    streamlit run ui/app.py
"""

import logging
import sys
import time
from pathlib import Path

# Add ui/ to sys.path so modules can use flat imports
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from api_client import (
    extract_business_context,
    fetch_priority_weights,
    fetch_ranked_recommendations,
    load_206_models,
)
from components.deployment import render_deployment_tab
from components.deployment_management import render_deployment_management_tab
from components.dialogs import (
    show_category_dialog,
    show_full_table_dialog,
    show_winner_details_dialog,
)
from components.extraction import (
    render_extraction_edit_form,
    render_extraction_result,
    render_extraction_with_approval,
)
from components.recommendations import render_recommendation_result
from components.settings import render_configuration_tab
from components.slo import render_slo_with_approval
from state import init_session_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# MINIMAL CSS OVERRIDES
# =============================================================================
st.markdown(
    """
<style>
    /* Reduce top whitespace and align content with toolbar */
    .block-container { padding-top: 0 !important; }
    /* Transparent header so menu appears inline with content */
    header[data-testid="stHeader"] { background: transparent; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# SESSION STATE INIT
# =============================================================================
init_session_state()


# =============================================================================
# VISUAL COMPONENTS
# =============================================================================


def render_hero():
    """Render compact hero section with logo."""
    logo_col, title_col = st.columns([1, 11], vertical_alignment="center")
    with logo_col:
        st.image("ui/static/planner-logo.png", width=48)
    with title_col:
        st.title("Planner")
    st.caption(
        "AI-Powered LLM Deployment Recommendations — From Natural Language to Production in Seconds"
    )


# =============================================================================
# TAB FUNCTIONS
# =============================================================================


def render_use_case_input_tab(priority: str, models_df: pd.DataFrame):
    """Tab 1: Use case input interface."""

    def clear_dialog_states():
        """Clear all dialog and expanded states when starting a new use case."""
        st.session_state.show_full_table_dialog = False
        st.session_state.show_category_dialog = False
        st.session_state.show_winner_dialog = False
        st.session_state.show_options_list_expanded = False

    # Transfer pending input from button clicks before rendering the text_area widget
    if "pending_user_input" in st.session_state:
        st.session_state.user_input = st.session_state.pending_user_input
        del st.session_state.pending_user_input

    st.subheader("Describe your use case or select from 9 predefined scenarios")

    # Input area
    st.text_area(
        "Your requirements:",
        key="user_input",
        height=120,
        max_chars=2000,
        placeholder="Describe your LLM use case in natural language...\n\nExample: I need a chatbot for customer support with 30 users. Low latency is important, and we have H100 GPUs available.",
        label_visibility="collapsed",
    )

    # Row 1: 5 task buttons
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("Chat Completion", use_container_width=True, key="task_chat"):
            clear_dialog_states()
            st.session_state.pending_user_input = "Customer service chatbot for 30 users."
            st.rerun()

    with col2:
        if st.button("Code Completion", use_container_width=True, key="task_code"):
            clear_dialog_states()
            st.session_state.pending_user_input = "IDE code completion tool for 300 developers."
            st.rerun()

    with col3:
        if st.button("Document Q&A", use_container_width=True, key="task_rag"):
            clear_dialog_states()
            st.session_state.pending_user_input = (
                "Document Q&A system for enterprise knowledge base, 300 users."
            )
            st.rerun()

    with col4:
        if st.button("Summarization", use_container_width=True, key="task_summ"):
            clear_dialog_states()
            st.session_state.pending_user_input = (
                "News article summarization for 300 users, cost-effective solution preferred."
            )
            st.rerun()

    with col5:
        if st.button("Legal Analysis", use_container_width=True, key="task_legal"):
            clear_dialog_states()
            st.session_state.pending_user_input = (
                "Legal document analysis for 300 lawyers, accuracy is critical."
            )
            st.rerun()

    # Row 2: 4 more task buttons
    col6, col7, col8, col9 = st.columns(4)

    with col6:
        if st.button("Translation", use_container_width=True, key="task_trans"):
            clear_dialog_states()
            st.session_state.pending_user_input = (
                "Multi-language translation service for 300 users."
            )
            st.rerun()

    with col7:
        if st.button("Content Generation", use_container_width=True, key="task_content"):
            clear_dialog_states()
            st.session_state.pending_user_input = (
                "Content generation tool for marketing team, 300 users."
            )
            st.rerun()

    with col8:
        if st.button("Long Doc Summary", use_container_width=True, key="task_longdoc"):
            clear_dialog_states()
            st.session_state.pending_user_input = (
                "Long document summarization for research papers, 30 researchers, accuracy matters."
            )
            st.rerun()

    with col9:
        if st.button("Code Generation", use_container_width=True, key="task_codegen"):
            clear_dialog_states()
            st.session_state.pending_user_input = (
                "Full code generation tool for implementing features, 30 developers."
            )
            st.rerun()

    # Show character count
    char_count = len(st.session_state.user_input) if st.session_state.user_input else 0
    st.markdown(
        f'<div style="text-align: right; font-size: 0.75rem; margin-top: -0.5rem;">{char_count}/2000 characters</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1.5, 1, 2])
    with col1:
        analyze_disabled = (
            len(st.session_state.user_input.strip()) < 10 if st.session_state.user_input else True
        )
        analyze_clicked = st.button(
            "Analyze Use Case", type="primary", use_container_width=True, disabled=analyze_disabled
        )
        if (
            analyze_disabled
            and st.session_state.user_input
            and len(st.session_state.user_input.strip()) < 10
        ):
            st.caption("Please enter at least 10 characters")
    with col2:
        if st.button("Clear", use_container_width=True):
            for key in [
                "user_input",
                "extraction_result",
                "recommendation_result",
                "extraction_approved",
                "slo_approved",
                "edited_extraction",
                "custom_ttft",
                "custom_itl",
                "custom_e2e",
                "custom_qps",
                "used_priority",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.user_input = ""
            st.rerun()

    # Input validation before analysis
    if (
        analyze_clicked
        and st.session_state.user_input
        and len(st.session_state.user_input.strip()) >= 10
    ):
        # Reset workflow state
        st.session_state.extraction_approved = None
        st.session_state.slo_approved = None
        st.session_state.recommendation_result = None
        st.session_state.edited_extraction = None
        # Clear previous recommendation selection and deployment state
        st.session_state.deployment_selected_config = None
        st.session_state.deployment_selected_category = None
        st.session_state.deployment_yaml_generated = False
        st.session_state.deployment_yaml_files = {}
        st.session_state.deployment_id = None
        st.session_state.deployment_error = None

        progress_container = st.empty()
        with progress_container:
            progress_bar = st.progress(0, text="Initializing extraction...")

        try:
            progress_bar.progress(20, text="Analyzing input text...")
            extraction = extract_business_context(st.session_state.user_input)
            progress_bar.progress(80, text="Extraction complete!")

            if extraction:
                st.session_state.recommendation_result = None
                st.session_state.extraction_approved = None
                st.session_state.slo_approved = None
                st.session_state.edited_extraction = None
                st.session_state.ranked_response = None

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

                st.session_state.extraction_result = extraction

                priority_config = fetch_priority_weights()
                pw_map = priority_config.get("priority_weights", {}) if priority_config else {}
                defaults_cfg = priority_config.get("defaults", {}) if priority_config else {}
                default_weights = defaults_cfg.get(
                    "weights", {"accuracy": 5, "cost": 4, "latency": 2}
                )

                st.session_state.accuracy_priority = extraction.get("accuracy_priority", "medium")
                st.session_state.cost_priority = extraction.get("cost_priority", "medium")
                st.session_state.latency_priority = extraction.get("latency_priority", "medium")

                st.session_state.weight_accuracy = pw_map.get("accuracy", {}).get(
                    st.session_state.accuracy_priority, default_weights["accuracy"]
                )
                st.session_state.weight_cost = pw_map.get("cost", {}).get(
                    st.session_state.cost_priority, default_weights["cost"]
                )
                st.session_state.weight_latency = pw_map.get("latency", {}).get(
                    st.session_state.latency_priority, default_weights["latency"]
                )

                logger.info(
                    f"Initialized priorities from extraction: accuracy={st.session_state.accuracy_priority}, "
                    f"cost={st.session_state.cost_priority}, latency={st.session_state.latency_priority}"
                )
                logger.info(
                    f"Initialized weights: accuracy={st.session_state.weight_accuracy}, "
                    f"cost={st.session_state.weight_cost}, latency={st.session_state.weight_latency}"
                )

                st.session_state.new_extraction_available = True

                st.session_state.used_priority = extraction.get("priority", priority)
                st.session_state.detected_use_case = extraction.get(
                    "use_case", "chatbot_conversational"
                )
                progress_bar.progress(100, text="Ready!")
            else:
                st.error("Could not extract business context. Please try rephrasing your input.")
                progress_bar.empty()

        except Exception:
            st.error("An error occurred during analysis. Please try again.")
            progress_bar.empty()
        finally:
            time.sleep(0.5)
            progress_container.empty()

    # Get the priority that was actually used
    used_priority = st.session_state.get("used_priority", priority)

    # Show extraction with approval if extraction exists but not approved
    if st.session_state.extraction_result and st.session_state.extraction_approved is None:
        render_extraction_with_approval(st.session_state.extraction_result, models_df)
        return

    # If editing, show edit form
    if st.session_state.extraction_approved is False:
        render_extraction_edit_form(st.session_state.extraction_result, models_df)
        return

    # If approved, show message to proceed to Technical Specifications tab
    if st.session_state.extraction_approved is True:
        render_extraction_result(st.session_state.extraction_result, used_priority)

        st.markdown(
            """
        <div style="padding: 0.75rem 1rem; border-radius: 8px; font-size: 1rem; margin-bottom: 0.75rem; max-width: 50%;">
            <strong>Step 1 Complete</strong> · You can now go to Technical Specification
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_technical_specs_tab():
    """Tab 2: Technical Specification (SLO targets and workload settings)."""
    if not st.session_state.extraction_approved:
        st.markdown(
            """
        <div style="padding: 1.5rem; border-radius: 8px; text-align: center; ">
            <strong style="font-size: 1.1rem;">Complete Step 1 First</strong><br>
            <span style="font-size: 0.95rem; ">Go to the <strong>Define Use Case</strong> tab to describe your use case and approve the extraction.</span>
        </div>
        """,
            unsafe_allow_html=True,
        )
        return

    final_extraction = (
        st.session_state.edited_extraction or st.session_state.extraction_result or {}
    )

    render_slo_with_approval(final_extraction)

    if st.session_state.slo_approved is True:
        st.markdown(
            """
        <div style="padding: 0.75rem 1rem; border-radius: 8px; font-size: 1rem; margin-bottom: 0.75rem; max-width: 50%;">
            <strong>Step 2 Complete</strong> · You can now view Recommendations
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_results_tab(priority: str, models_df: pd.DataFrame):
    """Tab 3: Results display - Best Model Recommendations."""
    used_priority = st.session_state.get("used_priority", priority)

    if not st.session_state.slo_approved:
        if not st.session_state.extraction_approved:
            st.markdown(
                """
            <div style="padding: 1.5rem; border-radius: 8px; text-align: center; ">
                <strong style="font-size: 1.1rem;">Complete Previous Steps First</strong><br>
                <span style="font-size: 0.95rem; ">1. Go to <strong>Define Use Case</strong> tab to describe your use case<br>
                2. Then go to <strong>Technical Specification</strong> tab to set your SLO targets</span>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div style="padding: 1.5rem; border-radius: 8px; text-align: center; ">
                <strong style="font-size: 1.1rem;">Complete Step 2 First</strong><br>
                <span style="font-size: 0.95rem; ">Go to the <strong>Technical Specification</strong> tab to set your SLO targets and workload parameters.</span>
            </div>
            """,
                unsafe_allow_html=True,
            )
        return

    final_extraction = (
        st.session_state.edited_extraction or st.session_state.extraction_result or {}
    )

    # Always regenerate recommendations to ensure fresh SLO filtering
    st.session_state.recommendation_result = None
    st.session_state.pop("ranked_response", None)

    # Get all specification values from session state
    use_case = final_extraction.get("use_case", "chatbot_conversational")
    user_count = final_extraction.get("user_count", 1000)

    ttft_target = st.session_state.get("custom_ttft") or st.session_state.get("input_ttft") or 500
    itl_target = st.session_state.get("custom_itl") or st.session_state.get("input_itl") or 50
    e2e_target = st.session_state.get("custom_e2e") or st.session_state.get("input_e2e") or 10000

    qps_target = (
        st.session_state.get("spec_expected_qps") or st.session_state.get("custom_qps") or 1
    )

    prompt_tokens = st.session_state.get("spec_prompt_tokens", 512)
    output_tokens = st.session_state.get("spec_output_tokens", 256)

    percentile = st.session_state.get("slo_percentile", "p95")

    weights = {
        "accuracy": st.session_state.get("weight_accuracy", 5),
        "price": st.session_state.get("weight_cost", 4),
        "latency": st.session_state.get("weight_latency", 2),
        "complexity": 0,
    }

    preferred_gpu_types = final_extraction.get("preferred_gpu_types", [])

    with st.spinner(f"Scoring {len(models_df)} models with MCDM..."):
        recommendation = fetch_ranked_recommendations(
            use_case=use_case,
            user_count=user_count,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            expected_qps=float(qps_target),
            ttft_target_ms=int(ttft_target),
            itl_target_ms=int(itl_target),
            e2e_target_ms=int(e2e_target),
            weights=weights,
            include_near_miss=False,
            percentile=percentile,
            preferred_gpu_types=preferred_gpu_types,
        )

    if recommendation is None:
        st.error("Unable to get recommendations. Please ensure backend is running.")
    else:
        st.session_state.recommendation_result = recommendation

    if st.session_state.recommendation_result:
        render_recommendation_result(
            st.session_state.recommendation_result, used_priority, final_extraction
        )


# =============================================================================
# MAIN APP
# =============================================================================


def main():
    # Show dialogs if triggered (Streamlit only renders one at a time)
    if st.session_state.show_winner_dialog and st.session_state.balanced_winner is not None:
        show_winner_details_dialog()
    elif st.session_state.show_category_dialog:
        show_category_dialog()
    elif st.session_state.show_full_table_dialog:
        show_full_table_dialog()

    # Load models
    if st.session_state.models_df is None:
        st.session_state.models_df = load_206_models()
    models_df = st.session_state.models_df

    priority = "balanced"

    # Main Content - Compact hero
    render_hero()

    # Tab-based navigation (6 tabs)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Define Use Case",
            "Technical Specification",
            "Recommendations",
            "Deployment",
            "Deployment Management",
            "Configuration",
        ]
    )

    with tab1:
        render_use_case_input_tab(priority, models_df)

    with tab2:
        render_technical_specs_tab()

    with tab3:
        render_results_tab(priority, models_df)

    with tab4:
        render_deployment_tab()

    with tab5:
        render_deployment_management_tab()

    with tab6:
        render_configuration_tab()

    # Auto-switch to pending tab after rerun
    pending_tab = st.session_state.pop("_pending_tab", None)
    if pending_tab is not None:
        components.html(
            f"""<script>
            var tabs = window.parent.document.querySelectorAll('[role="tab"]');
            if (tabs.length > {pending_tab}) {{ tabs[{pending_tab}].click(); }}
            </script>""",
            height=0,
        )


main()
