"""Session state defaults and initialization for the Planner UI."""

from typing import Any

import streamlit as st

SESSION_DEFAULTS: dict[str, Any] = {
    # Core workflow state
    "extraction_result": None,
    "recommendation_result": None,
    "user_input": "",
    "models_df": None,
    "selected_model": None,
    # Workflow approval (None = pending, True = approved, False = editing)
    "extraction_approved": None,
    "slo_approved": None,
    "edited_extraction": None,
    # SLO values
    "edit_slo": False,
    "custom_ttft": None,
    "custom_itl": None,
    "custom_e2e": None,
    "custom_qps": None,
    "slo_percentile": "p95",
    "ttft_percentile": "p95",
    "itl_percentile": "p95",
    "e2e_percentile": "p95",
    "include_near_miss": False,
    # Category expansion
    "expanded_categories": set(),
    # Dialog state
    "show_winner_dialog": False,
    "balanced_winner": None,
    "winner_priority": "balanced",
    "winner_extraction": {},
    "show_category_dialog": False,
    "explore_category": "balanced",
    "show_full_table_dialog": False,
    "show_options_list_expanded": False,
    # Use case tracking
    "detected_use_case": "chatbot_conversational",
    # Top-5 lists per category
    "top5_balanced": [],
    "top5_accuracy": [],
    "top5_latency": [],
    "top5_cost": [],
    "top5_simplest": [],
    # Category card navigation (index into top-5 lists)
    "cat_idx_balanced": 0,
    "cat_idx_accuracy": 0,
    "cat_idx_latency": 0,
    "cat_idx_cost": 0,
    # Deployment tab
    "deployment_selected_config": None,
    "deployment_yaml_files": {},
    "deployment_id": None,
    "deployment_yaml_generated": False,
    # Cluster / deployment management
    "cluster_accessible": None,
    "deployed_to_cluster": False,
    "selected_deployment": None,
}
# Ranking weights are initialized in the PRIORITIES section of the Tech Spec tab
# using values from priority_weights.json. Do NOT hardcode defaults here.


def init_session_state():
    """Initialize session state with defaults for any missing keys."""
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default
