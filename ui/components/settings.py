"""Settings / Configuration tab component.

Contains benchmark database management controls;
structured to support additional configuration sections.
"""

import streamlit as st
from api_client import (
    fetch_db_status,
    fetch_deployment_mode,
    reset_database,
    update_deployment_mode,
    upload_benchmarks,
)


_TAB_INDEX = 4  # Configuration is the 5th tab (0-indexed)


def render_configuration_tab():
    """Render the Configuration tab with deployment mode and database management."""
    # --- Deployment Mode ---
    st.subheader("Deployment Mode")

    current_mode = fetch_deployment_mode()
    modes = ["Production", "Simulator"]
    current_index = 1 if current_mode == "simulator" else 0

    selected = st.radio(
        "YAML generation target",
        modes,
        index=current_index,
        horizontal=True,
        key="deployment_mode_radio",
        help="Production uses real vLLM with GPU resources. "
        "Simulator uses the vLLM simulator (no GPU required).",
    )

    selected_mode = selected.lower()
    if current_mode and selected_mode != current_mode:
        result = update_deployment_mode(selected_mode)
        if result:
            st.success(f"Deployment mode set to **{selected}**.")
        else:
            st.error("Failed to update deployment mode.")

    st.divider()

    # --- Benchmark Database ---
    st.subheader("Benchmark Database")

    # Reserve space for stats — populated after actions so data is always fresh
    status_area = st.container()

    st.divider()

    # Track whether an action produced updated stats
    action_status = None

    # --- Upload ---
    st.markdown("**Upload Benchmarks**")
    st.caption("Upload a JSON file with a top-level `benchmarks` array. Duplicates are skipped.")

    # Counter-based key resets the file uploader after a successful load
    upload_counter = st.session_state.get("_upload_counter", 0)
    uploaded = st.file_uploader(
        "Choose benchmark JSON file",
        type=["json"],
        key=f"settings_file_upload_{upload_counter}",
        label_visibility="collapsed",
    )

    # Clear any stored message when the user selects a new file
    if uploaded is not None:
        st.session_state.pop("_load_msg", None)

    if uploaded is not None and st.button("Load DB", key="settings_upload_btn", type="primary"):
        with st.spinner("Loading..."):
            result = upload_benchmarks(uploaded.getvalue(), uploaded.name)
        if result and result.get("success"):
            msg = (
                f"Processed {result.get('records_in_file', '?')} records from "
                f"{result.get('filename', 'file')} (duplicates skipped). "
                f"Database now has {result.get('total_benchmarks', '?')} unique benchmarks."
            )
            st.session_state["_load_msg"] = ("success", msg)
            # Increment counter so the file uploader resets on next rerun
            st.session_state["_upload_counter"] = upload_counter + 1
            st.session_state["_pending_tab"] = _TAB_INDEX
            st.rerun()
        else:
            msg = result.get("message", "Unknown error") if result else "No response from server"
            st.session_state["_load_msg"] = ("error", f"Load failed: {msg}")

    # Show persisted load message (survives the rerun that clears the file uploader)
    load_msg = st.session_state.get("_load_msg")
    if load_msg:
        level, text = load_msg
        if level == "success":
            st.success(text)
        else:
            st.error(text)

    st.divider()

    # --- Reset ---
    st.markdown("**Reset Database**")
    if st.button("Reset Database", key="settings_reset_btn", type="secondary"):
        st.session_state["_pending_tab"] = _TAB_INDEX
        with st.spinner("Resetting..."):
            result = reset_database()
        if result and result.get("success"):
            st.success("Database has been reset. All benchmark data removed.")
            action_status = result
            # Clear any stale load message
            st.session_state.pop("_load_msg", None)
        else:
            msg = result.get("message", "Unknown error") if result else "No response from server"
            st.error(f"Reset failed: {msg}")

    # --- Populate status area (after actions, so stats reflect mutations) ---
    status = action_status if action_status else fetch_db_status()
    with status_area:
        if status and status.get("success"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Benchmarks", status.get("total_benchmarks", 0))
            c2.metric("Models", status.get("num_models", 0))
            c3.metric("Hardware Types", status.get("num_hardware_types", 0))

            traffic = status.get("traffic_distribution", [])
            if traffic:
                st.caption(
                    "Traffic profiles: "
                    + ", ".join(f"({t['prompt_tokens']}, {t['output_tokens']})" for t in traffic)
                )
        else:
            st.warning("Could not connect to database.")
