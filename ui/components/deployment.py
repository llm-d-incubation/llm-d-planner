"""Deployment tab component for the Planner UI.

YAML generation, file display, and download.
"""

import io
import zipfile

import requests
import streamlit as st
from api_client import API_BASE_URL, check_cluster_status, deploy_to_cluster


def render_deployment_tab():
    """Tab 4: Deployment configuration and YAML generation."""
    st.subheader("Deployment Configuration")

    selected_config = st.session_state.get("deployment_selected_config")

    if not selected_config:
        st.info(
            "No configuration selected for deployment. Please select a configuration from the Recommendations tab."
        )
        st.markdown("""
        **How to select a configuration:**
        1. Go to the **Recommendations** tab
        2. Browse the available configurations
        3. Click **Select for Deployment** on your preferred configuration
        """)
        return

    # Display selected configuration summary
    model_name = selected_config.get("model_name", "Unknown Model")
    gpu_config = selected_config.get("gpu_config", {})
    gpu_type = gpu_config.get("gpu_type", "Unknown")
    gpu_count = gpu_config.get("gpu_count", 1)
    replicas = gpu_config.get("replicas", 1)

    st.markdown(
        f"""
    <div style="border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem;">
        <div style="display: flex; align-items: center; gap: 2rem; flex-wrap: wrap;">
            <div>
                <span style="font-size: 0.85rem;">Model:</span>
                <span style="font-weight: 600; font-size: 1rem; margin-left: 0.5rem;">{model_name}</span>
            </div>
            <div>
                <span style="font-size: 0.85rem;">GPU:</span>
                <span style="font-weight: 600; font-size: 1rem; margin-left: 0.5rem;">{gpu_count}x{gpu_type}</span>
            </div>
            <div>
                <span style="font-size: 0.85rem;">Replicas:</span>
                <span style="font-weight: 600; font-size: 1rem; margin-left: 0.5rem;">{replicas}</span>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Deploy to Kubernetes button
    _render_deploy_to_cluster_button(selected_config)

    st.markdown("---")

    # YAML Generation Section
    if not st.session_state.get("deployment_yaml_generated"):
        st.subheader("Deployment Files")
        st.warning("YAML files have not been generated yet.")

        if st.button("Generate YAML Files", type="primary", key="generate_yaml_btn"):
            with st.spinner("Generating deployment files..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/api/v1/deploy",
                        json={
                            "recommendation": selected_config,
                            "namespace": "default",
                        },
                        timeout=30,
                    )
                    response.raise_for_status()
                    result = response.json()

                    if result.get("success"):
                        deployment_id = result.get("deployment_id")
                        st.session_state.deployment_id = deployment_id

                        yaml_response = requests.get(
                            f"{API_BASE_URL}/api/v1/deployments/{deployment_id}/yaml",
                            timeout=10,
                        )
                        yaml_response.raise_for_status()
                        yaml_data = yaml_response.json()

                        st.session_state.deployment_yaml_files = yaml_data.get("files", {})
                        st.session_state.deployment_yaml_generated = True
                        st.rerun()
                    else:
                        st.session_state.deployment_error = result.get("message", "Unknown error")
                        st.rerun()

                except requests.exceptions.RequestException as e:
                    st.session_state.deployment_error = f"Failed to connect to backend API: {e}"
                    st.rerun()
                except Exception as e:
                    st.session_state.deployment_error = f"Error generating YAML files: {e}"
                    st.rerun()
    else:
        # Display generated YAML files
        deployment_id = st.session_state.get("deployment_id", "unknown")
        yaml_files = st.session_state.get("deployment_yaml_files", {})
        deployment_error = st.session_state.get("deployment_error")

        # Prepare zip file for download
        zip_buffer = None
        if yaml_files:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for filename, content in yaml_files.items():
                    zf.writestr(filename, content)
            zip_buffer.seek(0)

        header_col1, header_col2 = st.columns([0.07, 0.93])
        with header_col1:
            if zip_buffer:
                st.download_button(
                    label="⬇",
                    data=zip_buffer.getvalue(),
                    file_name=f"{deployment_id}.zip",
                    mime="application/zip",
                    key="download_all_yaml",
                    help="Download all YAML files as zip",
                )
            else:
                st.markdown("⬇", unsafe_allow_html=True)
        with header_col2:
            if deployment_error:
                st.markdown(
                    '<h3 style="font-weight: 700; margin: 0;">Deployment Files <span style="color: #ef4444; font-weight: 700;">(ERROR!)</span></h3>',
                    unsafe_allow_html=True,
                )
                st.error(deployment_error)
            else:
                st.markdown(
                    f'<h3 style="font-weight: 700; margin: 0;">Deployment Files <span style="font-weight: 400; font-size: 0.8rem;">({deployment_id})</span></h3>',
                    unsafe_allow_html=True,
                )

        if yaml_files:
            file_order = ["inferenceservice", "autoscaling", "servicemonitor"]
            file_labels = {
                "inferenceservice": "InferenceService (KServe)",
                "autoscaling": "Autoscaling (HPA)",
                "servicemonitor": "ServiceMonitor (Prometheus)",
            }

            for file_key in file_order:
                matching_content = None
                for filename, content in yaml_files.items():
                    if file_key in filename.lower():
                        matching_content = content
                        break

                if matching_content:
                    label = file_labels.get(file_key, file_key)
                    with st.expander(f"{label}", expanded=False):
                        st.code(matching_content, language="yaml")

        # Regenerate button
        if st.button("Regenerate YAML Files", key="regenerate_yaml_btn"):
            st.session_state.deployment_yaml_generated = False
            st.session_state.deployment_yaml_files = {}
            st.session_state.deployment_id = None
            st.session_state.deployment_error = None
            st.rerun()


def _render_deploy_to_cluster_button(selected_config: dict):
    """Render the Deploy to Kubernetes button with cluster status check on click."""
    already_deployed = st.session_state.get("deployed_to_cluster", False)

    if st.button(
        "Deployed to Cluster" if already_deployed else "Deploy to Kubernetes",
        use_container_width=True,
        type="primary",
        disabled=already_deployed,
        help="Already deployed to cluster"
        if already_deployed
        else "Deploy to Kubernetes cluster (YAML auto-generated)",
        key="deploy_to_cluster_btn",
    ):
        # Check cluster accessibility when the user clicks
        with st.spinner("Checking cluster connectivity..."):
            status = check_cluster_status()
        if not status.get("accessible", False):
            st.error(
                "Kubernetes cluster is not accessible. Please ensure the cluster is running and try again."
            )
            return

        with st.spinner("Deploying to Kubernetes cluster..."):
            result = deploy_to_cluster(selected_config)

        if result.get("success") or result.get("deployment_id"):
            st.session_state.deployed_to_cluster = True
            st.session_state.deployment_id = result.get("deployment_id")

            # Store YAML files if returned
            files = result.get("files", {})
            if files:
                st.session_state.deployment_yaml_files = files
                st.session_state.deployment_yaml_generated = True

            st.success(
                f"Successfully deployed to cluster! Deployment ID: `{result.get('deployment_id')}`"
            )

            deployment_result = result.get("deployment_result", {})
            for applied_file in deployment_result.get("applied_files", []):
                st.markdown(f"- {applied_file['file']}")
        else:
            st.error(f"Deployment failed: {result.get('message', 'Unknown error')}")
