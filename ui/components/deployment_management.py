"""Deployment Management tab component for the NeuralNav UI.

Lists cluster deployments, shows K8s status, provides delete and inference testing.
"""

import json
import subprocess
import time

import pandas as pd
import streamlit as st
from api_client import delete_deployment, get_k8s_status, load_all_deployments


def render_deployment_management_tab():
    """Tab 5: Deployment Management — list, inspect, test, and delete cluster deployments."""
    st.subheader("Cluster Deployments")

    all_deployments = load_all_deployments()

    if all_deployments is None:
        st.warning("Could not connect to cluster to list deployments")
        st.info(
            "**Troubleshooting:**\n"
            "- Ensure Kubernetes cluster is running (e.g., KIND cluster)\n"
            "- Check that kubectl can access the cluster: `kubectl cluster-info`\n"
            "- Verify backend API is running on http://localhost:8000"
        )
        return

    if len(all_deployments) == 0:
        st.info(
            "**No deployments found in cluster**\n\n"
            "To create a deployment:\n"
            "1. Go to the **Recommendations** tab and select a configuration\n"
            "2. Go to the **Deployment** tab\n"
            "3. Click **Deploy to Kubernetes**"
        )
        return

    # Deployments table
    st.markdown(f"**Found {len(all_deployments)} deployment(s)**")

    table_data = []
    for dep in all_deployments:
        status = dep.get("status", {})
        pods = dep.get("pods", [])
        ready = status.get("ready", False)
        table_data.append({
            "Status": "Ready" if ready else "Pending",
            "Name": dep["deployment_id"],
            "Pods": len(pods),
            "Ready": "Yes" if ready else "No",
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Select Deployment to Manage")

    # Deployment selector
    deployment_options = {d["deployment_id"]: d for d in all_deployments}
    deployment_ids = list(deployment_options.keys())

    if (
        st.session_state.selected_deployment is None
        or st.session_state.selected_deployment not in deployment_ids
    ):
        st.session_state.selected_deployment = deployment_ids[0] if deployment_ids else None

    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox(
            "Choose a deployment:",
            deployment_ids,
            index=deployment_ids.index(st.session_state.selected_deployment)
            if st.session_state.selected_deployment in deployment_ids
            else 0,
            key="deployment_selector_mgmt",
        )
        st.session_state.selected_deployment = selected

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Refresh", use_container_width=True, key="refresh_mgmt"):
            st.rerun()

    if not st.session_state.selected_deployment:
        return

    deployment_info = deployment_options[st.session_state.selected_deployment]

    st.markdown("---")
    _render_deployment_controls(deployment_info)
    st.markdown("---")
    _render_k8s_status(deployment_info)
    st.markdown("---")
    _render_inference_testing(deployment_info)


def _render_deployment_controls(deployment_info: dict):
    """Render deployment status metrics and delete button."""
    deployment_id = deployment_info["deployment_id"]
    status = deployment_info.get("status", {})
    pods = deployment_info.get("pods", [])

    st.markdown("#### Deployment Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        ready_status = "Ready" if status.get("ready") else "Pending"
        st.metric("Status", ready_status)

    with col2:
        st.metric("Pods", len(pods))

    with col3:
        confirm_key = f"confirm_delete_{deployment_id}"
        if st.button(
            "Delete Deployment",
            use_container_width=True,
            type="secondary",
            key=f"delete_btn_{deployment_id}",
        ):
            if st.session_state.get(confirm_key):
                with st.spinner("Deleting deployment..."):
                    result = delete_deployment(deployment_id)
                if result.get("success"):
                    st.success(f"Deleted {deployment_id}")
                    if st.session_state.deployment_id == deployment_id:
                        st.session_state.deployment_id = None
                        st.session_state.deployed_to_cluster = False
                    st.session_state[confirm_key] = False
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Failed to delete: {result.get('message', 'Unknown error')}")
                    st.session_state[confirm_key] = False
            else:
                st.session_state[confirm_key] = True
                st.warning(f"Click again to confirm deletion of {deployment_id}")


def _render_k8s_status(deployment_info: dict):
    """Render Kubernetes status for a deployment."""
    deployment_id = deployment_info["deployment_id"]
    status = deployment_info.get("status", {})
    pods = deployment_info.get("pods", [])

    st.markdown("#### Kubernetes Status")

    if status.get("exists"):
        st.success(f"InferenceService: **{deployment_id}**")
        st.markdown(f"**Ready:** {'Yes' if status.get('ready') else 'Not yet'}")

        if status.get("url"):
            st.markdown(f"**URL:** `{status['url']}`")

        with st.expander("Resource Conditions"):
            for condition in status.get("conditions", []):
                status_icon = "+" if condition.get("status") == "True" else "-"
                st.markdown(
                    f"[{status_icon}] **{condition.get('type')}**: {condition.get('message', 'N/A')}"
                )
    else:
        st.warning(f"InferenceService not found: {status.get('error', 'Unknown error')}")

    if pods:
        st.markdown(f"**Pods:** {len(pods)} pod(s)")
        with st.expander("Pod Details"):
            for pod in pods:
                st.markdown(f"**{pod.get('name')}**")
                st.markdown(f"- Phase: {pod.get('phase')}")
                node_name = pod.get("node_name") or "Not assigned"
                st.markdown(f"- Node: {node_name}")
    else:
        st.info("No pods found yet (may still be creating)")


def _render_inference_testing(deployment_info: dict):
    """Render inference testing UI for a deployment."""
    deployment_id = deployment_info["deployment_id"]
    status = deployment_info.get("status", {})

    st.markdown("#### Inference Testing")

    if not status.get("ready"):
        st.info(
            "Deployment not ready yet. Inference testing will be available once the service is ready."
        )
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        test_prompt = st.text_area(
            "Test Prompt",
            value="Write a Python function that calculates the fibonacci sequence.",
            height=100,
            key=f"test_prompt_{deployment_id}",
        )

    with col2:
        max_tokens = st.number_input(
            "Max Tokens",
            value=150,
            min_value=10,
            max_value=500,
            key=f"max_tokens_{deployment_id}",
        )
        temperature = st.slider(
            "Temperature", 0.0, 2.0, 0.7, 0.1, key=f"temperature_{deployment_id}"
        )

    if st.button(
        "Send Test Request",
        use_container_width=True,
        key=f"test_button_{deployment_id}",
    ):
        _run_inference_test(deployment_id, test_prompt, max_tokens, temperature)

    with st.expander("How Inference Testing Works"):
        st.markdown(
            "**Process:**\n"
            "1. Uses `kubectl port-forward` to connect to the InferenceService\n"
            "2. Sends a POST request to `/v1/completions` (OpenAI-compatible API)\n"
            "3. Displays the response and metrics\n\n"
            "**Note:** This temporarily forwards port 8080 on your local machine to the service."
        )


def _run_inference_test(deployment_id: str, prompt: str, max_tokens: int, temperature: float):
    """Execute an inference test via kubectl port-forward."""
    with st.spinner("Sending inference request..."):
        service_name = f"{deployment_id}-predictor"
        st.info(f"Connecting to service: `{service_name}`")

        port_forward_proc = subprocess.Popen(
            ["kubectl", "port-forward", f"svc/{service_name}", "8080:80"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        time.sleep(3)

        if port_forward_proc.poll() is not None:
            pf_stdout, pf_stderr = port_forward_proc.communicate()
            st.error("Port-forward failed to start")
            st.code(
                f"stdout: {pf_stdout.decode()}\nstderr: {pf_stderr.decode()}",
                language="text",
            )
            return

        try:
            start_time = time.time()

            curl_cmd = [
                "curl", "-s", "-X", "POST",
                "http://localhost:8080/v1/completions",
                "-H", "Content-Type: application/json",
                "-d", json.dumps({
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }),
            ]

            with st.expander("Debug Info"):
                st.code(" ".join(curl_cmd), language="bash")

            result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=30)
            elapsed_time = time.time() - start_time

            if result.returncode == 0 and result.stdout:
                try:
                    response_data = json.loads(result.stdout)
                    st.success(f"Response received in {elapsed_time:.2f}s")

                    st.markdown("**Generated Response:**")
                    response_text = response_data.get("choices", [{}])[0].get("text", "")
                    st.code(response_text, language=None)

                    usage = response_data.get("usage", {})
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Prompt Tokens", usage.get("prompt_tokens", 0))
                    with c2:
                        st.metric("Completion Tokens", usage.get("completion_tokens", 0))
                    with c3:
                        st.metric("Total Tokens", usage.get("total_tokens", 0))

                    st.metric("Total Latency", f"{elapsed_time:.2f}s")

                    with st.expander("Raw API Response"):
                        st.json(response_data)

                except json.JSONDecodeError as e:
                    st.error(f"Failed to parse JSON response: {e}")
                    st.code(result.stdout, language="text")
            else:
                st.error(f"Request failed (return code: {result.returncode})")
                if result.stdout:
                    st.code(result.stdout, language="text")
                if result.stderr:
                    st.code(result.stderr, language="text")

        except subprocess.TimeoutExpired:
            st.error("Request timed out (30s). The model may still be starting up.")
        finally:
            port_forward_proc.terminate()
            try:
                port_forward_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                port_forward_proc.kill()
