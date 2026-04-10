"""
Main Page
"""

import json
from pathlib import Path
from typing import Any

import streamlit as st
import util
from api_client import (
    fetch_capacity_planner_calculate,
    fetch_capacity_planner_model_info,
    fetch_gpu_types,
)
from matplotlib import pyplot as plt


def _cached_calculate(
    model_name: str,
    max_model_len: int | None = None,
    batch_size: int = 1,
    gpu_memory: float | None = None,
    gpu_mem_util: float = 0.9,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
) -> dict[str, Any] | None:
    """Call fetch_capacity_planner_calculate with session-state caching.

    Each unique combination of parameters is fetched at most once per Streamlit
    render cycle, avoiding redundant HuggingFace round-trips when the same
    parameters appear in multiple UI sections.
    """
    cache_key = (model_name, max_model_len, batch_size, gpu_memory, gpu_mem_util, tp, pp, dp)
    cache: dict[tuple, dict[str, Any]] = st.session_state.setdefault("_calc_cache", {})
    if cache_key not in cache:
        result: dict[str, Any] | None = fetch_capacity_planner_calculate(
            model_name,
            max_model_len=max_model_len,
            batch_size=batch_size,
            gpu_memory=gpu_memory,
            gpu_mem_util=gpu_mem_util,
            tp=tp,
            pp=pp,
            dp=dp,
        )
        if result is not None:
            cache[cache_key] = result
        return result
    return cache[cache_key]


def _load_gpu_specs_fallback() -> dict[str, dict]:
    """Load GPU specs from gpu_catalog.json when the backend API is unavailable."""
    catalog_path = (
        Path(__file__).parent.parent.parent / "data" / "configuration" / "gpu_catalog.json"
    )
    data = json.loads(catalog_path.read_text())
    return {g["gpu_type"]: g for g in data.get("gpu_types", [])}


def update_gpu_spec():
    """
    Update user selected GPU spec in session state
    """
    st.session_state["scenario"].gpu_spec = st.session_state["gpu_spec"][
        st.session_state["selected_gpu_spec"]
    ]


@st.dialog("Register a new accelerator")
def register_new_accelerator():
    """
    Dialog to register a new accelerator type
    """
    acc_name = st.text_input("Name", placeholder="NVIDIA-A100-40GB")
    acc_mem = st.number_input("Memory (GB)", min_value=1, step=1)

    if st.button("Register", width="stretch") and acc_name:
        gpu_specs[acc_name] = {"name": acc_name, "memory": acc_mem}
        st.rerun()


def model_specification():
    """
    Get model inputs like model name, precision
    """
    with st.container(border=True):
        st.write("**Model Specification**")
        st.caption("Select a model and we'll figure out what hardware you need to serve it.")

        col1, col2 = st.columns(2)

        selected_model = col1.text_input(
            "Model (Hugging Face format)",
            key=util.SELECTED_MODEL_KEY,
            on_change=util.update_scenario,
            args=[util.SELECTED_MODEL_KEY, "model_name"],
        )

        # Fetch when user changes model
        if selected_model and selected_model != st.session_state.get("_last_model_id"):
            # Reset the top-level pre-fetch guard so the new model can be attempted there too
            st.session_state.pop("_model_info_fetch_attempted", None)
            model_info = fetch_capacity_planner_model_info(selected_model)
            if model_info:
                st.session_state["model_info_response"] = model_info
                st.session_state["_last_model_id"] = selected_model
            else:
                st.session_state.pop("model_info_response", None)

        model_info = st.session_state.get("model_info_response")
        if not model_info:
            if selected_model:
                col1.warning("Loading model information...")
            return None

        model_gpu_memory_req = model_info["model_memory_gb"]
        col1.info(f"Size of model in memory: ~{util.pretty_round(model_gpu_memory_req)} GB")

        with col2.expander("See how model size is calculated below"):
            st.write(
                "The model size is calculated based on the number of parameters and the precision they are stored in."
            )

            if model_info["quantization"]["is_quantized"]:
                st.write(f"Quantization method: `{model_info['quantization']['quant_method']}`")

            breakdown = model_info["memory_breakdown"]
            df_data = {
                "Data type": [r["dtype"] for r in breakdown],
                "Quantized data type": [r["quantized_dtype"] for r in breakdown],
                "Size in bytes": [r["bytes_per_param"] for r in breakdown],
                "Number of parameters": [r["num_parameters"] for r in breakdown],
                "Memory in GB (params x bytes)": [r["memory_gb"] for r in breakdown],
            }
            if all(r["dtype"] == r["quantized_dtype"] for r in breakdown):
                del df_data["Quantized data type"]
            st.dataframe(df_data, hide_index=True)


def parallelism_specification():
    """
    Parallelism configuration
    """
    user_scenario = st.session_state[util.USER_SCENARIO_KEY]
    model_info = st.session_state.get("model_info_response")
    if not model_info:
        return None

    arch = model_info["architecture"]
    possible_tp_sizes = model_info["possible_tp_values"]

    with st.container(border=True):
        st.write("**Parallelism Configuration**")
        st.caption("Parallelism determines the number of GPUs required.")

        col1, col2 = st.columns(2)

        tp_size = col1.selectbox(
            "Tensor parallel size (shard model weights across GPUs)",
            key=util.SELECTED_TP_SIZE_KEY,
            options=possible_tp_sizes,
            index=possible_tp_sizes.index(user_scenario.tp_size)
            if user_scenario.tp_size in possible_tp_sizes
            else 0,
            help=f"Must be divisible by the number of attention heads (`{arch['num_attention_heads']}` for this model)",
            on_change=util.on_update_parallelism,
            args=[util.SELECTED_TP_SIZE_KEY, "tp_size"],
        )

        pp_size = col2.number_input(
            "Pipeline parallel size (shard layers across GPUs)",
            key=util.SELECTED_PP_SIZE_KEY,
            min_value=1,
            max_value=arch["num_hidden_layers"],
            value=user_scenario.pp_size,
            help=f"This number is capped by the number of hidden layers (`{arch['num_hidden_layers']}` for this model). vLLM handles uneven splits, see the [documentation](https://docs.vllm.ai/en/latest/api/vllm/distributed/index.html#vllm.distributed.get_pp_indices)",
            on_change=util.on_update_parallelism,
            args=[util.SELECTED_PP_SIZE_KEY, "pp_size"],
        )

        dp_size = col1.number_input(
            "Data parallel size (replicas of model)",
            key=util.SELECTED_DP_SIZE_KEY,
            min_value=1,
            value=user_scenario.dp_size,
            on_change=util.on_update_parallelism,
            args=[util.SELECTED_DP_SIZE_KEY, "dp_size"],
        )

        # Enable EP
        is_moe_model = arch["is_moe"]
        help_text = "EP is not available as an option for non-MoE models."
        if is_moe_model:
            help_text = """Instead of the traditional single feed forward layer in transformers, mixture of expert (MoE) models exercise a parallel feed-forward neural-network layers where a number of selected "experts" are activated for each token ([citation](https://nvidia.github.io/TensorRT-LLM/advanced/expert-parallelism.html)).

Tensor parallelism splits expert weights across GPUs. Expert parallelism splits incoming token's hidden state across GPUs. In vLLM, enabling data parallelism on MoE models essentially achieves the latter purpose.
"""

        enable_ep = col2.toggle(
            "Enable expert parallelism",
            value=user_scenario.enable_ep,
            disabled=not is_moe_model,
            help=help_text,
            key=util.SELECTED_ENABLE_EP_KEY,
            on_change=util.update_scenario,
            args=[util.SELECTED_ENABLE_EP_KEY, "enable_ep"],
        )

        if enable_ep and arch["num_experts"]:
            total_experts = arch["num_experts"]
            ep_size = tp_size * dp_size
            experts_per_ep = total_experts / ep_size if ep_size > 0 else 0
            experts_per_ep_str = round(experts_per_ep)

            col2.info(f"""Total number of experts: {total_experts}

`EP size = (TP x DP) = {ep_size}`, meaning each group will get `{total_experts} / {ep_size} = {experts_per_ep_str}` experts per group.
""")
            if experts_per_ep < 1:
                col2.warning(
                    "Since some EP groups will get 0 expert, this is an under-utilization of GPU resources. We recommend decreasing TP or DP for better use of your accelerators."
                )

            if experts_per_ep % 1 != 0:
                col2.caption(
                    "The total number of experts is not divisible by EP size you selected. However, vLLM handles uneven split of experts (see this [PR](https://github.com/vllm-project/vllm/pull/21497)), so some EP groups will have fewer experts than others."
                )

        total_gpus = (tp_size or 1) * (pp_size or 1) * (dp_size or 1)
        st.info(f"GPUs required (`TP x PP x DP`): `{total_gpus}`")


def workload_specification():
    """
    Estimate total memory needed for KV cache
    """
    user_scenario = st.session_state[util.USER_SCENARIO_KEY]
    model_info = st.session_state.get("model_info_response")
    if not model_info:
        return None

    arch = model_info["architecture"]

    with st.container(border=True):
        st.write("**Workload Characteristics**")
        st.caption(
            f"Estimate KV cache memory requirements for the selected model based on workload. Note that the model uses data type of `{arch['inference_dtype']}` for KV cache during inference."
        )

        col1, col2 = st.columns(2)

        model_max_context_len = arch["max_context_len"]

        auto_max_model_len_checked = col1.checkbox(
            "Auto-calculate max model len",
            key=util.SELECTED_AUTO_MAX_MODEL_LEN_KEY,
            on_change=util.on_update_auto_max_model_len,
        )

        if auto_max_model_len_checked:
            calc_result = _cached_calculate(
                user_scenario.model_name,
                max_model_len=-1,
                gpu_memory=user_scenario.get_gpu_memory(gpu_specs),
                gpu_mem_util=user_scenario.gpu_mem_util,
                tp=user_scenario.tp_size,
                pp=user_scenario.pp_size,
                dp=user_scenario.dp_size,
            )
            if calc_result is None:
                return None
            for w in calc_result.get("warnings", []):
                col1.warning(w)
            auto_val = calc_result["input_parameters"]["max_model_len"]
            if auto_val <= 1:
                col1.error("Model does not fit in available GPU memory.")
            else:
                col1.info(f"Auto-calculated max model len: **{auto_val:,}** tokens")
            user_scenario.max_model_len = auto_val
        else:
            col1.number_input(
                f"Max model len (max model context length is: {model_max_context_len})",
                min_value=1,
                max_value=model_max_context_len,
                value=user_scenario.max_model_len,
                key=util.SELECTED_MAX_MODEL_LEN_KEY,
                on_change=util.on_update_max_model_len,
            )
        col1.caption(
            "Maximum model length for the model: how many tokens (input + output) the model can process. "
            "Higher max model length means fewer concurrent requests can be served, "
            "because for the same GPU memory available for KV cache, "
            "each request requires more memory allocation. "
        )

        col2.number_input(
            "Input the max number of concurrent requests to process",
            min_value=0,
            step=1,
            key=util.SELECTED_CONCURRENCY_KEY,
            value=user_scenario.concurrency,
            on_change=util.update_scenario,
            args=[util.SELECTED_CONCURRENCY_KEY, "concurrency"],
        )

        calc_result = _cached_calculate(
            user_scenario.model_name,
            max_model_len=user_scenario.max_model_len,
            batch_size=user_scenario.concurrency,
            gpu_memory=user_scenario.get_gpu_memory(gpu_specs),
            gpu_mem_util=user_scenario.gpu_mem_util,
            tp=user_scenario.tp_size,
            pp=user_scenario.pp_size,
            dp=user_scenario.dp_size,
        )
        if calc_result is None:
            return None

        kv = calc_result["kv_cache_detail"]
        max_concurrent_requests_num = calc_result.get("max_concurrent_requests")

        if max_concurrent_requests_num is not None:
            col2.info(
                f"Assuming the worst case scenario, such that every request contains `--max-model-len` tokens, each request takes {util.pretty_round(kv['per_request_kv_cache_gb'])} GB for KV cache, which means the maximum concurrent requests that can be processed is {max_concurrent_requests_num}."
            )

        # Display details on how KV cache is estimated
        with st.expander("See how KV cache is calculated below"):
            st.write(f"""First, the per-token memory requirement is estimated given the following inputs:
- KV cache data type: `{kv["kv_data_type"]}` = {kv["precision_in_bytes"]} bytes in memory
- Hidden layers: {kv["num_hidden_layers"]}

This model uses _{kv["attention_type"]}_. The relevant parameters are:
""")
            if kv.get("kv_lora_rank"):
                st.write(f"""- KV lora rank: {kv["kv_lora_rank"]}
- QK rope head dimension: {kv["qk_rope_head_dim"]}""")

                st.code(f"""
Per-token memory = layers x (kv_lora_rank + qk_rope_head_dim) x precision_in_bytes
                 = {kv["num_hidden_layers"]} x ({kv["kv_lora_rank"]} + {kv["qk_rope_head_dim"]}) x {kv["precision_in_bytes"]}
                 = {kv["per_token_memory_bytes"]} bytes
""")
            else:
                st.write(f"""- Head dimension: {kv["head_dimension"]}
- Attention heads: {kv["num_attention_heads"]}
- KV heads: {kv["num_key_value_heads"]}
- Number of attention groups: {kv["num_attention_group"]}
""")

                st.code(f"""
Per-token memory = layers x 2 (two for K and V matrices) x head_dimension x (kv_heads / num_attention_groups) x precision_in_bytes
                 = {kv["num_hidden_layers"]} x 2 x {kv["head_dimension"]} x ({kv["num_attention_heads"]} / {kv["num_key_value_heads"]}) x {kv["precision_in_bytes"]}
                 = {kv["per_token_memory_bytes"]} bytes
""")

            st.write(f"""Finally, the per-token-memory is then multiplied by the context length (max-model-len) and batch size (concurrency).
- Number of tokens (context length): {user_scenario.max_model_len}
- Concurrency: {user_scenario.concurrency}
""")
            st.code(f"""
KV cache per request = per_token_memory x context_len x batch_size
                     = {kv["per_token_memory_bytes"]} x {user_scenario.max_model_len} x {user_scenario.concurrency}
                     = {kv["per_request_kv_cache_bytes"]} bytes
                     = {kv["per_request_kv_cache_bytes"]} / (1024 ^ 3)
                     = {kv["per_request_kv_cache_gb"]} GB
""")

            st.code(f"""
KV cache for max concurrency = kv_cache_per_request x concurrency
                             = {kv["per_request_kv_cache_gb"]} GB x {user_scenario.concurrency}
                             = {kv["kv_cache_size_gb"]} GB
""")

        # Display details on how activation memory is estimated
        with st.expander("See how activation memory is calculated below"):
            st.write("""During inference, vLLM requires memory for activations (hidden states, attention workspace, FFN intermediates, CUDA graphs).

**CRITICAL: Activation memory is CONSTANT per model type, NOT dependent on max_model_len or batch_size!**

This was empirically validated:
- Qwen3-0.6B at max_model_len=16000: **5.56 GB**
- Qwen3-0.6B at max_model_len=32000: **5.56 GB** (SAME!)

**Why Activation Memory is Constant:**

The "peak activation memory" represents FIXED overhead from vLLM's initialization and warmup:
1. **CUDA graph compilation**: vLLM pre-captures graphs for fixed batch sizes (1,2,4,8,16,32...) during warmup, regardless of max_model_len
2. **Profiling phase allocations**: vLLM runs dummy sequences to measure memory, creating fixed-size buffers
3. **PyTorch allocator overhead**: Pre-allocation and fragmentation independent of max_model_len
4. **FlashAttention workspace**: Fixed-size buffers allocated during engine initialization

Runtime per-request activation buffers (which DO scale with actual sequence length) are dynamically allocated from the KV cache memory pool, not counted in this fixed overhead.
""")

            act = model_info["activation_memory"]
            arch_name = model_info["architecture"]["architecture_name"]
            model_type = act["model_type"]
            base_constant = act["activation_memory_gb"]
            source_label = act["source"]

            st.write(f"""
**Model Type:** {model_type} | **Architecture:** `{arch_name or "unknown"}`

**Activation Memory Constants:**
- Dense models (default): {act["base_constants"]["dense_gib"]} GB
- MoE models: {act["base_constants"]["moe_gib"]} GB
- Multimodal models: {act["base_constants"]["multimodal_gib"]} GB

**Validated Profiles:**
""")
            for profile_arch, profile_mem in act["validated_profiles"].items():
                marker = " **<-- your model**" if profile_arch == arch_name else ""
                st.write(f"- `{profile_arch}`: {profile_mem} GB{marker}")

            st.write(f"""
**Your Model:** {base_constant} GB ({source_label})
""")

            st.code(f"""
Activation memory = {base_constant} GB ({source_label})
""")

            st.info(f"**Peak activation memory: {util.pretty_round(base_constant)} GB (constant)**")

            st.write("""
**Note on Tensor Parallelism (TP):**

Empirical measurements show activation memory does NOT scale inversely with TP. Both Llama-70B TP=1 and TP=2 show ~4.8 GB per GPU activation memory, suggesting vLLM's per-GPU allocation doesn't simply divide by TP.

**What's Included in the Base Constant:**

The empirical base constant captures vLLM's actual peak memory allocation during profiling:
- Hidden state buffers (input/output for layers)
- FlashAttention workspace (LSE buffers, output accumulators)
- FFN intermediate activations
- Multiple concurrent prefill requests during warmup
- CUDA graph compilation overhead
- PyTorch memory allocator fragmentation
- Request scheduling buffers

**Additional Memory Overheads:**
- **CUDA graph memory**: Included in activation estimate (empirical measurements show -0.45 to +0.39 GB as separate measurement, suggesting it's already accounted for)
- **Non-torch memory**: ~0.15 GB per GPU (TP=1) or ~0.6 GB per GPU (TP≥2) for CUDA runtime and Python interpreter overhead
""")


def hardware_specification():
    """
    Get hardware inputs like name and number of accelerators available
    """
    user_scenario = st.session_state[util.USER_SCENARIO_KEY]
    model_info = st.session_state.get("model_info_response")
    if not model_info:
        return None

    tp = user_scenario.tp_size
    pp = user_scenario.pp_size
    dp = user_scenario.dp_size

    with st.container(border=True):
        st.write("**Hardware Specification**")
        st.caption(
            "Identify suitable accelerators for serving the model based on parallelism optimization and workload."
        )

        col1, col2 = st.columns([0.6, 0.4])

        col1.number_input(
            "GPU utilization ratio",
            key=util.SELECTED_GPU_MEMORY_UTIL_KEY,
            value=user_scenario.gpu_mem_util,
            min_value=0.0,
            step=0.01,
            on_change=util.update_scenario,
            args=[util.SELECTED_GPU_MEMORY_UTIL_KEY, "gpu_mem_util"],
        )

        # Select GPU type
        selected_gpu_name = col1.selectbox(
            "Accelerator",
            key=util.SELECTED_GPU_NAME_KEY,
            options=gpu_specs,
            on_change=util.update_scenario,
            args=[util.SELECTED_GPU_NAME_KEY, "gpu_name"],
        )

        # For the selected GPU, show memory requirements
        if selected_gpu_name:
            gpu_memory = user_scenario.get_gpu_memory(gpu_specs)

            calc_result = _cached_calculate(
                user_scenario.model_name,
                max_model_len=user_scenario.max_model_len,
                batch_size=user_scenario.concurrency,
                gpu_memory=gpu_memory,
                gpu_mem_util=user_scenario.gpu_mem_util,
                tp=tp,
                pp=pp,
                dp=dp,
            )
            if calc_result is None:
                return None

            # Extract all needed values from calc_result
            model_size = calc_result["model_memory_gb"]
            model_size_per_gpu = calc_result["per_gpu_model_memory_gb"]
            available_gpu_mem = calc_result["available_gpu_memory_gb"]
            available_gpu_count = calc_result["total_gpus_required"]
            allocatable_kv_cache = calc_result["allocatable_kv_cache_memory_gb"]
            kv = calc_result["kv_cache_detail"]
            per_request_kv_cache_memory = kv["per_request_kv_cache_gb"]
            all_request_kv_cache_memory = kv["kv_cache_size_gb"]
            activation_mem_per_gpu = calc_result["activation_memory_gb"]
            cuda_graph_mem_per_gpu = calc_result["cuda_graph_memory_gb"]
            non_torch_mem_per_gpu = calc_result["non_torch_memory_gb"]

            # Compute derived values
            total_memory = gpu_memory * available_gpu_count
            total_available_gpu_mem = available_gpu_mem * available_gpu_count
            reserved = total_memory - total_available_gpu_mem
            total_model_size = model_size * dp
            activation_mem_total = activation_mem_per_gpu * dp
            cuda_graph_mem_total = cuda_graph_mem_per_gpu * available_gpu_count
            non_torch_mem_total = non_torch_mem_per_gpu * available_gpu_count

            free = (
                total_available_gpu_mem
                - total_model_size
                - all_request_kv_cache_memory
                - activation_mem_total
                - cuda_graph_mem_total
                - non_torch_mem_total
            )
            kv_cache_available_per_gpu_adjusted = (
                available_gpu_mem
                - model_size_per_gpu
                - activation_mem_per_gpu
                - cuda_graph_mem_per_gpu
                - non_torch_mem_per_gpu
            )

            st.caption(
                f"GPU memory: {gpu_memory} GB, available: {util.pretty_round(available_gpu_mem)} GB"
            )

            # Determine if GPU has enough memory
            col1, col2 = st.columns([0.6, 0.4])

            col1.info(f"""Memory breakdown per GPU:
- Model weights: {util.pretty_round(model_size_per_gpu)} GB
- Activation memory: {util.pretty_round(activation_mem_per_gpu)} GB
- CUDA graphs: {util.pretty_round(cuda_graph_mem_per_gpu)} GB
- Non-torch overhead: {util.pretty_round(non_torch_mem_per_gpu)} GB
- Available for KV cache: {util.pretty_round(kv_cache_available_per_gpu_adjusted)} GB
""")

            memory_util_chart(col1, calc_result)

            with col1.expander("Total memory breakdown"):
                st.markdown(f"""
**Memory Allocation:**
- Total memory: {gpu_memory * available_gpu_count} GB
- Reserved (1 - gpu_memory_util): {util.pretty_round(reserved)} GB
- Total memory available: {available_gpu_mem * available_gpu_count} GB

**Model & Weights:**
- Single model weights: {util.pretty_round(model_size)} GB
- Total model weights (x {dp} data parallel): {util.pretty_round(total_model_size)} GB

**Inference Overheads:**
- Activation memory (peak): {util.pretty_round(activation_mem_total)} GB ({util.pretty_round(activation_mem_per_gpu)} GB per GPU × {dp} DP replicas)
- CUDA graph memory: {util.pretty_round(cuda_graph_mem_total)} GB (included in activation profiling)
- Non-torch memory: {util.pretty_round(non_torch_mem_total)} GB ({util.pretty_round(non_torch_mem_per_gpu)} GB per GPU × {available_gpu_count} GPUs)

**KV Cache:**
- Allocatable KV cache memory: {util.pretty_round(allocatable_kv_cache)} GB
- KV cache per request: {util.pretty_round(per_request_kv_cache_memory)} GB
- KV cache for max concurrent requests: {util.pretty_round(all_request_kv_cache_memory)} GB

**Summary:**
- Model + Activation + Overheads + KV cache: {util.pretty_round(total_model_size + activation_mem_total + cuda_graph_mem_total + non_torch_mem_total + all_request_kv_cache_memory)} GB
- Free: {util.pretty_round(free)} GB
    """)

            if free < 0:
                col2.error("""The accelerator selected does not have enough GPU memory. Here is what you can do:
- Select a GPU with higher memory
- Increase GPU utilization ratio
- Increase tensor parallelism or pipeline parallelism
- Decrease max model length
- Decrease max concurrency""")
            else:
                col2.success(f"""The overall configuration has enough memory to load the model and process the desired workload. You will need `{available_gpu_count}x{selected_gpu_name}`s for the selected scenario. Below is the general vLLM serve command.
""")
                vllm_serve_cmd = f"""vllm serve {user_scenario.model_name} \\
    --max-model-len {user_scenario.max_model_len} \\
    --gpu-memory-utilization {user_scenario.gpu_mem_util} \\
    --tensor-parallel-size {tp} \\
    --pipeline-parallel-size {pp} \\
    --data-parallel-size {user_scenario.dp_size}"""
                if user_scenario.enable_ep:
                    vllm_serve_cmd += """ \\
    --enable-expert-parallel
        """
                col2.code(vllm_serve_cmd)


def memory_util_chart(st_context: Any, calc_result: dict) -> None:
    """
    Show memory utilization chart with detailed breakdown
    """
    user_scenario = st.session_state[util.USER_SCENARIO_KEY]
    gpu_memory = user_scenario.get_gpu_memory(gpu_specs)
    dp = user_scenario.dp_size

    # Extract values from calc_result
    gpu_count = calc_result["total_gpus_required"]
    model_size_single = calc_result["model_memory_gb"]
    available_single = calc_result["available_gpu_memory_gb"]
    activation_mem_per_gpu = calc_result["activation_memory_gb"]
    cuda_graph_mem_per_gpu = calc_result["cuda_graph_memory_gb"]
    non_torch_mem_per_gpu = calc_result["non_torch_memory_gb"]
    kv_cache_size = calc_result["kv_cache_detail"]["kv_cache_size_gb"]

    # Calculate totals
    total_memory = gpu_count * gpu_memory
    available = gpu_count * available_single
    reserved = total_memory - available
    model_size = model_size_single * dp
    activation_memory = activation_mem_per_gpu * dp
    cuda_graph_memory = cuda_graph_mem_per_gpu * gpu_count
    non_torch_memory = non_torch_mem_per_gpu * gpu_count

    # Free memory
    free = (
        available
        - model_size
        - kv_cache_size
        - activation_memory
        - cuda_graph_memory
        - non_torch_memory
    )

    if free < 0:
        st.warning(f"Memory exceeds available by {abs(util.pretty_round(free))} GB.")
        return None

    # Display chart with detailed breakdown
    labels = [
        "Model Weights",
        "KV Cache",
        "Activation Memory",
        "CUDA Graphs",
        "Non-Torch",
        "Free",
        "Reserved",
    ]
    sizes = [
        util.pretty_round(model_size),
        util.pretty_round(kv_cache_size),
        util.pretty_round(activation_memory),
        util.pretty_round(cuda_graph_memory),
        util.pretty_round(non_torch_memory),
        util.pretty_round(free),
        util.pretty_round(reserved),
    ]
    colors = [
        "#ff9999",  # Model - red
        "#66b3ff",  # KV Cache - blue
        "#ffcc99",  # Activation - orange
        "#cc99ff",  # CUDA Graphs - purple
        "#ff99cc",  # Non-Torch - pink
        "#99ff99",  # Free - green
        "#808080",  # Reserved - gray
    ]

    # Create donut chart
    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, texts = ax.pie(  # type: ignore[misc]
        sizes,
        colors=colors,
        startangle=90,  # Start at top
        wedgeprops={"width": 0.4},  # <-- Makes it a donut,
        labeldistance=1.1,  # Push labels outward
        pctdistance=0.7,  # Adjust percentage position
    )

    # Add total as text in the center of the donut
    ax.text(
        0,
        0,
        f"Total\n{util.pretty_round(total_memory)} GB",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    # Create a custom legend, including the total
    legend_labels = [f"{labels[i]}: {sizes[i]} GB" for i in range(len(labels))]

    # Position legend on the right
    ax.legend(
        wedges + [plt.Line2D([0], [0], color="#CCCCCC", lw=10)],  # Add fake handle for total
        legend_labels,
        title="Total GPU Memory Breakdown",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=9,
    )

    # Render in Streamlit
    _, col, _ = st_context.columns([0.5, 1, 0.5])
    with col:
        st.pyplot(fig, bbox_inches="tight")


st.title("Configuration Explorer")
st.caption(
    "This tool helps you find the most cost-effective, optimal configuration for serving models on llm-d based on hardware specification, workload characteristics, and SLO requirements."
)

util.init_session_state()
gpu_specs = fetch_gpu_types() or _load_gpu_specs_fallback()

# Pre-fetch model info for the default/current model if not already cached.
# The _model_info_fetch_attempted guard prevents an infinite rerun loop when
# the backend is unavailable or the fetch consistently returns None.
if not st.session_state.get("model_info_response") and not st.session_state.get(
    "_model_info_fetch_attempted"
):
    _current_model = st.session_state.get(util.SELECTED_MODEL_KEY, "")
    if _current_model:
        st.session_state["_model_info_fetch_attempted"] = True
        with st.spinner(f"Loading model info for `{_current_model}`..."):
            _model_info = fetch_capacity_planner_model_info(_current_model)
        if _model_info:
            st.session_state["model_info_response"] = _model_info
            st.session_state["_last_model_id"] = _current_model
            st.rerun()

# Display Capacity Planner headings
st.subheader("Capacity Planner")
st.caption(
    "Determine how many GPUs you need to fit your model and how many requests can be served at once depending on request patterns."
)

# Get user inputs and show outputs
model_specification()
parallelism_specification()
workload_specification()
hardware_specification()
