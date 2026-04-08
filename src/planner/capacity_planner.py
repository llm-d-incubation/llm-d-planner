"""
Capacity planner for LLM inference memory estimation.

This module implements memory estimation formulas for LLM inference with vLLM:
- Model weight memory requirements
- KV cache memory for different attention mechanisms (MHA, GQA, MQA, MLA)
- Activation memory during forward pass
- CUDA graph and system overhead

Calculates minimum GPU requirements based on model architecture, parallelism
configuration, and workload characteristics.
"""

import contextlib
import io
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Any, cast

from huggingface_hub import HfApi
from huggingface_hub.hf_api import ModelInfo, SafetensorsRepoMetadata

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from transformers import AutoConfig

# Memory Overhead Constants (in GiB)
# Empirically validated against vLLM on H100 GPUs with seq_len=16000, batch_size=1
# Source: empirical-test/analysis-results.md
# Test environment: H100 (79.18 GiB), vLLM with FlashAttention, max_model_len=16000
ACTIVATION_MEMORY_BASE_DENSE_GIB = (
    5.5  # Dense models: Qwen3-0.6B (5.56), Llama-8B (4.76), Llama-70B/TP2 (4.84)
)
ACTIVATION_MEMORY_BASE_MOE_GIB = 8.0  # MoE models: gpt-oss-20b (7.38)
ACTIVATION_MEMORY_BASE_MULTIMODAL_GIB = 2.5  # Multimodal models: Mistral-Small-3.2-24B (2.12)
ACTIVATION_REFERENCE_SEQ_LEN = 16000  # Reference sequence length for empirical measurements
VLLM_NON_TORCH_MEMORY_TP1_GIB = 0.15  # TP=1: empirical range 0.13-0.14 GiB
VLLM_NON_TORCH_MEMORY_TPN_GIB = 0.6  # TP≥2: empirical 0.55 GiB (TP=2)
# Note: CUDA graph memory is included in activation memory profiling, not a separate constant

# Tier 1: Validated activation profiles from empirical vLLM measurements on H100.
# Key = architecture string from model_config.architectures[0]
# Value = activation memory in GiB (torch peak memory increase from vLLM profiling)
# Source: config_explorer/empirical-vllm-memory-results.md
VALIDATED_ACTIVATION_PROFILES = {
    "LlamaForCausalLM": 4.8,  # Empirical: Llama-8B (4.76), Llama-70B/TP2 (4.84)
    "Qwen2ForCausalLM": 5.6,  # Empirical: same family as Qwen3
    "Qwen3ForCausalLM": 5.6,  # Empirical: Qwen3-0.6B (5.56), Qwen3-32B (5.64)
    "PixtralForConditionalGeneration": 2.5,  # Empirical: Mistral-Small-3.2-24B (2.12)
    "Mistral3ForConditionalGeneration": 2.5,  # Same architecture family as Pixtral
}

# Tier 2: Multimodal architectures typically have lower activation memory
# because the vision encoder does not participate in CUDA graph capture
MULTIMODAL_ARCHITECTURES = [
    "PixtralForConditionalGeneration",
    "Mistral3ForConditionalGeneration",
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
]

# Computational Constants
BYTES_PER_GIB = 1024**3
FP16_BF16_BYTES = 2  # Computational dtype for most inference workloads
HIGH_PRECISION_THRESHOLD_BYTES = 2  # Distinguish quantized vs full-precision
DEFAULT_KV_CACHE_DTYPE_BYTES = 1  # FP8 KV cache default


class AttentionType(StrEnum):
    """Attention mechanism types supported by the capacity planner."""

    MLA = "Multi-head latent attention"
    MHA = "Multi-head attention"
    GQA = "Grouped-query attention"
    MQA = "Multi-query attention"


@dataclass
class KVCacheDetail:
    # Required inputs from model config
    model: str
    attention_type: AttentionType
    kv_data_type: str
    precision_in_bytes: float
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dimension: int
    model_architecture: str

    # Derived outputs from input
    num_attention_group: int
    per_token_memory_bytes: int
    per_request_kv_cache_bytes: int
    per_request_kv_cache_gb: float  # Single request kv cache
    kv_cache_size_gb: float  # Batch size kv cache

    # Workload inputs
    context_len: int = 1
    batch_size: int = 1

    # Required inputs for MLA attention models
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None

    def __init__(
        self, model_name: str, model_config: AutoConfig, context_len: int = 1, batch_size: int = 1
    ):
        """
        KVCacheDetail stores information that are relevant to calculating KV cache memory requirement

        Args:
            model_name: HuggingFace model ID
            model_config: Model configuration from AutoConfig
            context_len: Context length (max tokens per request)
            batch_size: Batch size for KV cache calculation
        """
        self.model = model_name
        self.kv_data_type = inference_dtype(model_config)
        self.precision_in_bytes = inference_dtype_byte(model_config)
        architectures = getattr(model_config, "architectures", None)
        self.model_architecture = architectures[0] if architectures else ""

        # kv_data_type is stored at the model_config level, so need to fetch text_config afterward
        text_config = get_text_config(model_config)

        self.num_hidden_layers = text_config.num_hidden_layers
        self.hidden_size = text_config.hidden_size
        self.num_attention_heads = text_config.num_attention_heads
        self.num_key_value_heads = text_config.num_key_value_heads
        head_dim = getattr(text_config, "head_dim", None)
        self.head_dimension = (
            head_dim if head_dim is not None else int(self.hidden_size / self.num_attention_heads)
        )
        # Determine attention type
        if use_mla(self.model_architecture):
            self.attention_type = AttentionType.MLA
            self.kv_lora_rank = text_config.kv_lora_rank
            self.qk_rope_head_dim = text_config.qk_rope_head_dim
        else:
            if self.num_key_value_heads == 1:
                self.attention_type = AttentionType.MQA

            elif self.num_key_value_heads == self.num_attention_heads:
                self.attention_type = AttentionType.MHA

            else:
                # At this point, 1 < num_key_value_heads < num_attention_heads
                # For example, 8 KV heads with 32 attention heads, so 4 attention heads share the same KV matrices
                self.attention_type = AttentionType.GQA

        # Calculate kv cache size in bytes and in gb
        self.set_context_len(context_len)
        self.set_batch_size(batch_size)

    def set_context_len(self, context_len: int):
        """
        Sets context length and recalculates memory requirement
        """
        self.context_len = context_len
        self.__recalculate()

    def set_batch_size(self, batch_size: int):
        """
        Sets batch size and recalculates memory requirement
        """
        self.batch_size = batch_size
        self.__recalculate()

    def __recalculate(self):
        """ "
        Recalculates per token memory, kv cache size in bytes, and in GB

        KV Cache Memory Formulas:
        - Standard Attention (MHA, GQA, MQA):
          num_layers * 2 * num_kv_heads * head_dim * precision_bytes
          Factor of 2 for separate K and V caches

        - Multi-head Latent Attention (MLA):
          num_layers * (kv_lora_rank + qk_rope_head_dim) * precision_bytes
          Uses compressed KV representation (DeepSeek-V2/V3)

        Attention Types:
        - MHA: num_kv_heads == num_attention_heads (all heads have dedicated K,V)
        - GQA: 1 < num_kv_heads < num_attention_heads (multiple Q heads share K,V)
        - MQA: num_kv_heads == 1 (single K,V pair shared across all Q heads)
        - MLA: Compressed KV with low-rank projection
        """
        if self.attention_type == AttentionType.MLA:
            if self.kv_lora_rank is None:
                raise ValueError("kv_lora_rank is required for MLA attention")
            if self.qk_rope_head_dim is None:
                raise ValueError("qk_rope_head_dim is required for MLA attention")
            self.per_token_memory_bytes = int(
                self.num_hidden_layers
                * (self.kv_lora_rank + self.qk_rope_head_dim)
                * self.precision_in_bytes
            )
        else:
            self.num_attention_group = int(self.num_attention_heads / self.num_key_value_heads)
            self.per_token_memory_bytes = int(
                self.num_hidden_layers
                * 2
                * self.head_dimension
                * self.num_key_value_heads
                * self.precision_in_bytes
            )

        self.per_request_kv_cache_bytes = self.per_token_memory_bytes * self.context_len
        self.per_request_kv_cache_gb = bytes_to_gib(self.per_request_kv_cache_bytes)
        self.kv_cache_size_gb = self.per_request_kv_cache_gb * self.batch_size


# Model
def get_model_info_from_hf(model_name: str, hf_token: str | None = None) -> ModelInfo:
    """
    Fetches model info from HF, does not handle error
    """
    api = HfApi(token=hf_token)
    model_info = api.model_info(model_name)
    return model_info


def get_model_config_from_hf(model_name: str, hf_token: str | None = None) -> Any:
    """
    Returns LLM model config
    """

    model_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token or None,
    )

    return model_config


@lru_cache(maxsize=128)
def _get_safetensors_metadata_cached(
    model_name: str, hf_token: str | None = None
) -> SafetensorsRepoMetadata:
    """Cached internal function for fetching safetensors metadata."""
    api = HfApi(token=hf_token)
    return api.get_safetensors_metadata(model_name)


def get_safetensors_metadata_from_hf(
    model_name: str, hf_token: str | None = None
) -> SafetensorsRepoMetadata:
    """
    Fetches safetensors metadata directly from HuggingFace Hub.

    This uses HfApi.get_safetensors_metadata which parses safetensor headers
    directly, providing reliable parameter counts. Results are cached to avoid
    repeated API calls.

    Args:
        model_name: HuggingFace model ID (e.g., "meta-llama/Llama-3.3-70B")
        hf_token: Optional HuggingFace token for gated models

    Returns:
        SafetensorsRepoMetadata with parameter_count, sharded status, etc.

    Raises:
        NotASafetensorsRepoError: If model doesn't have safetensors files
    """
    return _get_safetensors_metadata_cached(model_name, hf_token)


def model_params_by_dtype(model_name: str, hf_token: str | None = None) -> dict[str, int]:
    """
    Returns parameter counts broken down by dtype.

    Example return: {"BF16": 70553706496} or {"BF16": 2109382656, "F8_E4M3": 68451041280}

    Args:
        model_name: HuggingFace model ID
        hf_token: Optional HuggingFace token for gated models

    Returns:
        Dict mapping dtype string to parameter count
    """
    metadata = get_safetensors_metadata_from_hf(model_name, hf_token)
    return cast(dict[str, int], metadata.parameter_count)


def get_text_config(model_config: Any) -> Any:
    """
    Returns text config (for LLMs)

    Some models nest LLM architecture inside 'text_config', some don't
    Compare https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json with https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506/blob/main/config.json
    """

    if hasattr(model_config, "text_config"):
        model_config = model_config.text_config

    return model_config


def get_quantization_config(model_config: Any) -> Any:
    """
    Returns the quantization config
    """

    return model_config.quantization_config


def is_quantized(model_config: AutoConfig) -> bool:
    """
    Returns True if model is quantized
    """

    return hasattr(model_config, "quantization_config")


def model_total_params(model_name: str, hf_token: str | None = None) -> int:
    """
    Returns the total parameters of the model.

    Uses HfApi.get_safetensors_metadata for reliable parameter counting.

    Args:
        model_name: HuggingFace model ID
        hf_token: Optional HuggingFace token for gated models

    Returns:
        Total number of parameters across all dtypes
    """
    metadata = get_safetensors_metadata_from_hf(model_name, hf_token)
    return sum(metadata.parameter_count.values())


def max_context_len(model_config: AutoConfig) -> int:
    """
    Returns the max context length accepted by model
    """
    text_config = get_text_config(model_config)
    return int(text_config.max_position_embeddings)


def estimate_vllm_non_torch_memory(tp: int = 1) -> float:
    """
    Estimate non-torch memory (CUDA runtime, Python interpreter) in GiB.

    Non-torch memory increases with TP due to NCCL/communication overhead.

    Args:
        tp: Tensor parallelism degree

    Returns:
        Non-torch memory in GiB per GPU
    """
    return VLLM_NON_TORCH_MEMORY_TP1_GIB if tp == 1 else VLLM_NON_TORCH_MEMORY_TPN_GIB


def estimate_vllm_cuda_graph_memory() -> float:
    """
    CUDA graph memory overhead per GPU in GiB.

    Note: Empirical measurements show CUDA graph memory is included in the
    activation memory profiling (range: -0.45 to +0.39 GiB as separate measurement).
    Returning 0.0 to avoid double-counting.

    Returns:
        0.0 (CUDA graph memory already included in activation estimate)
    """
    return 0.0


def estimate_vllm_activation_memory(config: AutoConfig, tp: int = 1) -> float:
    """
    Estimate peak activation memory for vLLM inference in GiB.

    Uses a tiered estimation strategy:
    1. Validated profiles: exact empirical measurements for known architectures
    2. Model type fallback: constants for MoE, multimodal, or dense models

    CRITICAL: Activation memory is CONSTANT per model type, NOT dependent on
    max_model_len or batch_size. This was empirically validated:
    - Qwen3-0.6B at max_model_len=16000: 5.56 GiB
    - Qwen3-0.6B at max_model_len=32000: 5.56 GiB (SAME!)

    The activation memory represents FIXED overhead from:
    - CUDA graph compilation and capture (fixed batch sizes: 1,2,4,8,16,32...)
    - vLLM's warmup profiling phase with dummy sequences
    - PyTorch memory allocator pre-allocation and fragmentation
    - Fixed-size workspace buffers allocated during engine initialization
    - FlashAttention workspace buffers (pre-allocated)

    Runtime per-request activation buffers (which DO scale with seq_len) are
    allocated from the KV cache memory pool, not counted here.

    Empirical validation:
    - Dense models: 4.76-5.56 GiB (Qwen3-0.6B, Llama-8B, Llama-70B)
    - MoE models: 7.38 GiB (gpt-oss-20b with 32 experts)
    - Multimodal models: 2.12 GiB (Mistral-Small-3.2-24B)

    Source: config_explorer/empirical-vllm-memory-results.md

    Args:
        config: Model configuration (can be full config or text_config)
        tp: Tensor parallelism degree (note: empirical data shows activation
            memory does NOT scale inversely with TP)

    Returns:
        float: Estimated peak activation memory in GiB (constant per model type)

    Raises:
        ValueError: If tp <= 0
    """
    if tp <= 0:
        raise ValueError(f"Tensor parallelism must be positive, got tp={tp}")

    # Tier 1: Check validated profiles by architecture
    if hasattr(config, "architectures") and config.architectures:
        arch = config.architectures[0]
        if arch in VALIDATED_ACTIVATION_PROFILES:
            return VALIDATED_ACTIVATION_PROFILES[arch]

    # Tier 2: Detect model type and use appropriate constant
    text_config = get_text_config(config)
    if is_moe(text_config):
        return ACTIVATION_MEMORY_BASE_MOE_GIB
    if is_multimodal(config):
        return ACTIVATION_MEMORY_BASE_MULTIMODAL_GIB
    return ACTIVATION_MEMORY_BASE_DENSE_GIB


def precision_to_byte(precision: str) -> float:
    """
    Returns the byte requirement the data type
    """

    precision = precision.strip().lower()

    mapping = {
        # Floating point
        "f64": 8,
        "f32": 4,
        "f16": 2,
        "bf16": 2,
        "f8_e5m2": 1,
        "f8_e4m3": 1,
        "fp4": 0.5,
        # Integers
        "i64": 8,
        "int64": 8,
        "i32": 4,
        "int32": 4,
        "i16": 2,
        "int16": 2,
        "i8": 1,
        "int8": 1,
        "u8": 1,
        "u4": 0.5,
        "i4": 0.5,
        "int4": 0.5,
        # Boolean
        "bool": 1,  # stored as byte per element
        # Special data types
        # gpt-oss: https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf
        # 4.25 bits per param
        "mxfp4": 4.25 / 8,
    }

    if precision in mapping:
        return float(mapping[precision])
    else:
        # Try to infer the precision from the first whole number
        match = re.search(r"\d+", precision)
        if match:
            bits = int(match.group(0))
            if bits % 8 == 0:
                return bits // 8

    raise ValueError("Unsupported precision type.")


def parameter_memory_req(parameter: int, precision: str) -> float:
    """
    Calculates the memory requirement (in GiB) for the number of parameters for the specified precision
    """

    precision_byte = precision_to_byte(precision)
    return bytes_to_gib(parameter * precision_byte)


def parameter_precision_memory_req(parameter: int, precision_in_byte: float) -> float:
    """
    Calculates the memory requirement (in GiB) for the number of parameters for the specified precision in bytes.
    """

    return bytes_to_gib(parameter * precision_in_byte)


def get_quant_method(model_config: AutoConfig) -> str:
    """
    Tries to determine the quant method used in quantization_config
    """

    if is_quantized(model_config):
        quantization_config = get_quantization_config(model_config)

        if "quant_method" in quantization_config:
            return str(quantization_config["quant_method"])

    return ""


def get_quant_bytes(model_config: AutoConfig) -> float:
    """
    Returns the number of bytes specified by quant_method
    """

    quant_config = get_quantization_config(model_config)
    quant_method = get_quant_method(model_config)
    if quant_method != "":
        try:
            return precision_to_byte(quant_method)

        # Quant method not convertible like "compressed-tensors"
        # Example: https://huggingface.co/RedHatAI/Qwen3-8B-FP8-dynamic/blob/main/config.json
        except ValueError:
            # Sometimes bits are given
            if "bits" in quant_config:
                return float(bits_to_bytes(quant_config["bits"]))

            # Sometimes bits are nested in config groups
            if (
                "config_groups" in quant_config
                and "group_0" in quant_config["config_groups"]
                and "weights" in quant_config["config_groups"]["group_0"]
            ):
                num_bits = quant_config["config_groups"]["group_0"]["weights"]["num_bits"]
                return float(bits_to_bytes(num_bits))

            return 0.0
    # Not quantized
    else:
        return 0.0


def model_memory_req(
    model_name: str, model_config: AutoConfig, hf_token: str | None = None
) -> float:
    """
    Calculates the GPU memory (in GiB) required for loading the model.

    Args:
        model_name: HuggingFace model ID
        model_config: Model configuration from AutoConfig
        hf_token: Optional HuggingFace token for gated models

    Returns:
        Memory requirement in GiB
    """
    model_params = model_params_by_dtype(model_name, hf_token)
    memory: float = 0.0

    # Check if model is quantized
    quantization_byte = None
    quant_method = get_quant_method(model_config) if is_quantized(model_config) else ""

    # MXFP4 (gpt-oss): safetensor metadata already reflects actual storage bytes.
    # U8 tensors contain packed 4-bit blocks and scales — use storage dtype directly.
    if quant_method == "mxfp4":
        for precision, num_params in model_params.items():
            memory += parameter_memory_req(num_params, precision)
        return memory

    if quant_method:
        quantization_byte = get_quant_bytes(model_config)

    for precision, num_params in model_params.items():
        precision_in_byte = precision_to_byte(precision)

        # IF FP16 or FP32, keep it as so
        if precision_in_byte >= 2:
            memory += parameter_memory_req(num_params, precision)
        else:
            # Otherwise, check if model is quantized, and use that as the precision
            if quantization_byte is not None:
                memory += parameter_precision_memory_req(num_params, quantization_byte)
            else:
                memory += parameter_memory_req(num_params, precision)

    return memory


def _extract_dtype_from_config(model_config: AutoConfig) -> str | None:
    """
    Extract dtype from model config, checking common attribute names.

    Returns:
        Dtype string if found, None otherwise
    """
    for attr in ["torch_dtype", "dtype"]:
        if hasattr(model_config, attr):
            dtype = getattr(model_config, attr)
            if dtype is not None:
                return str(dtype)
    return None


def inference_dtype(model_config: AutoConfig) -> str:
    """
    Returns the inference KV cache data type used.

    Checks model config dtype attributes first, falls back to quantization
    method if available, returns empty string if neither found.
    """
    dtype = _extract_dtype_from_config(model_config)
    if dtype is not None:
        return dtype

    if is_quantized(model_config):
        return get_quant_method(model_config)

    return ""


def inference_dtype_byte(model_config: AutoConfig) -> float:
    """
    Returns the precision for the inference KV cache data type in bytes.

    For standard dtypes (fp32, bf16, etc.), converts directly.
    For compressed formats (compressed-tensors), extracts from quantization config.
    Falls back to FP8 (1 byte) as default.
    """
    native_kv_dtype = inference_dtype(model_config)

    try:
        return precision_to_byte(native_kv_dtype)
    except ValueError:
        # Cannot determine from dtype string (e.g., "compressed-tensors")
        if is_quantized(model_config):
            return get_quant_bytes(model_config)

        return DEFAULT_KV_CACHE_DTYPE_BYTES


def use_mla(model_architecture: str) -> bool:
    """
    Returns true for models that use MLA attention
    """

    deepseek_mla_models = [
        "DeepseekV3ForCausalLM",
        "DeepseekV2ForCausalLM",
    ]

    return any(deepseek in model_architecture for deepseek in deepseek_mla_models)


def kv_cache_req(
    model_name: str,
    model_config: AutoConfig,
    context_len: int,
    batch_size: int = 1,
) -> float:
    """
    Calculates the KV cache requirement in GiB.

    Args:
        model_name: HuggingFace model ID
        model_config: Model configuration
        context_len: Context length (max tokens per request)
        batch_size: Batch size for KV cache calculation

    Returns:
        KV cache requirement in GiB
    """
    return KVCacheDetail(model_name, model_config, context_len, batch_size).kv_cache_size_gb


def total_kv_cache_blocks(
    model_name: str,
    model_config: AutoConfig,
    context_len: int,
    gpu_memory: int,
    gpu_mem_util: float = 0.9,
    batch_size: int = 1,
    block_size: int = 16,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    hf_token: str | None = None,
) -> int:
    """
    Calculate total number of KV cache blocks that can fit in GPU memory.

    Implements vLLM's block-based memory management. KV cache is divided into
    fixed-size blocks (default 16 tokens) for dynamic allocation and efficient
    memory sharing across requests.

    Args:
        model_name: HuggingFace model ID
        model_config: Model configuration
        context_len: Context length
        gpu_memory: GPU memory per device in GiB
        gpu_mem_util: GPU memory utilization factor
        batch_size: Batch size
        block_size: KV cache block size in tokens
        tp: Tensor parallelism degree
        pp: Pipeline parallelism degree
        dp: Data parallelism degree
        hf_token: Optional HuggingFace token for gated models

    Returns:
        Total number of KV cache blocks
    """
    kv_cache_detail = KVCacheDetail(model_name, model_config, context_len, batch_size)
    per_token_memory = kv_cache_detail.per_token_memory_bytes / (tp * pp)
    per_block_memory = per_token_memory * block_size

    kv_cache_allocatable = allocatable_kv_cache_memory(
        model_name,
        model_config,
        gpu_memory,
        gpu_mem_util,
        tp,
        pp,
        dp,
        max_model_len=context_len,
        batch_size=batch_size,
        hf_token=hf_token,
    )

    total_kv_blocks = gib_to_bytes(kv_cache_allocatable) // per_block_memory
    return int(total_kv_blocks)


def max_concurrent_requests(
    model_name: str,
    model_config: AutoConfig,
    max_model_len: int,
    gpu_memory: int,
    gpu_mem_util: float = 0.9,
    batch_size: int = 1,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    hf_token: str | None = None,
) -> int:
    """
    Calculate maximum number of concurrent requests that can be served.

    Args:
        model_name: HuggingFace model ID
        model_config: Model configuration
        max_model_len: Maximum sequence length per request
        gpu_memory: GPU memory per device in GiB
        gpu_mem_util: GPU memory utilization factor
        batch_size: Batch size for activation memory estimation
        tp: Tensor parallelism degree
        pp: Pipeline parallelism degree
        dp: Data parallelism degree
        hf_token: Optional HuggingFace token for gated models

    Returns:
        int: Maximum number of concurrent requests
    """
    # Find allocatable memory for KV cache
    kv_cache_allocatable = allocatable_kv_cache_memory(
        model_name,
        model_config,
        gpu_memory,
        gpu_mem_util,
        tp,
        pp,
        dp,
        max_model_len=max_model_len,
        batch_size=batch_size,
        hf_token=hf_token,
    )

    # Find kv cache requirement for one request of max-model-len
    per_request_kv_cache_req = kv_cache_req(model_name, model_config, max_model_len)
    # MEDIUM FIX: Check if allocatable_kv is non-positive to prevent division by zero
    if per_request_kv_cache_req == 0 or kv_cache_allocatable <= 0:
        return 0
    return max(0, math.floor(kv_cache_allocatable / per_request_kv_cache_req))


def find_possible_tp(model_config: AutoConfig) -> list[int]:
    """
    Find possible tensor parallelism values for the model.

    TP must be a divisor of num_attention_heads to ensure each TP rank has
    an integer number of heads. For example, 32 heads supports TP ∈ {1,2,4,8,16,32}.
    """
    text_config = get_text_config(model_config)
    num_attention_heads = text_config.num_attention_heads

    factors_list: list[int] = sorted(
        {
            x
            for i in range(1, int(num_attention_heads**0.5) + 1)
            if num_attention_heads % i == 0
            for x in [i, num_attention_heads // i]
        }
    )
    return factors_list


def available_gpu_memory(memory: int, gpu_utilization: float = 0.9) -> float:
    """
    Returns the available GPU memory
    """

    return memory * gpu_utilization


def gpus_required(tp: int = 1, pp: int = 1, dp: int = 1) -> int:
    """
    Determines the number of GPUs required based on parallelism strategies
    """

    return tp * pp * dp


def per_gpu_model_memory_required(
    model_name: str, model_config: AutoConfig, tp: int = 1, pp: int = 1, hf_token: str | None = None
) -> float:
    """
    Calculate model memory requirement per GPU.

    With parallelism: TP shards layers horizontally, PP distributes layers vertically.
    Memory per GPU = Total_model_memory / (TP × PP)

    Args:
        model_name: HuggingFace model ID
        model_config: Model configuration
        tp: Tensor parallelism degree
        pp: Pipeline parallelism degree
        hf_token: Optional HuggingFace token for gated models

    Returns:
        Memory requirement per GPU in GiB
    """
    model_memory = model_memory_req(model_name, model_config, hf_token)
    return model_memory / (tp * pp)


def allocatable_kv_cache_memory(
    model_name: str,
    model_config: AutoConfig,
    gpu_memory: int,
    gpu_util: float = 0.9,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    max_model_len: int | None = None,
    batch_size: int = 1,
    hf_token: str | None = None,
) -> float:
    """
    Calculate allocatable memory for KV cache after accounting for model weights,
    activation memory, CUDA graphs, and system overhead.

    Memory Formula:
    Available = (GPU_memory × utilization × num_GPUs)
              - (Model_weights × DP)
              - (Activation_memory × DP)
              - CUDA_graph_overhead
              - Non_torch_overhead

    Args:
        model_name: HuggingFace model ID
        model_config: Model configuration
        gpu_memory: GPU memory per device in GiB
        gpu_util: GPU memory utilization factor (default 0.9)
        tp: Tensor parallelism degree
        pp: Pipeline parallelism degree
        dp: Data parallelism degree
        max_model_len: Maximum sequence length (defaults to model's max_position_embeddings)
        batch_size: Batch size for activation memory estimation
        hf_token: Optional HuggingFace token for gated models

    Returns:
        float: Available memory for KV cache in GiB
    """
    gpu_count = tp * pp * dp
    available_memory = available_gpu_memory(gpu_memory, gpu_util) * gpu_count
    model_size = model_memory_req(model_name, model_config, hf_token) * dp

    if max_model_len is None:
        try:
            max_model_len = max_context_len(model_config)
        except AttributeError:
            max_model_len = 2048

    # Each data parallel replica needs its own activation memory
    # Note: activation memory is constant per model type, not dependent on max_model_len
    activation_memory = estimate_vllm_activation_memory(model_config, tp=tp) * dp

    # CUDA graph memory is included in activation memory profiling
    cuda_graph_memory = estimate_vllm_cuda_graph_memory() * gpu_count  # Returns 0.0

    # Non-torch memory scales with TP due to NCCL/communication overhead
    non_torch_memory = estimate_vllm_non_torch_memory(tp) * gpu_count

    total_consumed = model_size + activation_memory + cuda_graph_memory + non_torch_memory

    return max(0, available_memory - total_consumed)


def auto_max_model_len(
    model_name: str,
    model_config: AutoConfig,
    gpu_memory: int,
    gpu_mem_util: float = 0.9,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    hf_token: str | None = None,
) -> int:
    """
    Calculate the maximum possible max_model_len that fits in available GPU memory
    while allowing at least 1 concurrent request.

    Solves the memory equation backwards:
        max_model_len = floor(allocatable_kv_bytes / per_token_kv_bytes)
    Then caps at model's max_position_embeddings.

    Args:
        model_name: HuggingFace model ID
        model_config: Model configuration
        gpu_memory: GPU memory per device in GiB
        gpu_mem_util: GPU memory utilization factor (default 0.9)
        tp: Tensor parallelism degree
        pp: Pipeline parallelism degree
        dp: Data parallelism degree
        hf_token: Optional HuggingFace token for gated models

    Returns:
        int: Largest max_model_len that fits, or 0 if model doesn't fit at all
    """
    allocatable_kv = allocatable_kv_cache_memory(
        model_name,
        model_config,
        gpu_memory,
        gpu_mem_util,
        tp,
        pp,
        dp,
        max_model_len=1,
        batch_size=1,
        hf_token=hf_token,
    )

    if allocatable_kv <= 0:
        return 0

    kv_detail = KVCacheDetail(model_name, model_config, context_len=1, batch_size=1)
    per_token_bytes = kv_detail.per_token_memory_bytes / (tp * pp)

    if per_token_bytes <= 0:
        return 0

    max_tokens = int(gib_to_bytes(allocatable_kv) // per_token_bytes)

    if max_tokens <= 0:
        return 0

    try:
        model_max = max_context_len(model_config)
    except AttributeError:
        model_max = max_tokens

    return min(max_tokens, model_max)


def is_moe(model_config: AutoConfig) -> bool:
    """
    Returns true if model is MoE
    """
    indicators = [
        "n_routed_experts",
        "n_shared_experts",
        "num_experts",
        "num_experts_per_tok",
    ]
    return any(hasattr(model_config, indicator) for indicator in indicators)


def is_multimodal(model_config: AutoConfig) -> bool:
    """
    Returns true if model uses a multimodal (vision-language) architecture.

    Multimodal models typically have lower activation memory because the
    vision encoder does not participate in CUDA graph capture.
    """
    if hasattr(model_config, "architectures") and model_config.architectures:
        return any(arch in MULTIMODAL_ARCHITECTURES for arch in model_config.architectures)
    return False


def get_num_experts(model_config: AutoConfig) -> int | None:
    """
    Returns the number of experts or None for non-MoE models
    """

    if hasattr(model_config, "n_routed_experts"):
        return int(model_config.n_routed_experts)
    if hasattr(model_config, "num_experts"):
        return int(model_config.num_experts)
    return None


def get_ep_size(tp_size: int, dp_size: int) -> int:
    """
    Returns EP size
    """
    return tp_size * dp_size


def experts_per_ep_group(
    model_config: AutoConfig,
    tp: int = 1,
    dp: int = 1,
) -> float:
    """
    Calculate number of experts per GPU for MoE models.

    Expert Parallelism distributes expert FFN layers across GPUs.
    EP size = TP × DP, and experts are evenly sharded across the EP group.
    Each GPU stores (total_experts / EP_size) expert parameters.
    """
    num_experts = get_num_experts(model_config)
    ep_size = get_ep_size(tp, dp)
    if num_experts is None:
        return 0
    return num_experts / ep_size


# ---------------------- Utility helpers ----------------------
def bits_to_bytes(bits: int) -> int:
    """
    Convert number of bits to byte, assuming num bits is divisible
    """

    return int(bits / 8)


def bytes_to_gib(num_bytes: float) -> float:
    """
    Convert bytes to gibibytes (GiB)
    """
    return num_bytes / BYTES_PER_GIB


def gib_to_bytes(gib: float) -> float:
    """
    Convert gibibytes (GiB) to bytes
    """
    return gib * BYTES_PER_GIB


def get_model_info_summary(
    model_id: str, hf_token: str | None = None
) -> dict[str, Any]:
    """Assemble full model metadata for the /model-info API endpoint.

    Fetches model config from HuggingFace and computes memory breakdown,
    architecture info, quantization info, and activation memory estimate.

    Args:
        model_id: HuggingFace model ID (e.g. "meta-llama/Llama-3-8B")
        hf_token: Optional HF token for gated models

    Returns:
        Nested dict matching the ModelInfoResponse schema.

    Raises:
        Any exception raised by get_model_config_from_hf (HF fetch errors,
        auth errors, etc.) — callers are responsible for HTTP error mapping.
    """
    model_config = get_model_config_from_hf(model_id, hf_token)
    text_config = get_text_config(model_config)

    # --- model_info (parameter counts) ---
    try:
        params_by_dtype = model_params_by_dtype(model_id, hf_token)
    except Exception:
        params_by_dtype = {}
    total_params = (
        sum(params_by_dtype.values())
        if params_by_dtype
        else model_total_params(model_id, hf_token)
    )

    memory_gb = model_memory_req(model_id, model_config, hf_token)

    # --- architecture ---
    archs = getattr(model_config, "architectures", None) or []
    arch_name: str | None = archs[0] if archs else None
    is_moe_model = is_moe(text_config)
    is_multimodal_model = is_multimodal(model_config)
    if is_moe_model:
        model_type = "MoE"
    elif is_multimodal_model:
        model_type = "Multimodal"
    else:
        model_type = "Dense"

    # --- quantization ---
    is_quantized_model = is_quantized(model_config)
    quant_method_val = get_quant_method(model_config) if is_quantized_model else None
    quant_bytes_val = get_quant_bytes(model_config) if is_quantized_model else None

    # --- activation memory ---
    if arch_name and arch_name in VALIDATED_ACTIVATION_PROFILES:
        act_gb = VALIDATED_ACTIVATION_PROFILES[arch_name]
        act_source = f"Validated profile for {arch_name}"
    elif is_moe_model:
        act_gb = ACTIVATION_MEMORY_BASE_MOE_GIB
        act_source = "MoE default"
    elif is_multimodal_model:
        act_gb = ACTIVATION_MEMORY_BASE_MULTIMODAL_GIB
        act_source = "Multimodal default"
    else:
        act_gb = ACTIVATION_MEMORY_BASE_DENSE_GIB
        act_source = "Dense default"

    # --- memory breakdown (one row per dtype) ---
    breakdown: list[dict[str, Any]] = []
    quant_method_str = quant_method_val or ""
    quant_bytes_float = quant_bytes_val or 0.0
    for dtype, param_count in params_by_dtype.items():
        try:
            param_bytes = precision_to_byte(dtype)
        except ValueError:
            param_bytes = 0.0
        if param_bytes >= HIGH_PRECISION_THRESHOLD_BYTES or not quant_method_str:
            q_dtype = dtype
            q_bytes = param_bytes
            mem_gb = parameter_memory_req(param_count, dtype) if param_bytes > 0 else 0.0
        else:
            q_dtype = quant_method_str
            q_bytes = quant_bytes_float
            mem_gb = parameter_precision_memory_req(param_count, quant_bytes_float)
        breakdown.append(
            {
                "dtype": dtype,
                "quantized_dtype": q_dtype,
                "bytes_per_param": q_bytes,
                "num_parameters": param_count,
                "memory_gb": round(mem_gb, 2),
            }
        )

    return {
        "success": True,
        "model_id": model_id,
        "model_memory_gb": round(memory_gb, 2),
        "possible_tp_values": find_possible_tp(model_config),
        "model_info": {
            "total_parameters": total_params,
            "parameters_by_dtype": params_by_dtype,
        },
        "architecture": {
            "architecture_name": arch_name,
            "model_type": model_type,
            "num_hidden_layers": text_config.num_hidden_layers,
            "num_attention_heads": text_config.num_attention_heads,
            "inference_dtype": inference_dtype(model_config),
            "max_context_len": max_context_len(text_config),
            "is_moe": is_moe_model,
            "is_multimodal": is_multimodal_model,
            "num_experts": get_num_experts(model_config) if is_moe_model else None,
        },
        "quantization": {
            "is_quantized": is_quantized_model,
            "quant_method": quant_method_val,
            "quant_bytes": quant_bytes_val,
        },
        "activation_memory": {
            "activation_memory_gb": act_gb,
            "source": act_source,
            "model_type": model_type,
            "validated_profiles": dict(VALIDATED_ACTIVATION_PROFILES),
            "base_constants": {
                "dense_gib": ACTIVATION_MEMORY_BASE_DENSE_GIB,
                "moe_gib": ACTIVATION_MEMORY_BASE_MOE_GIB,
                "multimodal_gib": ACTIVATION_MEMORY_BASE_MULTIMODAL_GIB,
            },
        },
        "memory_breakdown": breakdown,
    }


def calculate_capacity(
    model_id: str,
    max_model_len: int | None,
    batch_size: int,
    gpu_memory: float | None,
    tp: int,
    pp: int,
    dp: int,
    gpu_mem_util: float,
    block_size: int,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Run capacity planning calculations for a model and hardware configuration.

    Args:
        model_id: HuggingFace model ID
        max_model_len: Token context length. -1 triggers auto-calculation
            (requires gpu_memory). None defaults to the model's max.
        batch_size: Max concurrent requests (KV cache batch dimension)
        gpu_memory: Per-GPU memory in GB. Required when max_model_len=-1.
        tp: Tensor parallelism degree
        pp: Pipeline parallelism degree
        dp: Data parallelism degree
        gpu_mem_util: Fraction of GPU memory available to vLLM (e.g. 0.9)
        block_size: KV cache block size in tokens
        hf_token: Optional HF token for gated models

    Returns:
        Nested dict matching the CalculateResponse schema.

    Raises:
        ValueError: Invalid input combinations (missing gpu_memory, bad tp).
        Any HF fetch exception — callers map these to HTTP errors.
    """
    model_config = get_model_config_from_hf(model_id, hf_token)
    text_config = get_text_config(model_config)
    warnings_list: list[str] = []

    # Resolve max_model_len
    max_model_len_auto = False
    if max_model_len == -1:
        if gpu_memory is None:
            raise ValueError(
                "max_model_len=-1 requires gpu_memory to be specified for auto-calculation"
            )
        max_len = auto_max_model_len(
            model_id,
            model_config,
            gpu_memory=int(gpu_memory),
            gpu_mem_util=gpu_mem_util,
            tp=tp,
            pp=pp,
            dp=dp,
            hf_token=hf_token,
        )
        if max_len == 0:
            raise ValueError(
                "Model does not fit in available GPU memory. Increase gpu_memory, tp, or pp."
            )
        if max_len < 128:
            warnings_list.append(
                f"Auto-calculated max_model_len is {max_len} tokens, which may be too small for practical use."
            )
        max_model_len_auto = True
    elif max_model_len is not None:
        max_len = max_model_len
    else:
        max_len = max_context_len(text_config)

    # Validate TP
    possible_tp = find_possible_tp(model_config)
    if tp not in possible_tp:
        raise ValueError(
            f"Invalid tp value {tp}. Valid values for this model: {possible_tp}"
        )

    kv = KVCacheDetail(model_id, model_config, max_len, batch_size)

    input_params: dict[str, Any] = {
        "model": model_id,
        "max_model_len": max_len,
        "batch_size": batch_size,
    }
    if max_model_len_auto:
        input_params["max_model_len_auto"] = True

    result: dict[str, Any] = {
        "success": True,
        "input_parameters": input_params,
        "kv_cache_detail": {
            "attention_type": str(kv.attention_type),
            "kv_data_type": kv.kv_data_type,
            "precision_in_bytes": kv.precision_in_bytes,
            "num_hidden_layers": kv.num_hidden_layers,
            "num_attention_heads": kv.num_attention_heads,
            "num_key_value_heads": kv.num_key_value_heads,
            "num_attention_group": kv.num_attention_group,
            "head_dimension": kv.head_dimension,
            "per_token_memory_bytes": kv.per_token_memory_bytes,
            "per_request_kv_cache_bytes": kv.per_request_kv_cache_bytes,
            "per_request_kv_cache_gb": round(kv.per_request_kv_cache_gb, 4),
            "kv_cache_size_gb": round(kv.kv_cache_size_gb, 2),
            "context_len": kv.context_len,
            "batch_size": kv.batch_size,
            "kv_lora_rank": kv.kv_lora_rank,
            "qk_rope_head_dim": kv.qk_rope_head_dim,
        },
        "warnings": warnings_list,
        "per_gpu_model_memory_gb": None,
        "total_gpus_required": None,
        "allocatable_kv_cache_memory_gb": None,
        "max_concurrent_requests": None,
        "total_kv_cache_blocks": None,
        "activation_memory_gb": None,
        "cuda_graph_memory_gb": None,
        "non_torch_memory_gb": None,
        "model_memory_gb": None,
        "available_gpu_memory_gb": None,
    }

    if gpu_memory is not None:
        gpu_memory_int = int(gpu_memory)
        input_params.update(
            {
                "tp": tp,
                "pp": pp,
                "dp": dp,
                "gpu_mem_util": gpu_mem_util,
                "block_size": block_size,
            }
        )
        result["per_gpu_model_memory_gb"] = round(
            per_gpu_model_memory_required(model_id, model_config, tp, pp, hf_token), 2
        )
        result["total_gpus_required"] = gpus_required(tp, pp, dp)
        result["allocatable_kv_cache_memory_gb"] = round(
            allocatable_kv_cache_memory(
                model_id,
                model_config,
                gpu_memory_int,
                gpu_mem_util,
                tp,
                pp,
                dp,
                max_model_len=max_len,
                batch_size=batch_size,
                hf_token=hf_token,
            ),
            2,
        )
        result["max_concurrent_requests"] = max_concurrent_requests(
            model_id,
            model_config,
            max_len,
            gpu_memory_int,
            gpu_mem_util,
            batch_size=batch_size,
            tp=tp,
            pp=pp,
            dp=dp,
            hf_token=hf_token,
        )
        result["total_kv_cache_blocks"] = int(
            total_kv_cache_blocks(
                model_id,
                model_config,
                max_len,
                gpu_memory_int,
                gpu_mem_util,
                batch_size,
                block_size,
                tp,
                pp,
                dp,
                hf_token=hf_token,
            )
        )
        result["activation_memory_gb"] = round(
            estimate_vllm_activation_memory(model_config, tp=tp), 4
        )
        result["cuda_graph_memory_gb"] = round(estimate_vllm_cuda_graph_memory(), 4)
        result["non_torch_memory_gb"] = round(estimate_vllm_non_torch_memory(tp), 4)
        result["model_memory_gb"] = round(model_memory_req(model_id, model_config, hf_token), 2)
        result["available_gpu_memory_gb"] = round(
            available_gpu_memory(gpu_memory_int, gpu_mem_util), 2
        )

    return result
