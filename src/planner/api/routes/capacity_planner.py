"""Capacity planner endpoints."""

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

import planner.capacity_planner as cp
from planner.api.routes.common import handle_hf_error

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["capacity-planner"])


def _get_hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or None


# ---------------------------------------------------------------------------
# /model-info schemas
# ---------------------------------------------------------------------------


class ModelInfoRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str


class ModelInfoDetail(BaseModel):
    total_parameters: int
    parameters_by_dtype: dict[str, int]


class ArchitectureInfo(BaseModel):
    model_config = {"protected_namespaces": ()}
    architecture_name: str | None
    model_type: str
    num_hidden_layers: int
    num_attention_heads: int
    inference_dtype: str
    max_context_len: int
    is_moe: bool
    is_multimodal: bool
    num_experts: int | None = None


class QuantizationInfo(BaseModel):
    is_quantized: bool
    quant_method: str | None = None
    quant_bytes: float | None = None


class ActivationMemoryInfo(BaseModel):
    model_config = {"protected_namespaces": ()}
    activation_memory_gb: float
    source: str
    model_type: str
    validated_profiles: dict[str, float]
    base_constants: dict[str, float]


class MemoryBreakdownRow(BaseModel):
    dtype: str
    quantized_dtype: str
    bytes_per_param: float
    num_parameters: int
    memory_gb: float


class ModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    success: bool
    model_id: str
    model_memory_gb: float
    possible_tp_values: list[int]
    model_info: ModelInfoDetail
    architecture: ArchitectureInfo
    quantization: QuantizationInfo
    activation_memory: ActivationMemoryInfo
    memory_breakdown: list[MemoryBreakdownRow]


# ---------------------------------------------------------------------------
# /model-info handler
# ---------------------------------------------------------------------------


@router.post("/model-info")
async def model_info(request: ModelInfoRequest) -> ModelInfoResponse:
    """Fetch model metadata from HuggingFace.

    Reads HF_TOKEN from backend environment — never from the request.
    """
    hf_token = _get_hf_token()
    try:
        summary = cp.get_model_info_summary(request.model_id, hf_token)
    except Exception as e:
        handle_hf_error(e)
    return ModelInfoResponse(**summary)


# ---------------------------------------------------------------------------
# /calculate schemas
# ---------------------------------------------------------------------------


class CalculateRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str
    max_model_len: int | None = None
    batch_size: int = 1
    gpu_memory: float | None = None
    tp: int = 1
    pp: int = 1
    dp: int = 1
    gpu_mem_util: float = 0.9
    block_size: int = 16


class KVCacheDetailSchema(BaseModel):
    attention_type: str
    kv_data_type: str
    precision_in_bytes: float
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_attention_group: int
    head_dimension: int
    per_token_memory_bytes: int
    per_request_kv_cache_bytes: int
    per_request_kv_cache_gb: float
    kv_cache_size_gb: float
    context_len: int
    batch_size: int
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None


class CalculateResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    success: bool
    input_parameters: dict[str, Any]
    kv_cache_detail: KVCacheDetailSchema
    warnings: list[str]
    per_gpu_model_memory_gb: float | None = None
    total_gpus_required: int | None = None
    allocatable_kv_cache_memory_gb: float | None = None
    max_concurrent_requests: int | None = None
    total_kv_cache_blocks: int | None = None
    activation_memory_gb: float | None = None
    cuda_graph_memory_gb: float | None = None
    non_torch_memory_gb: float | None = None
    model_memory_gb: float | None = None
    available_gpu_memory_gb: float | None = None


# ---------------------------------------------------------------------------
# /calculate handler
# ---------------------------------------------------------------------------


@router.post("/calculate")
async def calculate(request: CalculateRequest) -> CalculateResponse:
    """Run capacity planning calculations for a given model and hardware config."""
    hf_token = _get_hf_token()
    try:
        result = cp.calculate_capacity(
            model_id=request.model_id,
            max_model_len=request.max_model_len,
            batch_size=request.batch_size,
            gpu_memory=request.gpu_memory,
            tp=request.tp,
            pp=request.pp,
            dp=request.dp,
            gpu_mem_util=request.gpu_mem_util,
            block_size=request.block_size,
            hf_token=hf_token,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except Exception as e:
        handle_hf_error(e)
    return CalculateResponse(**result)
