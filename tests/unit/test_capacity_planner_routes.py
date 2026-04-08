"""Unit tests for capacity planner API routes."""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from planner.api.app import create_app

client = TestClient(create_app())


# Minimal mock AutoConfig with the attributes the route accesses
def _mock_model_config(
    arch="LlamaForCausalLM",
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    hidden_size=4096,
    is_moe=False,
    is_multimodal=False,
    is_quantized=False,
):
    cfg = MagicMock()
    cfg.architectures = [arch]
    cfg.num_hidden_layers = num_hidden_layers
    cfg.num_attention_heads = num_attention_heads
    cfg.num_key_value_heads = num_key_value_heads
    cfg.hidden_size = hidden_size
    # text_config is the same object for non-multimodal
    cfg.text_config = cfg
    if not is_quantized:
        del cfg.quantization_config
        type(cfg).quantization_config = property(
            lambda self: (_ for _ in ()).throw(AttributeError("no quant"))
        )
    if not is_moe:
        # Remove MoE indicator attributes so is_moe() returns False
        for attr in ["n_routed_experts", "n_shared_experts", "num_experts", "num_experts_per_tok"]:
            if hasattr(cfg, attr):
                delattr(cfg, attr)

            # Capture attr in the closure
            def _raise_attr_error(a: str = attr) -> property:
                return property(lambda self: (_ for _ in ()).throw(AttributeError(f"no {a}")))

            setattr(type(cfg), attr, _raise_attr_error())
    return cfg


ROUTE = "/api/v1/model-info"

MOCK_PATH = "planner.capacity_planner"

_SAMPLE_MODEL_INFO = {
    "success": True,
    "model_id": "meta-llama/Llama-3-8B",
    "model_memory_gb": 14.0,
    "possible_tp_values": [1, 2, 4],
    "model_info": {
        "total_parameters": 7_000_000_000,
        "parameters_by_dtype": {"BF16": 7_000_000_000},
    },
    "architecture": {
        "architecture_name": "LlamaForCausalLM",
        "model_type": "Dense",
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "inference_dtype": "fp8",
        "max_context_len": 131072,
        "is_moe": False,
        "is_multimodal": False,
    },
    "quantization": {"is_quantized": False},
    "activation_memory": {
        "activation_memory_gb": 4.8,
        "source": "Validated profile for LlamaForCausalLM",
        "model_type": "Dense",
        "validated_profiles": {"LlamaForCausalLM": 4.8},
        "base_constants": {"dense_gib": 5.5, "moe_gib": 8.0, "multimodal_gib": 2.5},
    },
    "memory_breakdown": [],
}


@pytest.mark.unit
@patch("planner.capacity_planner.get_model_info_summary", return_value=_SAMPLE_MODEL_INFO)
def test_model_info_success(mock_summary):
    resp = client.post(ROUTE, json={"model_id": "meta-llama/Llama-3-8B"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["model_memory_gb"] == 14.0
    assert data["architecture"]["model_type"] == "Dense"
    # Verify the service function was called with the model_id (HF token comes from env)
    assert mock_summary.call_count == 1
    assert mock_summary.call_args[0][0] == "meta-llama/Llama-3-8B"


@pytest.mark.unit
@patch(
    "planner.capacity_planner.get_model_info_summary",
    side_effect=Exception("gated repo"),
)
def test_model_info_gated_model(mock_summary):
    resp = client.post(ROUTE, json={"model_id": "meta-llama/Llama-3-70B"})
    assert resp.status_code == 403


@pytest.mark.unit
@patch(
    "planner.capacity_planner.get_model_info_summary",
    side_effect=Exception("repo not found"),
)
def test_model_info_not_found(mock_summary):
    resp = client.post(ROUTE, json={"model_id": "nonexistent/model"})
    assert resp.status_code == 400


@pytest.mark.unit
def test_model_info_missing_model_id():
    resp = client.post(ROUTE, json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /calculate tests
# ---------------------------------------------------------------------------

CALC_ROUTE = "/api/v1/calculate"

_SAMPLE_CALC_NO_GPU: dict = {
    "success": True,
    "input_parameters": {"model": "meta-llama/Llama-3-8B", "max_model_len": 32768, "batch_size": 1},
    "kv_cache_detail": {
        "attention_type": "Grouped-query attention",
        "kv_data_type": "fp8",
        "precision_in_bytes": 1,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_attention_group": 4,
        "head_dimension": 128,
        "per_token_memory_bytes": 65536,
        "per_request_kv_cache_bytes": 2_147_483_648,
        "per_request_kv_cache_gb": 2.0,
        "kv_cache_size_gb": 2.0,
        "context_len": 32768,
        "batch_size": 1,
        "kv_lora_rank": None,
        "qk_rope_head_dim": None,
    },
    "warnings": [],
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

_SAMPLE_CALC_WITH_GPU: dict = {
    **_SAMPLE_CALC_NO_GPU,
    "per_gpu_model_memory_gb": 14.0,
    "total_gpus_required": 1,
    "allocatable_kv_cache_memory_gb": 10.5,
    "max_concurrent_requests": 4,
    "total_kv_cache_blocks": 512,
    "activation_memory_gb": 5.5,
    "cuda_graph_memory_gb": 0.5,
    "non_torch_memory_gb": 0.15,
    "model_memory_gb": 14.0,
    "available_gpu_memory_gb": 72.0,
}


@pytest.mark.unit
@patch("planner.capacity_planner.calculate_capacity", return_value=_SAMPLE_CALC_NO_GPU)
def test_calculate_basic_no_gpu(mock_calc):
    resp = client.post(CALC_ROUTE, json={"model_id": "meta-llama/Llama-3-8B"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["per_gpu_model_memory_gb"] is None
    assert data["warnings"] == []


@pytest.mark.unit
@patch("planner.capacity_planner.calculate_capacity", return_value=_SAMPLE_CALC_WITH_GPU)
def test_calculate_with_gpu_memory(mock_calc):
    resp = client.post(CALC_ROUTE, json={"model_id": "meta-llama/Llama-3-8B", "gpu_memory": 80})
    assert resp.status_code == 200
    data = resp.json()
    assert data["per_gpu_model_memory_gb"] == 14.0
    assert data["activation_memory_gb"] == 5.5


@pytest.mark.unit
@patch(
    "planner.capacity_planner.calculate_capacity",
    side_effect=ValueError("max_model_len=-1 requires gpu_memory"),
)
def test_calculate_auto_max_model_len_requires_gpu_memory(mock_calc):
    resp = client.post(CALC_ROUTE, json={"model_id": "meta-llama/Llama-3-8B", "max_model_len": -1})
    assert resp.status_code == 400
    assert "gpu_memory" in resp.json()["detail"]


@pytest.mark.unit
@patch(
    "planner.capacity_planner.calculate_capacity",
    side_effect=ValueError("Invalid tp value 3. Valid values for this model: [1, 2, 4]"),
)
def test_calculate_invalid_tp(mock_calc):
    resp = client.post(CALC_ROUTE, json={"model_id": "meta-llama/Llama-3-8B", "tp": 3})
    assert resp.status_code == 400
    assert "tp" in resp.json()["detail"].lower()


@pytest.mark.unit
@patch(
    "planner.capacity_planner.calculate_capacity",
    return_value={**_SAMPLE_CALC_NO_GPU, "warnings": ["Auto-calculated max_model_len is 64 tokens, which may be too small for practical use."]},
)
def test_calculate_auto_max_model_len_small_warning(mock_calc):
    resp = client.post(
        CALC_ROUTE,
        json={"model_id": "meta-llama/Llama-3-8B", "max_model_len": -1, "gpu_memory": 80},
    )
    assert resp.status_code == 200
    assert len(resp.json()["warnings"]) > 0
    assert "64" in resp.json()["warnings"][0]
