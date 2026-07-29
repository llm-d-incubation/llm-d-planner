#!/usr/bin/env python3
"""
Generate capacity planner predictions for configs in results_raw.csv.

Calls the capacity_planner module with each row's model/config and outputs
the formula-based predictions for direct comparison with actual log values.
"""

import csv
import os
import sys
import traceback
from pathlib import Path

# Add src/ to path so we can import planner modules
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from planner.capacity_planner import (
    KVCacheDetail,
    allocatable_kv_cache_memory,
    estimate_vllm_activation_memory,
    estimate_vllm_cuda_graph_memory,
    estimate_vllm_non_torch_memory,
    get_model_config_from_hf,
    model_memory_req,
    per_gpu_model_memory_required,
)

GPU_MEMORY_GIB = 80  # H100 catalog value
HF_TOKEN = os.environ.get("HF_TOKEN") or None


def predict(row: dict) -> dict:
    model = row["model"]
    tp = int(row["tp"])
    pp = int(row["pp"])
    dp = int(row["dp"])
    max_model_len = int(row["max_model_len"])
    gpu_util = float(row["gpu_memory_utilization"])

    model_config = get_model_config_from_hf(model, HF_TOKEN)

    # Weight memory
    total_weight_gib = model_memory_req(model, model_config, HF_TOKEN)
    per_gpu_weight_gib = per_gpu_model_memory_required(model, model_config, tp, pp, HF_TOKEN)

    # Activation (constant per model type, independent of max_model_len)
    activation_gib = estimate_vllm_activation_memory(model_config, tp=tp)

    # Non-torch (system overhead)
    non_torch_gib = estimate_vllm_non_torch_memory(tp)

    # CUDA graph (included in activation profiling → 0 as separate term)
    cuda_graph_gib = estimate_vllm_cuda_graph_memory()

    # Total non-KV per GPU = weight/gpu + activation + non_torch + cuda_graph
    total_non_kv_per_gpu = per_gpu_weight_gib + activation_gib + non_torch_gib + cuda_graph_gib

    # Allocatable KV cache: total across tp*pp*dp GPUs
    alloc_kv_total = allocatable_kv_cache_memory(
        model,
        model_config,
        GPU_MEMORY_GIB,
        gpu_util,
        tp,
        pp,
        dp,
        max_model_len=max_model_len,
        hf_token=HF_TOKEN,
    )
    # Per-GPU KV = total / (tp * pp * dp) — matches what vLLM reports per worker
    per_gpu_kv_gib = alloc_kv_total / (tp * pp * dp)

    # KV cache detail: per-token bytes and per-request bytes
    kv_detail = KVCacheDetail(model, model_config, context_len=max_model_len)
    per_token_bytes = kv_detail.per_token_memory_bytes
    per_token_bytes_per_gpu = per_token_bytes / (tp * pp)

    # Derive token count and max concurrency from predicted KV memory
    alloc_kv_bytes = per_gpu_kv_gib * (1024**3)
    # Account for TP sharding: each GPU holds 1/(tp*pp) of each token's KV
    kv_tokens = int(alloc_kv_bytes / per_token_bytes_per_gpu) if per_token_bytes_per_gpu > 0 else 0
    per_request_bytes = per_token_bytes_per_gpu * max_model_len
    max_concurrency = alloc_kv_bytes / per_request_bytes if per_request_bytes > 0 else 0

    arch = ""
    if hasattr(model_config, "architectures") and model_config.architectures:
        arch = model_config.architectures[0]

    return {
        # Config (pass-through)
        "model": model,
        "gpu": row["gpu"],
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "max_model_len": max_model_len,
        "dtype": row["dtype"],
        "quantization": row["quantization"],
        "kv_cache_dtype": row["kv_cache_dtype"],
        "gpu_memory_utilization": gpu_util,
        # Model architecture
        "architecture": arch,
        "attention_type": kv_detail.attention_type.value,
        "num_hidden_layers": kv_detail.num_hidden_layers,
        "num_kv_heads": kv_detail.num_key_value_heads,
        "head_dimension": kv_detail.head_dimension,
        "kv_dtype_bytes": kv_detail.precision_in_bytes,
        "per_token_kv_bytes": per_token_bytes,
        "per_token_kv_bytes_per_gpu": per_token_bytes_per_gpu,
        # Memory predictions (per GPU)
        "pred_weight_memory_gib": round(per_gpu_weight_gib, 4),
        "pred_activation_memory_gib": round(activation_gib, 4),
        "pred_non_torch_gib": round(non_torch_gib, 4),
        "pred_cuda_graph_gib": round(cuda_graph_gib, 4),
        "pred_total_non_kv_cache_gib": round(total_non_kv_per_gpu, 4),
        "pred_kv_cache_memory_gib": round(per_gpu_kv_gib, 4),
        # Derived capacity predictions
        "pred_kv_cache_tokens": kv_tokens,
        "pred_max_concurrency": round(max_concurrency, 2),
        # Totals used in formula
        "pred_total_weight_gib": round(total_weight_gib, 4),
        "pred_alloc_kv_total_gib": round(alloc_kv_total, 4),
    }


COLUMNS = [
    "model", "gpu", "tp", "pp", "dp", "max_model_len",
    "dtype", "quantization", "kv_cache_dtype", "gpu_memory_utilization",
    "architecture", "attention_type",
    "num_hidden_layers", "num_kv_heads", "head_dimension",
    "kv_dtype_bytes", "per_token_kv_bytes", "per_token_kv_bytes_per_gpu",
    "pred_weight_memory_gib",
    "pred_activation_memory_gib",
    "pred_non_torch_gib",
    "pred_cuda_graph_gib",
    "pred_total_non_kv_cache_gib",
    "pred_kv_cache_memory_gib",
    "pred_kv_cache_tokens",
    "pred_max_concurrency",
    "pred_total_weight_gib",
    "pred_alloc_kv_total_gib",
]


def main():
    raw_csv = REPO_ROOT / "accuracy/results/v0.19.0/results_raw.csv"
    out_csv = REPO_ROOT / "accuracy/results/v0.19.0/results_predicted.csv"

    rows = [r for r in csv.DictReader(raw_csv.open()) if r["status"] == "ok"]
    print(f"Processing {len(rows)} successful rows (H100, gpu_memory={GPU_MEMORY_GIB} GiB)\n")

    results = []
    failed = []
    for row in rows:
        model = row["model"]
        label = f"{model} tp={row['tp']} pp={row['pp']} len={row['max_model_len']}"
        try:
            pred = predict(row)
            results.append(pred)
            print(f"  ✓ {label}")
            print(f"      weight={pred['pred_weight_memory_gib']} GiB  "
                  f"activ={pred['pred_activation_memory_gib']} GiB  "
                  f"kv={pred['pred_kv_cache_memory_gib']} GiB")
        except Exception as e:
            failed.append((label, str(e)))
            print(f"  ✗ {label}: {e}")
            if "--verbose" in sys.argv:
                traceback.print_exc()

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote {len(results)} rows → {out_csv}")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for label, err in failed:
            print(f"  {label}: {err}")


if __name__ == "__main__":
    main()
