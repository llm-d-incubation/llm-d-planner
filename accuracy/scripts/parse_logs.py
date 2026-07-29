#!/usr/bin/env python3
"""Parse vLLM v0.19.0 startup logs to extract memory metrics into a CSV."""

import csv
import re
import sys
from pathlib import Path


def parse_log(log_path: Path) -> dict:
    text = log_path.read_text(errors="replace")
    lines = text.splitlines()

    row = {"log_file": log_path.name, "status": "ok"}

    # ── Config: Initializing a V1 LLM engine line ──────────────────────────
    m = re.search(
        r"Initializing a V1 LLM engine.*?model='([^']+)'.*?"
        r"dtype=([^,]+).*?max_seq_len=(\d+).*?"
        r"tensor_parallel_size=(\d+).*?pipeline_parallel_size=(\d+).*?"
        r"data_parallel_size=(\d+).*?quantization=([^,]+).*?"
        r"kv_cache_dtype=([^,\)]+)",
        text,
    )
    if m:
        row["model"] = m.group(1)
        row["dtype"] = m.group(2).strip()
        row["max_model_len"] = int(m.group(3))
        row["tp"] = int(m.group(4))
        row["pp"] = int(m.group(5))
        row["dp"] = int(m.group(6))
        row["quantization"] = m.group(7).strip()
        row["kv_cache_dtype"] = m.group(8).strip()

    # ── Config: gpu_memory_utilization from non-default args ───────────────
    m = re.search(r"gpu_memory_utilization':\s*([0-9.]+)", text)
    if not m:
        m = re.search(r"Desired GPU memory utilization is \(([0-9.]+),", text)
    if m:
        row["gpu_memory_utilization"] = float(m.group(1))

    # ── GPU type from filename (e.g. h100-80gb) ────────────────────────────
    gpu_m = re.search(r"--([a-z0-9]+-\d+gb)--", log_path.name, re.I)
    row["gpu"] = gpu_m.group(1).upper() if gpu_m else ""

    # ── Worker init memory snapshot (take first occurrence) ───────────────
    m = re.search(
        r"worker init memory snapshot: torch_peak=([0-9.]+)GiB, "
        r"free_memory=([0-9.]+)GiB, total_memory=([0-9.]+)GiB, "
        r"cuda_memory=([0-9.]+)GiB, torch_memory=([0-9.]+)GiB, "
        r"non_torch_memory=([0-9.]+)GiB",
        text,
    )
    if m:
        row["init_free_memory_gib"] = float(m.group(2))
        row["init_total_memory_gib"] = float(m.group(3))
        row["init_cuda_memory_gib"] = float(m.group(4))
        row["init_non_torch_memory_gib"] = float(m.group(6))

    # ── Worker requested memory (take first) ──────────────────────────────
    m = re.search(r"worker requested memory: ([0-9.]+)GiB", text)
    if m:
        row["requested_memory_gib"] = float(m.group(1))

    # ── Model loading: weight memory ──────────────────────────────────────
    m = re.search(r"Model loading took ([0-9.]+) GiB memory", text)
    if m:
        row["weight_memory_gib"] = float(m.group(1))

    # ── Memory profiling breakdown (take first occurrence = TP0 or solo) ──
    m = re.search(
        r"Memory profiling takes [0-9.]+ seconds\. "
        r"Total non KV cache memory: ([0-9.]+)GiB; "
        r"torch peak memory increase: ([0-9.]+)GiB; "
        r"non-torch forward increase memory: ([0-9.]+)GiB; "
        r"weights memory: ([0-9.]+)GiB\.",
        text,
    )
    if m:
        row["total_non_kv_cache_gib"] = float(m.group(1))
        row["activation_memory_gib"] = float(m.group(2))
        row["non_torch_forward_gib"] = float(m.group(3))
        row["weights_memory_gib"] = float(m.group(4))

    # ── Estimated CUDA graph memory (take first) ──────────────────────────
    m = re.search(r"Estimated CUDA graph memory: ([0-9.]+) GiB total", text)
    if m:
        row["cuda_graph_estimated_gib"] = float(m.group(1))

    # ── Actual CUDA graph pool (take first) ───────────────────────────────
    m = re.search(
        r"CUDA graph pool memory: ([0-9.]+) GiB \(actual\), ([0-9.]+) GiB \(estimated\)",
        text,
    )
    if m:
        row["cuda_graph_actual_gib"] = float(m.group(1))

    # ── Available KV cache memory (take first = Worker_TP0 or solo) ───────
    m = re.search(r"Available KV cache memory: ([0-9.]+) GiB", text)
    if m:
        row["kv_cache_memory_gib"] = float(m.group(1))

    # ── GPU KV cache tokens (EngineCore, single line) ─────────────────────
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", text)
    if m:
        row["kv_cache_tokens"] = int(m.group(1).replace(",", ""))

    # ── Max concurrency ───────────────────────────────────────────────────
    m = re.search(
        r"Maximum concurrency for ([\d,]+) tokens per request: ([0-9.]+)x", text
    )
    if m:
        row["max_concurrency"] = float(m.group(2))

    # ── KV cache blocks (from metrics log line) ───────────────────────────
    m = re.search(r"num_gpu_blocks is: (\d+)", text)
    if m:
        row["kv_cache_blocks"] = int(m.group(1))

    # ── Free memory on device summary (take first) ────────────────────────
    m = re.search(
        r"Free memory on device \(([0-9.]+)/([0-9.]+) GiB\)", text
    )
    if m:
        row["summary_free_gib"] = float(m.group(1))
        row["summary_total_gib"] = float(m.group(2))

    # ── PIECEWISE / FULL CUDA graph capture counts ────────────────────────
    m = re.search(
        r"Profiling CUDA graph memory: PIECEWISE=(\d+) \(largest=(\d+)\), FULL=(\d+)",
        text,
    )
    if m:
        row["cudagraph_piecewise_count"] = int(m.group(1))
        row["cudagraph_piecewise_largest"] = int(m.group(2))
        row["cudagraph_full_count"] = int(m.group(3))

    return row


COLUMNS = [
    "log_file",
    "status",
    "model",
    "gpu",
    "tp",
    "pp",
    "dp",
    "max_model_len",
    "dtype",
    "quantization",
    "kv_cache_dtype",
    "gpu_memory_utilization",
    # initial GPU state
    "init_free_memory_gib",
    "init_total_memory_gib",
    "init_cuda_memory_gib",
    "init_non_torch_memory_gib",
    "requested_memory_gib",
    # per-GPU memory breakdown
    "weight_memory_gib",
    "weights_memory_gib",       # from profiling line (should match)
    "activation_memory_gib",
    "non_torch_forward_gib",
    "total_non_kv_cache_gib",
    "cuda_graph_estimated_gib",
    "cuda_graph_actual_gib",
    "kv_cache_memory_gib",
    # KV cache sizing
    "kv_cache_tokens",
    "kv_cache_blocks",
    "max_concurrency",
    # summary
    "summary_free_gib",
    "summary_total_gib",
    "cudagraph_piecewise_count",
    "cudagraph_piecewise_largest",
    "cudagraph_full_count",
]


def main():
    logs_dir = Path(__file__).parent.parent / "results/v0.19.0/logs"
    out_path = Path(__file__).parent.parent / "results/v0.19.0/results_raw.csv"

    log_files = sorted(logs_dir.glob("*.log"))
    print(f"Found {len(log_files)} log files (including .FAILED.log)")

    rows = []
    for lf in log_files:
        row = parse_log(lf)
        if lf.name.endswith(".FAILED.log"):
            row["status"] = "failed"
        rows.append(row)
        print(f"  {'✓' if row['status'] == 'ok' else '✗'} {lf.name}")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows → {out_path}")


if __name__ == "__main__":
    main()
