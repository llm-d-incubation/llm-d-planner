"""
Parse a vLLM v0.19.0 startup log and extract memory quantities.

Usage:
    python parse_log.py <path/to/startup.log>            # prints JSON to stdout
    python parse_log.py <path/to/startup.log> --out <path/to/output.json>

Key lines captured (verified against real vLLM v0.19.0 logs):
    Model loading took 14.99 GiB memory and 7.41 seconds
    Available KV cache memory: 58.11 GiB
    GPU KV cache size: 476,000 tokens
    Maximum concurrency for 8,192 tokens per request: 58.11x
    Estimated CUDA graph memory: 0.84 GiB total

Derived fields:
    kv_cache_blocks  = kv_cache_tokens / 16   (vLLM default block_size=16)
    kv_block_size_bytes = kv_cache_memory_gib * 2^30 / kv_cache_blocks
"""
import argparse
import json
import re
from pathlib import Path
from typing import Any

_FLOAT_PATTERNS: dict[str, str] = {
    "weight_memory_gib":      r"Model loading took ([\d.]+) GiB memory",
    "kv_cache_memory_gib":    r"Available KV cache memory:\s*([\d.]+) GiB",
    "cuda_graph_memory_gib":  r"Estimated CUDA graph memory:\s*([\d.]+) GiB",
    "max_concurrency":        r"Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x",
}
_INT_PATTERNS: dict[str, str] = {
    # Comma-separated integer, e.g. "476,000"
    "kv_cache_tokens": r"GPU KV cache size:\s*([\d,]+) tokens",
}
_OPT_STR_PATTERNS: dict[str, str] = {
    "vllm_version": r"vLLM version:\s*(\S+)",
    "vllm_commit":  r"\(commit:\s*([0-9a-f]+)\)",
}

# vLLM v0.19 memory profiling line (optional — only present in DEBUG logs).
# Multiple workers emit identical values; we take the first match.
# Example: "Total non KV cache memory: 17.12GiB; torch peak memory increase: 1.89GiB;
#           non-torch forward increase memory: 0.25GiB; weights memory: 14.99GiB."
_PROFILING_PATTERN = re.compile(
    r"Total non KV cache memory:\s*([\d.]+)GiB;"
    r"\s*torch peak memory increase:\s*([\d.]+)GiB;"
    r"\s*non-torch forward increase memory:\s*([\d.]+)GiB;"
    r"\s*weights memory:\s*([\d.]+)GiB"
)

_VLLM_BLOCK_SIZE = 16  # tokens per KV block, constant in vLLM v0.19.0


def parse(log_path: str | Path) -> dict[str, Any]:
    text = Path(log_path).read_text()
    result: dict[str, Any] = {}
    missing: list[str] = []

    for field, pattern in _FLOAT_PATTERNS.items():
        m = re.search(pattern, text)
        if m:
            result[field] = float(m.group(1))
        else:
            missing.append(field)

    for field, pattern in _INT_PATTERNS.items():
        m = re.search(pattern, text)
        if m:
            result[field] = int(m.group(1).replace(",", ""))
        else:
            missing.append(field)

    required = set(_FLOAT_PATTERNS) | set(_INT_PATTERNS)
    required -= {"cuda_graph_memory_gib", "max_concurrency"}  # optional
    actual_missing = [f for f in missing if f in required]
    if actual_missing:
        raise ValueError(
            f"Log {log_path} is missing required fields: {actual_missing}. "
            "Confirm vLLM started cleanly (no OOM/offload errors)."
        )

    for field, pattern in _OPT_STR_PATTERNS.items():
        m = re.search(pattern, text)
        result[field] = m.group(1) if m else None

    # Memory profiling breakdown (vLLM v0.19 DEBUG line, optional)
    mp = _PROFILING_PATTERN.search(text)
    if mp:
        result["total_non_kv_memory_gib"]      = float(mp.group(1))
        result["activation_memory_gib"]         = float(mp.group(2))
        result["non_torch_forward_memory_gib"]  = float(mp.group(3))
        # weights_memory from profiling line should match weight_memory_gib; keep for cross-check
        result["profiling_weights_memory_gib"]  = float(mp.group(4))

    # Derived fields
    if "kv_cache_tokens" in result:
        result["kv_cache_blocks"] = result["kv_cache_tokens"] // _VLLM_BLOCK_SIZE
    if "kv_cache_memory_gib" in result and "kv_cache_blocks" in result and result["kv_cache_blocks"] > 0:
        result["kv_block_size_bytes"] = int(
            result["kv_cache_memory_gib"] * (2**30) / result["kv_cache_blocks"]
        )

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    result = parse(args.log_path)
    output = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
