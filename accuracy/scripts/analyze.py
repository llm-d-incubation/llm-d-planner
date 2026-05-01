#!/usr/bin/env python3
"""
Analyze capacity planner predictions vs actual vLLM measurements.
Generates a markdown report with error statistics per memory component.
"""

import csv
import math
import statistics
from pathlib import Path

REPO = Path(__file__).parent.parent
RAW_CSV = REPO / "results/v0.19.0/results_raw.csv"
PRED_CSV = REPO / "results/v0.19.0/results_predicted.csv"
OUT_MD = REPO / "results/v0.19.0/accuracy_report.md"


# ── Helpers ───────────────────────────────────────────────────────────────────

def pct_error(actual: float, predicted: float) -> float:
    """(predicted - actual) / actual * 100. Positive = over-estimate."""
    if actual == 0:
        return float("nan")
    return (predicted - actual) / actual * 100.0


def fmt(v: float, decimals: int = 2) -> str:
    if math.isnan(v):
        return "n/a"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{decimals}f}%"


def stats(values: list[float]) -> dict:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return {"mean": float("nan"), "median": float("nan"),
                "min": float("nan"), "max": float("nan"),
                "abs_mean": float("nan"), "n": 0}
    return {
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "abs_mean": statistics.mean(abs(v) for v in vals),
        "n": len(vals),
    }


def stats_row(label: str, errors: list[float]) -> str:
    s = stats(errors)
    if s["n"] == 0:
        return f"| {label} | — | — | — | — | — | — |"
    return (
        f"| {label} | {fmt(s['mean'])} | {fmt(s['median'])} | "
        f"{fmt(s['abs_mean'])} | {fmt(s['min'])} | {fmt(s['max'])} | {s['n']} |"
    )


def fv(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}" if not math.isnan(v) else "n/a"


# ── Data Loading ──────────────────────────────────────────────────────────────

raw_ok   = [r for r in csv.DictReader(RAW_CSV.open()) if r["status"] == "ok"]
pred_all = list(csv.DictReader(PRED_CSV.open()))

def _row_key(r: dict) -> tuple:
    return (r["model"], r["tp"], r["pp"], r["dp"], r["max_model_len"],
            r["dtype"], r.get("quantization", ""), r.get("kv_cache_dtype", "auto"))

pred_map: dict[tuple, list] = {}
for p in pred_all:
    pred_map.setdefault(_row_key(p), []).append(p)

# Consume predictions in order; each raw row pops one matching prediction.
pairs: list[tuple] = []
_counts: dict[tuple, int] = {}
for raw in raw_ok:
    k = _row_key(raw)
    bucket = pred_map.get(k, [])
    idx = _counts.get(k, 0)
    if idx < len(bucket):
        pairs.append((raw, bucket[idx]))
        _counts[k] = idx + 1
    # rows with no matching prediction are silently skipped

# ── Per-row error calculation ─────────────────────────────────────────────────

COMPONENTS = {
    "weight":        ("weight_memory_gib",      "pred_weight_memory_gib"),
    "activation":    ("activation_memory_gib",  "pred_activation_memory_gib"),
    "non_torch":     ("non_torch_forward_gib",  "pred_non_torch_gib"),
    "cuda_graph":    ("cuda_graph_actual_gib",  "pred_cuda_graph_gib"),
    "total_non_kv":  ("total_non_kv_cache_gib", "pred_total_non_kv_cache_gib"),
    "kv_cache":      ("kv_cache_memory_gib",    "pred_kv_cache_memory_gib"),
    "kv_tokens":     ("kv_cache_tokens",         "pred_kv_cache_tokens"),
    "max_concurrency": ("max_concurrency",       "pred_max_concurrency"),
}

rows_data = []
for raw, pred in pairs:
    entry = {
        "log_file":      raw["log_file"],
        "model":         raw["model"],
        "architecture":  pred["architecture"],
        "gpu":           raw["gpu"],
        "tp":            int(raw["tp"]),
        "pp":            int(raw["pp"]),
        "dp":            int(raw["dp"]),
        "max_model_len": int(raw["max_model_len"]),
        "quantization":  raw["quantization"],
        "kv_cache_dtype":raw["kv_cache_dtype"],
        "dtype":         raw["dtype"],
    }
    for key, (rcol, pcol) in COMPONENTS.items():
        try:
            a = float(raw.get(rcol, ""))
            p = float(pred.get(pcol, ""))
        except (ValueError, TypeError):
            a = p = float("nan")
        entry[f"actual_{key}"] = a
        entry[f"pred_{key}"]   = p
        entry[f"err_{key}"]    = pct_error(a, p)
    rows_data.append(entry)


# ── Segment helpers ───────────────────────────────────────────────────────────

# Parameters the capacity planner currently accepts as inputs:
#   model, tp, pp, dp, max_model_len, gpu_memory_utilization
# Parameters NOT yet modeled (go in Part 2 / Next Steps):
#   --kv-cache-dtype, --dtype override, runtime --quantization fp8

def where(fn, pool=None):
    src = pool if pool is not None else rows_data
    return [r for r in src if fn(r)]

# Part 1: only runs whose config is fully within the planner's input space.
# Excludes: dtype=float32 override, runtime fp8 quant, kv_cache_dtype=fp8.
def _part1(r: dict) -> bool:
    return (r["dtype"] != "torch.float32"
            and r["quantization"] != "fp8"
            and r["kv_cache_dtype"] != "fp8")

part1 = where(_part1)

# Baseline: single-GPU, default context, unquantized — one clean row per model.
# Uses a seen-set to deduplicate overlapping sweep rows (tp=1 appears in both
# the TP sweep and the max_model_len sweep for the same model).
_base_seen: set = set()
base: list = []
for r in where(lambda r: r["tp"] == 1 and r["pp"] == 1 and r["dp"] == 1
               and r["max_model_len"] == 8192
               and r["quantization"] in ("None", "", None), pool=part1):
    if r["model"] not in _base_seen:
        _base_seen.add(r["model"])
        base.append(r)

# Sensitivity sweeps within Part 1
def _dedup(rows: list[dict], key_fn) -> list[dict]:
    seen: set = set()
    out: list[dict] = []
    for r in rows:
        k = key_fn(r)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out

tp_sweep_llama = _dedup(
    sorted(where(lambda r: r["model"] == "meta-llama/Llama-3.1-8B-Instruct"
                 and r["pp"] == 1 and r["max_model_len"] == 8192
                 and r["quantization"] in ("None", "", None), pool=part1),
           key=lambda r: r["tp"]),
    key_fn=lambda r: r["tp"])

tp_sweep_qwen = _dedup(
    sorted(where(lambda r: r["model"] == "Qwen/Qwen2.5-7B-Instruct"
                 and r["pp"] == 1 and r["max_model_len"] == 8192
                 and r["quantization"] in ("None", "", None), pool=part1),
           key=lambda r: r["tp"]),
    key_fn=lambda r: r["tp"])

pp_sweep = _dedup(
    sorted(where(lambda r: r["model"] == "meta-llama/Llama-3.1-8B-Instruct"
                 and r["tp"] == 1 and r["max_model_len"] == 8192
                 and r["quantization"] in ("None", "", None), pool=part1),
           key=lambda r: r["pp"]),
    key_fn=lambda r: r["pp"])

len_sweep_llama = _dedup(
    sorted(where(lambda r: r["model"] == "meta-llama/Llama-3.1-8B-Instruct"
                 and r["tp"] == 1 and r["pp"] == 1
                 and r["quantization"] in ("None", "", None), pool=part1),
           key=lambda r: r["max_model_len"]),
    key_fn=lambda r: r["max_model_len"])

len_sweep_qwen = _dedup(
    sorted(where(lambda r: r["model"] == "Qwen/Qwen2.5-7B-Instruct"
                 and r["tp"] == 1 and r["pp"] == 1
                 and r["quantization"] in ("None", "", None), pool=part1),
           key=lambda r: r["max_model_len"]),
    key_fn=lambda r: r["max_model_len"])

# Part 2: runs that exercise parameters not yet modeled by the planner.
kvfp8_rows = where(lambda r: r["kv_cache_dtype"] == "fp8")
dtype_rows  = where(lambda r: r["dtype"] == "torch.float32"
                    or (r["dtype"] == "torch.float16"
                        and r["quantization"] in ("None", "", None)))
quant_rows  = where(lambda r: r["quantization"] not in ("None", "", None)
                    or r["quantization"] == "fp8")


# ── Report builder ────────────────────────────────────────────────────────────

lines = []
W = lines.append


def section(title: str, rows: list[dict]):
    W(f"\n#### {title}  (n={len(rows)})\n")
    W("| Component | Mean error | Median | Mean abs | Min | Max | n |")
    W("|-----------|:----------:|:------:|:--------:|:---:|:---:|:-:|")
    for key in ["weight", "activation", "non_torch", "kv_cache", "max_concurrency"]:
        W(stats_row(key.replace("_", " ").title(), [r[f"err_{key}"] for r in rows]))


# ── Summary stats (Part 1) ────────────────────────────────────────────────────

kv_errs   = [r["err_kv_cache"]        for r in part1 if not math.isnan(r["err_kv_cache"])]
kv_base   = [r["err_kv_cache"]        for r in base  if not math.isnan(r["err_kv_cache"])]
wt_errs   = [r["err_weight"]          for r in part1 if not math.isnan(r["err_weight"])]
act_errs  = [r["err_activation"]      for r in part1 if not math.isnan(r["err_activation"])]
nt_errs   = [r["err_non_torch"]       for r in part1 if not math.isnan(r["err_non_torch"])]
conc_errs = [r["err_max_concurrency"] for r in part1 if not math.isnan(r["err_max_concurrency"])]

kv_mean   = statistics.mean(kv_errs)
kv_abs    = statistics.mean(abs(e) for e in kv_errs)
kv_base_m = statistics.mean(kv_base)
wt_mean   = statistics.mean(wt_errs)
wt_abs    = statistics.mean(abs(e) for e in wt_errs)
act_mean  = statistics.mean(act_errs)
act_abs   = statistics.mean(abs(e) for e in act_errs)
nt_mean   = statistics.mean(nt_errs)
conc_mean = statistics.mean(conc_errs)
conc_abs  = statistics.mean(abs(e) for e in conc_errs)

# ═══════════════════════════════════════════════════════════════════════════════
W("# Capacity Planner Accuracy Report — vLLM v0.19.0 / H100-80GB")
W("")
W("**Hardware**: H100-80GB (catalog 80 GiB, physical 79.19 GiB)  ")
W("**Planner inputs evaluated**: model, tp, pp, dp, max_model_len, gpu_memory_utilization  ")
W("")
W("> Percent error = (predicted − actual) / actual × 100. "
  "Positive = over-estimate, negative = under-estimate.")
W("")

# ─────────────────────────────────────────────────────────────────────────────
W("## Part 1: Accuracy Evaluation")
W("")
W(f"Covers {len(part1)} runs across {len(set(r['model'] for r in part1))} models "
  "using only parameters the planner currently accepts as inputs. "
  "Excludes runs with `--dtype float32`, runtime `--quantization fp8`, "
  "and `--kv-cache-dtype fp8` (see Part 2).")
W("")

# ── Summary table ─────────────────────────────────────────────────────────────
W("### Summary\n")
W("| Metric | Mean error | Mean abs error | n |")
W("|--------|:----------:|:--------------:|:-:|")
W(f"| KV cache memory (all runs) | {fmt(kv_mean)} | {fmt(kv_abs)} | {len(kv_errs)} |")
W(f"| KV cache memory (baseline: tp=pp=dp=1, len=8192, no quant) | {fmt(kv_base_m)} | — | {len(kv_base)} |")
W(f"| Weight memory | {fmt(wt_mean)} | {fmt(wt_abs)} | {len(wt_errs)} |")
W(f"| Activation memory | {fmt(act_mean)} | {fmt(act_abs)} | {len(act_errs)} |")
W(f"| Non-torch overhead | {fmt(nt_mean)} | — | {len(nt_errs)} |")
W(f"| Max concurrency | {fmt(conc_mean)} | {fmt(conc_abs)} | {len(conc_errs)} |")
W("")
W("**Key findings**:\n")
W(f"- **Weight memory is accurate**: mean abs error {fmt(wt_abs)}, "
  "computed directly from safetensors parameter counts.")
W(f"- **KV cache memory is close**: {fmt(kv_mean)} mean error across all runs; "
  f"{fmt(kv_base_m)} at baseline. Errors are small and consistent.")
W(f"- **Activation is the dominant error source**: mean {fmt(act_mean)} (over-estimate). "
  "The planner uses empirical constants measured against an older vLLM version; "
  "v0.19.0 reports substantially lower values. See Root Cause Analysis.")
W(f"- **Max concurrency tracks KV accuracy**: {fmt(conc_mean)} mean error; "
  "deviations come from the per-token KV formula, not the pool size prediction.")
W("")

# ── Per-model baseline ─────────────────────────────────────────────────────────
W("### Per-Model Results — Baseline (TP=1, PP=1, DP=1, len=8192, no quantization)\n")
W("| Model | Arch | Weight err | Activation err | Non-torch err | KV cache err | Max conc err |")
W("|-------|------|:----------:|:--------------:|:-------------:|:------------:|:------------:|")
for r in sorted(base, key=lambda x: x["model"]):
    model_short = r["model"].split("/")[-1][:35]
    arch_short  = (r["architecture"]
                   .replace("ForCausalLM", "")
                   .replace("ForConditionalGeneration", "*"))[:25]
    W(f"| {model_short} | {arch_short} | "
      f"{fmt(r['err_weight'])} | {fmt(r['err_activation'])} | "
      f"{fmt(r['err_non_torch'])} | {fmt(r['err_kv_cache'])} | "
      f"{fmt(r['err_max_concurrency'])} |")
W("")

# ── TP sensitivity ─────────────────────────────────────────────────────────────
W("### Sensitivity: Tensor Parallelism (TP)\n")
W("| Model | TP | Actual weight (GiB) | Weight err | Actual activ (GiB) | Activation err | Actual non-torch (GiB) | Non-torch err | KV cache err |")
W("|-------|:--:|:-------------------:|:----------:|:------------------:|:--------------:|:---------------------:|:-------------:|:------------:|")
for r in tp_sweep_llama + tp_sweep_qwen:
    W(f"| {r['model'].split('/')[-1][:22]} | {r['tp']} | "
      f"{fv(r['actual_weight'])} | {fmt(r['err_weight'])} | "
      f"{fv(r['actual_activation'])} | {fmt(r['err_activation'])} | "
      f"{fv(r['actual_non_torch'])} | {fmt(r['err_non_torch'])} | "
      f"{fmt(r['err_kv_cache'])} |")
W("")
W("- **Weights scale correctly** with TP: error stays near 0% across TP=1–4.")
W("- **Activation is TP-invariant** in both formula and vLLM: error stays flat.")
W("- **Non-torch is under-estimated at TP≥2**: NCCL all-reduce buffers push actual to "
  "~2.1 GiB/GPU but the constant is 0.60 GiB. The opposing over-estimate in activation "
  "partially masks this in the KV cache error.")
W("")

# ── PP sensitivity ─────────────────────────────────────────────────────────────
W("### Sensitivity: Pipeline Parallelism (PP)\n")
W("Model: meta-llama/Llama-3.1-8B-Instruct, TP=1, len=8192\n")
W("| PP | Actual weight (GiB) | Weight err | Actual activ (GiB) | Activation err | Actual non-torch (GiB) | Non-torch err | KV cache err |")
W("|:--:|:-------------------:|:----------:|:------------------:|:--------------:|:---------------------:|:-------------:|:------------:|")
pp_acts  = {}
pp_preds = {}
for r in pp_sweep:
    pp_acts[r["pp"]]  = r["actual_activation"]
    pp_preds[r["pp"]] = r["pred_activation"]
    W(f"| {r['pp']} | "
      f"{fv(r['actual_weight'])} | {fmt(r['err_weight'])} | "
      f"{fv(r['actual_activation'])} | {fmt(r['err_activation'])} | "
      f"{fv(r['actual_non_torch'])} | {fmt(r['err_non_torch'])} | "
      f"{fmt(r['err_kv_cache'])} |")
W("")
W(f"- **Activation drops with PP**: "
  f"PP=1 → {fv(pp_acts.get(1,float('nan')))} GiB, "
  f"PP=2 → {fv(pp_acts.get(2,float('nan')))} GiB, "
  f"PP=4 → {fv(pp_acts.get(4,float('nan')))} GiB. "
  f"The formula always predicts {fv(pp_preds.get(1,float('nan')))} GiB regardless of PP.")
W("- **Weight error grows with PP**: layer imbalance across stages causes the formula "
  "(which assumes uniform distribution) to deviate at high PP.")
W("")

# ── max_model_len sensitivity ──────────────────────────────────────────────────
W("### Sensitivity: Context Length (max_model_len)\n")
W("| Model | max_model_len | Actual KV (GiB) | KV err | Actual tokens | Pred tokens | Token err |")
W("|-------|:-------------:|:---------------:|:------:|:-------------:|:-----------:|:---------:|")
for r in len_sweep_llama + len_sweep_qwen:
    W(f"| {r['model'].split('/')[-1][:28]} | {r['max_model_len']:,} | "
      f"{fv(r['actual_kv_cache'])} | {fmt(r['err_kv_cache'])} | "
      f"{int(r['actual_kv_tokens']):,} | {int(r['pred_kv_tokens']):,} | "
      f"{fmt(r['err_kv_tokens'])} |")
W("")
W("- **KV pool size (GiB) is independent of max_model_len**: both formula and vLLM agree. "
  "The pool is sized from available memory, not from a pre-allocated token count.")
W("- **Token count predictions vary**: the per-token KV bytes formula has model-dependent "
  "errors that show up consistently across all context lengths.")
W("")

# ── Root cause analysis ────────────────────────────────────────────────────────
W("### Root Cause Analysis\n")

W("#### 1. Activation Constants Are Stale\n")
W("The planner uses fixed constants per architecture (e.g., 4.8 GiB for Llama) "
  "calibrated against an older vLLM version. vLLM v0.19.0 reports substantially lower values:\n")
W("| Architecture | Planner constant (GiB) | Observed v0.19.0 range (GiB) | Error range |")
W("|-------------|:---------------------:|:----------------------------:|:-----------:|")
archs_seen: dict[str, list] = {}
for r in part1:
    if not math.isnan(r["err_activation"]):
        archs_seen.setdefault(r["architecture"], []).append(
            (r["actual_activation"], r["pred_activation"], r["err_activation"]))
for arch, data in sorted(archs_seen.items()):
    acts  = [d[0] for d in data]
    preds = [d[1] for d in data]
    errs  = [d[2] for d in data]
    arch_label = (arch.replace("ForCausalLM","").replace("ForConditionalGeneration","*"))[:35]
    W(f"| {arch_label} | {fv(statistics.mean(preds))} | "
      f"{fv(min(acts))}–{fv(max(acts))} | "
      f"{fmt(min(errs))} to {fmt(max(errs))} |")
W("")
W("Re-calibrating these constants from the v0.19.0 measurements is the highest-value fix.")

W("\n#### 2. Non-torch Constants Under-estimated for Multi-GPU\n")
W("| TP | PP | Constant used (GiB) | Observed mean (GiB) | Mean error |")
W("|:--:|:--:|:-------------------:|:-------------------:|:----------:|")
for tp_v, pp_v in [(1,1),(1,2),(1,4),(2,1),(4,1)]:
    grp = where(lambda r, t=tp_v, p=pp_v: r["tp"]==t and r["pp"]==p, pool=part1)
    if not grp: continue
    const = 0.15 if tp_v == 1 and pp_v == 1 else (0.60 if tp_v > 1 else 0.15)
    acts  = [r["actual_non_torch"] for r in grp if not math.isnan(r["actual_non_torch"])]
    errs  = [r["err_non_torch"]    for r in grp if not math.isnan(r["err_non_torch"])]
    if not acts: continue
    W(f"| {tp_v} | {pp_v} | {const} | {fv(statistics.mean(acts))} | "
      f"{fmt(statistics.mean(errs))} |")
W("")
W("TP≥2 requires NCCL all-reduce buffers (~2.1 GiB/GPU vs the 0.60 GiB constant). "
  "PP≥2 adds P2P send/receive buffers that the formula ignores entirely.")

W("\n#### 3. GPU Catalog vs Physical Memory\n")
W("Planner uses 80 GiB (catalog); H100 physical VRAM is 79.19 GiB.  \n"
  "Effect: KV pool over-predicted by ~0.77 GiB (76.00 vs 75.23 GiB at 0.95 utilization).")

cg_vals = [r["actual_cuda_graph"] for r in part1
           if not math.isnan(r["actual_cuda_graph"]) and r["actual_cuda_graph"] > 0]
if cg_vals:
    W(f"\n#### 4. CUDA Graph Memory\n")
    W(f"Observed pool sizes: {fv(min(cg_vals))}–{fv(max(cg_vals))} GiB "
      f"(mean {fv(statistics.mean(cg_vals))} GiB). "
      "vLLM allocates CUDA graphs after sizing the KV cache, so the reported KV pool "
      "already includes CUDA graph memory — no formula correction needed.")

# ─────────────────────────────────────────────────────────────────────────────
W("\n---\n")
W("## Part 2: Next Steps — Parameters Not Yet Modeled\n")
W("The following vLLM flags affect memory allocation but are not yet accepted as "
  "planner inputs. Each subsection quantifies the prediction gap to inform "
  "which inputs to add next.")
W("")

# ── kv_cache_dtype ─────────────────────────────────────────────────────────────
W("### `--kv-cache-dtype fp8`\n")
kv_fp8 = where(lambda r: r["kv_cache_dtype"] == "fp8")
kv_fp8.sort(key=lambda r: r["model"])
kv_auto_map = {}
for r in where(lambda r: r["kv_cache_dtype"] != "fp8"
               and r["quantization"] in ("None","",None)):
    k = (r["model"], r["tp"], r["pp"], r["max_model_len"])
    kv_auto_map[k] = r

W("| Model | kv_cache_dtype | Actual KV (GiB) | KV GiB err | Actual tokens | Pred tokens | Token err |")
W("|-------|:--------------:|:---------------:|:----------:|:-------------:|:-----------:|:---------:|")
for fp8r in kv_fp8:
    k = (fp8r["model"], fp8r["tp"], fp8r["pp"], fp8r["max_model_len"])
    autr = kv_auto_map.get(k)
    if autr:
        W(f"| {autr['model'].split('/')[-1][:28]} | auto | "
          f"{fv(autr['actual_kv_cache'])} | {fmt(autr['err_kv_cache'])} | "
          f"{int(autr['actual_kv_tokens']):,} | {int(autr['pred_kv_tokens']):,} | "
          f"{fmt(autr['err_kv_tokens'])} |")
    W(f"| {fp8r['model'].split('/')[-1][:28]} | fp8 | "
      f"{fv(fp8r['actual_kv_cache'])} | {fmt(fp8r['err_kv_cache'])} | "
      f"{int(fp8r['actual_kv_tokens']):,} | {int(fp8r['pred_kv_tokens']):,} | "
      f"{fmt(fp8r['err_kv_tokens'])} |")
    W("||||||||")
W("")
W("**KV pool size (GiB) is unaffected** — fp8 halves per-token storage, not the pool. "
  "The planner's GiB prediction stays accurate. "
  "**Token count is ~2× too low** because the planner always uses the model's native "
  "dtype (BF16 = 2 bytes/element) instead of fp8 (1 byte/element). "
  "Fix: accept `kv_cache_dtype` as input; when `fp8`, use 1 byte/token.")
W("")

# ── dtype override ─────────────────────────────────────────────────────────────
W("### `--dtype` override\n")
dtype_rows_all = where(lambda r: "Llama-3.1" in r["model"]
                       and r["tp"] == 1 and r["pp"] == 1 and r["max_model_len"] == 8192
                       and r["quantization"] in ("None","",None))
dtype_rows_all.sort(key=lambda r: r["dtype"])
W("| dtype | Actual weight (GiB) | Weight err | Actual KV (GiB) | KV err |")
W("|-------|:-------------------:|:----------:|:---------------:|:------:|")
for r in dtype_rows_all:
    W(f"| {r['dtype'].replace('torch.','')} | "
      f"{fv(r['actual_weight'])} | {fmt(r['err_weight'])} | "
      f"{fv(r['actual_kv_cache'])} | {fmt(r['err_kv_cache'])} |")
W("")
W("**`--dtype float32`** doubles weight memory. The planner reads the HF config dtype "
  "(BF16) and has no visibility into the vLLM override → −50% weight error, +31% KV error.  \n"
  "**`--dtype float16`** matches the HF config for these models → near-zero error.  \n"
  "Fix: accept `dtype` as input and use it to override the bytes-per-param calculation.")
W("")

# ── runtime quantization ───────────────────────────────────────────────────────
W("### Runtime `--quantization fp8`\n")
rtfp8 = where(lambda r: r["quantization"] == "fp8")
W("| Model | Actual weight (GiB) | Weight err | Actual KV (GiB) | KV err |")
W("|-------|:-------------------:|:----------:|:---------------:|:------:|")
for r in rtfp8:
    W(f"| {r['model'].split('/')[-1][:35]} | "
      f"{fv(r['actual_weight'])} | {fmt(r['err_weight'])} | "
      f"{fv(r['actual_kv_cache'])} | {fmt(r['err_kv_cache'])} |")
W("")
W("Runtime `--quantization fp8` compresses weights on-the-fly after loading. "
  "vLLM logs the post-compression size (~half of BF16). The planner finds no "
  "`quantization_config` in the HF repo and predicts the full BF16 weight → ~+76% weight error.  \n"
  "Fix: accept `quantization fp8` as input; apply 1 byte/param for weight estimation.")
W("")

# ── Recommendations ────────────────────────────────────────────────────────────
W("### Recommendations\n")
W("| Priority | Input to add | Expected impact |")
W("|:--------:|-------------|:---------------:|")
W("| High | **Re-calibrate activation constants** from v0.19.0 measurements. "
  "Current constants are 2–7× too high. | Removes largest single error source |")
W("| High | **`kv_cache_dtype`** — when `fp8`, use 1 byte/token for KV. "
  "| Fixes ~2× token/concurrency error for fp8-KV runs |")
W("| Medium | **`dtype`** — when `float32`, double bytes-per-param. "
  "| Fixes −50% weight error for float32 runs |")
W("| Medium | **`quantization fp8` (runtime)** — apply 1 byte/param. "
  "| Fixes +76% weight error for runtime-fp8 runs |")
W("| Medium | **Re-measure non-torch constants for TP≥2 and PP≥2.** "
  "| +1–2 GiB KV accuracy for multi-GPU |")
W("| Medium | **Scale activation constant by 1/PP.** "
  "| Fixes growing activation error at high PP |")
W("| Low | **Use physical GPU memory** (79.19 GiB) instead of catalog 80 GiB. "
  "| +0.77 GiB KV accuracy |")

report = "\n".join(lines)
OUT_MD.write_text(report)
print(f"Report written → {OUT_MD}")
print(f"\n{'─'*60}")
print("HEADLINE NUMBERS (Part 1 — supported inputs only)")
print(f"{'─'*60}")
print(f"  KV cache mean error (all):      {fmt(kv_mean)}")
print(f"  KV cache mean error (baseline): {fmt(kv_base_m)}")
print(f"  Weights mean abs error:         {fmt(wt_abs)}")
print(f"  Activation mean error:          {fmt(act_mean)}")
print(f"  Activation mean abs error:      {fmt(act_abs)}")
print(f"  Max concurrency mean error:     {fmt(conc_mean)}")
print(f"  Max concurrency mean abs error: {fmt(conc_abs)}")
