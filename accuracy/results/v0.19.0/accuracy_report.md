# Capacity Planner Accuracy Report — vLLM v0.19.0 / H100-80GB

**Hardware**: H100-80GB (catalog 80 GiB, physical 79.19 GiB)  
**Planner inputs evaluated**: model, tp, pp, dp, max_model_len, gpu_memory_utilization  

> Percent error = (predicted − actual) / actual × 100. Positive = over-estimate, negative = under-estimate.

## Part 1: Accuracy Evaluation

Covers 50 runs across 29 models using only parameters the planner currently accepts as inputs. Excludes runs with `--dtype float32`, runtime `--quantization fp8`, and `--kv-cache-dtype fp8` (see Part 2).

### Summary

| Metric | Mean error | Mean abs error | n |
|--------|:----------:|:--------------:|:-:|
| KV cache memory (all runs) | +0.28% | +6.70% | 50 |
| KV cache memory (baseline: tp=pp=dp=1, len=8192, no quant) | -5.29% | — | 19 |
| Weight memory | -0.75% | +0.75% | 50 |
| Activation memory | +188.64% | +188.64% | 50 |
| Non-torch overhead | -43.54% | — | 50 |
| Max concurrency | -2.35% | +9.90% | 50 |

**Key findings**:

- **Weight memory is accurate**: mean abs error +0.75%, computed directly from safetensors parameter counts.
- **KV cache memory is close**: +0.28% mean error across all runs; -5.29% at baseline. Errors are small and consistent.
- **Activation is the dominant error source**: mean +188.64% (over-estimate). The planner uses empirical constants measured against an older vLLM version; v0.19.0 reports substantially lower values. See Root Cause Analysis.
- **Max concurrency tracks KV accuracy**: -2.35% mean error; deviations come from the per-token KV formula, not the pool size prediction.

### Per-Model Results — Baseline (TP=1, PP=1, DP=1, len=8192, no quantization)

| Model | Arch | Weight err | Activation err | Non-torch err | KV cache err | Max conc err |
|-------|------|:----------:|:--------------:|:-------------:|:------------:|:------------:|
| Qwen2.5-7B-Instruct | Qwen2 | -0.45% | +153.39% | -37.50% | -4.21% | -4.22% |
| Qwen3-30B-A3B | Qwen3Moe | -0.02% | +198.51% | -44.44% | -28.75% | -28.72% |
| Qwen3-8B | Qwen3 | -0.09% | +153.39% | -40.00% | -4.36% | -4.36% |
| CodeLlama-7b-hf | Llama | -0.07% | +523.38% | -40.00% | -5.13% | -5.13% |
| DeepSeek-V2-Lite-Chat | DeepseekV2 | -0.59% | +314.51% | -42.31% | -11.50% | -11.50% |
| gemma-2-27b-it | Gemma2 | -0.01% | +50.27% | -42.31% | -4.64% | -4.61% |
| gemma-2-2b-it | Gemma2 | -0.62% | +51.93% | -37.50% | -1.49% | -1.39% |
| gemma-2-9b-it | Gemma2 | -0.03% | +50.68% | -40.00% | -1.82% | -1.75% |
| gemma-3-12b-it | Gemma3* | -2.61% | +39.59% | -40.00% | -0.15% | +0.00% |
| gemma-3-27b-it | Gemma3* | -0.69% | +37.84% | -42.31% | -1.42% | +11.43% |
| gemma-3-4b-it | Gemma3* | -6.65% | +41.39% | -40.00% | -0.27% | +2.84% |
| granite-3.1-2b-instruct | Granite | -0.44% | +633.33% | -67.39% | -5.27% | -5.27% |
| granite-3.1-8b-instruct | Granite | -0.20% | +547.06% | -67.39% | -6.02% | -6.03% |
| granite-3.3-8b-instruct | Granite | -0.20% | +547.06% | -67.39% | -6.02% | -6.03% |
| granite-vision-3.3-2b | LlavaNext* | +0.04% | +216.46% | -40.00% | -1.23% | -1.23% |
| Llama-3.1-8B-Instruct | Llama | -0.22% | +153.97% | -40.00% | -3.47% | -3.48% |
| phi-4 | Phi3 | -0.31% | +261.84% | -40.00% | -6.59% | -6.58% |
| Mistral-Small-3.1-24B-Instruct-2503 | Mistral3* | -0.08% | +23.15% | -40.00% | +1.54% | +1.55% |
| Kimi-VL-A3B-Instruct | KimiVL* | -0.58% | +173.97% | -40.00% | -9.76% | -87.31% |

### Sensitivity: Tensor Parallelism (TP)

| Model | TP | Actual weight (GiB) | Weight err | Actual activ (GiB) | Activation err | Actual non-torch (GiB) | Non-torch err | KV cache err |
|-------|:--:|:-------------------:|:----------:|:------------------:|:--------------:|:---------------------:|:-------------:|:------------:|
| Llama-3.1-8B-Instruct | 1 | 14.99 | -0.22% | 1.89 | +153.97% | 0.25 | -40.00% | -3.47% |
| Llama-3.1-8B-Instruct | 2 | 7.51 | -0.42% | 1.89 | +153.97% | 2.07 | -71.01% | +2.76% |
| Llama-3.1-8B-Instruct | 4 | 3.77 | -0.81% | 1.89 | +153.97% | 2.13 | -71.83% | +4.48% |
| Qwen2.5-7B-Instruct | 1 | 14.25 | -0.45% | 2.21 | +153.39% | 0.24 | -37.50% | -4.21% |
| Qwen2.5-7B-Instruct | 2 | 7.12 | -0.38% | 2.21 | +153.39% | 2.06 | -70.87% | +2.61% |
| Qwen2.5-7B-Instruct | 4 | 3.55 | -0.10% | 2.21 | +153.39% | 2.13 | -71.83% | +4.62% |

- **Weights scale correctly** with TP: error stays near 0% across TP=1–4.
- **Activation is TP-invariant** in both formula and vLLM: error stays flat.
- **Non-torch is under-estimated at TP≥2**: NCCL all-reduce buffers push actual to ~2.1 GiB/GPU but the constant is 0.60 GiB. The opposing over-estimate in activation partially masks this in the KV cache error.

### Sensitivity: Pipeline Parallelism (PP)

Model: meta-llama/Llama-3.1-8B-Instruct, TP=1, len=8192

| PP | Actual weight (GiB) | Weight err | Actual activ (GiB) | Activation err | Actual non-torch (GiB) | Non-torch err | KV cache err |
|:--:|:-------------------:|:----------:|:------------------:|:--------------:|:---------------------:|:-------------:|:------------:|
| 1 | 14.99 | -0.22% | 1.89 | +153.97% | 0.25 | -40.00% | -3.47% |
| 2 | 7.51 | -0.42% | 1.10 | +336.36% | 0.07 | +114.29% | -0.85% |
| 4 | 4.26 | -12.22% | 1.05 | +357.14% | 0.07 | +114.29% | +1.59% |

- **Activation drops with PP**: PP=1 → 1.89 GiB, PP=2 → 1.10 GiB, PP=4 → 1.05 GiB. The formula always predicts 4.80 GiB regardless of PP.
- **Weight error grows with PP**: layer imbalance across stages causes the formula (which assumes uniform distribution) to deviate at high PP.

### Sensitivity: Context Length (max_model_len)

| Model | max_model_len | Actual KV (GiB) | KV err | Actual tokens | Pred tokens | Token err |
|-------|:-------------:|:---------------:|:------:|:-------------:|:-----------:|:---------:|
| Llama-3.1-8B-Instruct | 2,048 | 58.11 | -3.47% | 476,016 | 459,509 | -3.47% |
| Llama-3.1-8B-Instruct | 4,096 | 58.11 | -3.47% | 476,016 | 459,509 | -3.47% |
| Llama-3.1-8B-Instruct | 8,192 | 58.11 | -3.47% | 476,000 | 459,509 | -3.46% |
| Llama-3.1-8B-Instruct | 16,384 | 58.11 | -3.47% | 476,016 | 459,509 | -3.47% |
| Llama-3.1-8B-Instruct | 32,768 | 58.11 | -3.47% | 476,016 | 459,509 | -3.47% |
| Qwen2.5-7B-Instruct | 2,048 | 58.53 | -4.21% | 1,096,000 | 1,049,789 | -4.22% |
| Qwen2.5-7B-Instruct | 4,096 | 58.53 | -4.21% | 1,096,000 | 1,049,789 | -4.22% |
| Qwen2.5-7B-Instruct | 8,192 | 58.53 | -4.21% | 1,096,000 | 1,049,789 | -4.22% |
| Qwen2.5-7B-Instruct | 16,384 | 58.53 | -4.21% | 1,095,968 | 1,049,789 | -4.21% |
| Qwen2.5-7B-Instruct | 32,768 | 58.53 | -4.21% | 1,096,000 | 1,049,789 | -4.22% |

- **KV pool size (GiB) is independent of max_model_len**: both formula and vLLM agree. The pool is sized from available memory, not from a pre-allocated token count.
- **Token count predictions vary**: the per-token KV bytes formula has model-dependent errors that show up consistently across all context lengths.

### Root Cause Analysis

#### 1. Activation Constants Are Stale

The planner uses fixed constants per architecture (e.g., 4.8 GiB for Llama) calibrated against an older vLLM version. vLLM v0.19.0 reports substantially lower values:

| Architecture | Planner constant (GiB) | Observed v0.19.0 range (GiB) | Error range |
|-------------|:---------------------:|:----------------------------:|:-----------:|
| DeepseekV2 | 8.00 | 1.93–1.93 | +314.51% to +314.51% |
| Gemma2 | 5.50 | 3.62–3.66 | +50.27% to +51.93% |
| Gemma3* | 5.50 | 3.89–3.99 | +37.84% to +41.39% |
| Granite | 5.50 | 0.75–0.85 | +547.06% to +633.33% |
| KimiVL* | 8.00 | 2.85–2.92 | +173.97% to +180.70% |
| Llama | 4.80 | 0.77–1.97 | +143.65% to +523.38% |
| LlavaNext* | 2.50 | 0.79–0.79 | +216.46% to +216.46% |
| Mistral3* | 2.50 | 2.03–2.18 | +14.68% to +23.15% |
| Mixtral | 8.00 | 1.21–1.21 | +561.16% to +561.16% |
| Phi3 | 5.50 | 1.52–1.52 | +261.84% to +261.84% |
| Qwen2 | 5.60 | 2.21–2.29 | +144.54% to +153.39% |
| Qwen3 | 5.60 | 2.21–2.21 | +153.39% to +153.39% |
| Qwen3Moe | 8.00 | 2.68–2.68 | +198.51% to +198.51% |

Re-calibrating these constants from the v0.19.0 measurements is the highest-value fix.

#### 2. Non-torch Constants Under-estimated for Multi-GPU

| TP | PP | Constant used (GiB) | Observed mean (GiB) | Mean error |
|:--:|:--:|:-------------------:|:-------------------:|:----------:|
| 1 | 1 | 0.15 | 0.27 | -42.17% |
| 1 | 2 | 0.15 | 0.07 | +114.29% |
| 1 | 4 | 0.15 | 0.07 | +114.29% |
| 2 | 1 | 0.6 | 2.08 | -71.17% |
| 4 | 1 | 0.6 | 2.17 | -72.34% |

TP≥2 requires NCCL all-reduce buffers (~2.1 GiB/GPU vs the 0.60 GiB constant). PP≥2 adds P2P send/receive buffers that the formula ignores entirely.

#### 3. GPU Catalog vs Physical Memory

Planner uses 80 GiB (catalog); H100 physical VRAM is 79.19 GiB.  
Effect: KV pool over-predicted by ~0.77 GiB (76.00 vs 75.23 GiB at 0.95 utilization).

#### 4. CUDA Graph Memory

Observed pool sizes: 0.51–1.85 GiB (mean 1.03 GiB). vLLM allocates CUDA graphs after sizing the KV cache, so the reported KV pool already includes CUDA graph memory — no formula correction needed.

---

## Part 2: Next Steps — Parameters Not Yet Modeled

The following vLLM flags affect memory allocation but are not yet accepted as planner inputs. Each subsection quantifies the prediction gap to inform which inputs to add next.

### `--kv-cache-dtype fp8`

| Model | kv_cache_dtype | Actual KV (GiB) | KV GiB err | Actual tokens | Pred tokens | Token err |
|-------|:--------------:|:---------------:|:----------:|:-------------:|:-----------:|:---------:|
| Qwen2.5-7B-Instruct | auto | 58.53 | -4.21% | 1,096,000 | 1,049,789 | -4.22% |
| Qwen2.5-7B-Instruct | fp8 | 58.53 | -4.21% | 2,192,000 | 1,049,789 | -52.11% |
||||||||
| Llama-3.1-8B-Instruct | auto | 58.11 | -3.47% | 476,016 | 459,509 | -3.47% |
| Llama-3.1-8B-Instruct | fp8 | 58.11 | -3.47% | 952,032 | 459,509 | -51.73% |
||||||||

**KV pool size (GiB) is unaffected** — fp8 halves per-token storage, not the pool. The planner's GiB prediction stays accurate. **Token count is ~2× too low** because the planner always uses the model's native dtype (BF16 = 2 bytes/element) instead of fp8 (1 byte/element). Fix: accept `kv_cache_dtype` as input; when `fp8`, use 1 byte/token.

### `--dtype` override

| dtype | Actual weight (GiB) | Weight err | Actual KV (GiB) | KV err |
|-------|:-------------------:|:----------:|:---------------:|:------:|
| bfloat16 | 14.99 | -0.22% | 58.11 | -3.47% |
| bfloat16 | 14.99 | -0.22% | 58.11 | -3.47% |
| bfloat16 | 14.99 | -0.22% | 58.11 | -3.47% |
| float16 | 14.99 | -0.22% | 58.11 | -3.47% |
| float32 | 29.98 | -50.11% | 42.80 | +31.06% |

**`--dtype float32`** doubles weight memory. The planner reads the HF config dtype (BF16) and has no visibility into the vLLM override → −50% weight error, +31% KV error.  
**`--dtype float16`** matches the HF config for these models → near-zero error.  
Fix: accept `dtype` as input and use it to override the bytes-per-param calculation.

### Runtime `--quantization fp8`

| Model | Actual weight (GiB) | Weight err | Actual KV (GiB) | KV err |
|-------|:-------------------:|:----------:|:---------------:|:------:|
| Llama-3.1-8B-Instruct | 8.49 | +76.18% | 64.61 | -13.18% |

Runtime `--quantization fp8` compresses weights on-the-fly after loading. vLLM logs the post-compression size (~half of BF16). The planner finds no `quantization_config` in the HF repo and predicts the full BF16 weight → ~+76% weight error.  
Fix: accept `quantization fp8` as input; apply 1 byte/param for weight estimation.

### Recommendations

| Priority | Input to add | Expected impact |
|:--------:|-------------|:---------------:|
| High | **Re-calibrate activation constants** from v0.19.0 measurements. Current constants are 2–7× too high. | Removes largest single error source |
| High | **`kv_cache_dtype`** — when `fp8`, use 1 byte/token for KV. | Fixes ~2× token/concurrency error for fp8-KV runs |
| Medium | **`dtype`** — when `float32`, double bytes-per-param. | Fixes −50% weight error for float32 runs |
| Medium | **`quantization fp8` (runtime)** — apply 1 byte/param. | Fixes +76% weight error for runtime-fp8 runs |
| Medium | **Re-measure non-torch constants for TP≥2 and PP≥2.** | +1–2 GiB KV accuracy for multi-GPU |
| Medium | **Scale activation constant by 1/PP.** | Fixes growing activation error at high PP |
| Low | **Use physical GPU memory** (79.19 GiB) instead of catalog 80 GiB. | +0.77 GiB KV accuracy |