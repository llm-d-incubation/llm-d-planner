# Capacity Planner Accuracy Report

**Hardware**: H100-80GB (catalog 80 GiB, physical 79.19 GiB)  
**vLLM version evaluated**: v0.19.0  
**Planner inputs evaluated**: model, tp, pp, dp, max_model_len, gpu_memory_utilization  

> Percent error = (predicted − actual) / actual × 100. Positive = over-estimate, negative = under-estimate.

---

## Part 1: Accuracy Evaluation — vLLM v0.19.0

Covers 49 runs across 28 models using only parameters the planner currently accepts as inputs. Excludes runs with `--dtype float32`, runtime `--quantization fp8`, and `--kv-cache-dtype fp8` (see Part 3). Additional models appear in the per-model and run matrix tables below but are not included in aggregate statistics because predictions have not yet been generated for them.

### Summary

| Metric | Mean error | Mean abs error | n |
|--------|:----------:|:--------------:|:-:|
| KV cache memory (all runs) | +0.89% | +7.82% | 49 |
| KV cache memory (baseline: tp=pp=dp=1, len=8192, no quant) | -6.96% | — | 15 |
| Weight memory | -0.84% | +0.84% | 49 |
| Activation memory | +212.88% | +212.88% | 49 |
| Non-torch overhead | -44.81% | — | 49 |
| Max concurrency | +3.68% | +16.65% | 49 |

**Key findings**:

- **Weight memory is accurate**: mean abs error +0.84%, computed directly from safetensors parameter counts.
- **KV cache memory is close**: +0.89% mean error across all runs; -6.96% at baseline. Errors are small and consistent.
- **Activation is the dominant error source**: mean +212.88% (over-estimate). The planner uses empirical constants calibrated against vLLM v0.16.0 or earlier; v0.17.0 introduced a ~60% reduction in reported activation overhead that persists through v0.19.0. See Part 2.
- **Max concurrency tracks KV accuracy**: +3.68% mean error; deviations come from the per-token KV formula, not the pool size prediction.

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
| gemma-7b | Gemma | -0.05% | +51.52% | -40.00% | -1.79% | -1.77% |
| granite-3.1-2b-instruct | Granite | -0.44% | +633.33% | -67.39% | -5.27% | -5.27% |
| granite-3.1-8b-instruct | Granite | -0.20% | +547.06% | -67.39% | -6.02% | -6.03% |
| granite-3.3-8b-instruct | Granite | -0.20% | +547.06% | -67.39% | -6.02% | -6.03% |
| granite-vision-3.3-2b | LlavaNext* | +0.04% | +216.46% | -40.00% | -1.23% | -1.23% |
| Llama-3.1-8B-Instruct | Llama | -0.22% | +153.97% | -40.00% | -3.47% | -3.48% |
| phi-4 | Phi3 | -0.31% | +261.84% | -40.00% | -6.59% | -6.58% |
| Mistral-Small-3.1-24B-Instruct-2503 | Mistral3* | -0.08% | +23.15% | -40.00% | +1.54% | +1.55% |
| Qwen1.5-MoE-A2.7B | Qwen2Moe | -0.02% | +223.89% | -40.00% | -10.14% | -10.12% |
| Kimi-VL-A3B-Instruct | KimiVL* | -0.58% | +173.97% | -40.00% | -9.76% | -87.31% |
| MiMo-VL-7B-SFT | Qwen2_5_VL* | -0.82% | +120.88% | -40.00% | -3.54% | -3.54% |

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
- **Non-torch is under-estimated at TP≥2**: NCCL all-reduce buffers push actual to ~2.1 GiB/GPU but the constant is 0.60 GiB.

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

- **KV pool size (GiB) is independent of max_model_len**: both formula and vLLM agree. The pool is sized from available memory, not a pre-allocated token count.
- **Token count error is model-dependent but context-length-stable**: the per-token KV bytes formula has architecture-specific errors that are consistent across all context lengths.

### Root Cause Analysis

#### 1. Activation Constants Are Stale (Primary Error Source)

The planner uses fixed constants per architecture calibrated against vLLM v0.16.0 or earlier. vLLM v0.17.0 reduced reported activation overhead by ~60% (see Part 2: Version Sensitivity), and this lower level persists through v0.19.0.

| Architecture | Planner constant (GiB) | Observed v0.19.0 range (GiB) | Error range |
|-------------|:---------------------:|:----------------------------:|:-----------:|
| DeepseekV2 | 8.00 | 1.93 | +314.51% |
| Gemma2 | 5.50 | 3.62–3.66 | +50.27% to +51.93% |
| Gemma3* | 5.50 | 3.89–3.99 | +37.84% to +41.39% |
| Gemma | 5.50 | 3.63 | +51.52% |
| GptOss | 8.00 | 2.87 | +178.75% |
| Granite | 5.50 | 0.75–0.85 | +547.06% to +633.33% |
| KimiVL* | 8.00 | 2.85–2.92 | +173.97% to +180.70% |
| Llama | 4.80 | 0.77–1.97 | +143.65% to +523.38% |
| Llama4* | 8.00 | 3.19 | +150.78% |
| LlavaNext* | 2.50 | 0.79 | +216.46% |
| Mistral3* | 2.50 | 2.03–2.18 | +14.68% to +23.15% |
| Mixtral | 8.00 | 1.21 | +561.16% |
| Phi3 | 5.50 | 1.52 | +261.84% |
| Phi | 5.50 | 0.79 | +596.20% |
| Qwen2 | 5.60 | 2.21–2.29 | +144.54% to +153.39% |
| Qwen3 | 5.60 | 2.21 | +153.39% |
| Qwen2_5_VL* | 5.50 | 2.49 | +120.88% |
| Qwen2Moe | 8.00 | 2.47 | +223.89% |
| Qwen3Moe | 8.00 | 2.68 | +198.51% |

Re-calibrating these constants against v0.19.0 measurements is the highest-value fix. See Part 2 for the version history showing when constants became stale.

#### 2. Non-torch Constants Under-estimated for Multi-GPU

| TP | PP | Constant used (GiB) | Observed mean (GiB) | Mean error |
|:--:|:--:|:-------------------:|:-------------------:|:----------:|
| 1 | 1 | 0.15 | 0.27 | -42.17% |
| 1 | 2 | 0.15 | 0.07 | +114.29% |
| 1 | 4 | 0.15 | 0.07 | +114.29% |
| 2 | 1 | 0.60 | 2.08 | -71.15% |
| 4 | 1 | 0.60 | 2.17 | -72.34% |

TP≥2 requires NCCL all-reduce buffers (~2.1 GiB/GPU vs the 0.60 GiB constant). PP≥2 adds P2P send/receive buffers that the formula ignores entirely.

#### 3. GPU Catalog vs Physical Memory

Planner uses 80 GiB (catalog); H100 physical VRAM is 79.19 GiB.  
Effect: KV pool over-predicted by ~0.77 GiB (76.00 vs 75.23 GiB at 0.95 utilization).

#### 4. CUDA Graph Memory

Observed pool sizes: 0.51–1.85 GiB (mean 1.04 GiB). vLLM allocates CUDA graphs after sizing the KV cache, so the reported KV pool already includes CUDA graph memory — no formula correction needed.

---

## Part 2: vLLM Version Sensitivity

**Goal**: Determine when the planner's activation constants became stale by measuring activation memory across vLLM releases.

**Model**: `Qwen/Qwen3-14B` — tp=1, pp=1, dp=1, max_model_len=8192, dtype=auto, no quantization.  
**Hardware**: H100-80GB, gpu_memory_utilization=0.95.

### Results

| vLLM version | Weight (GiB) | Activation (GiB) | Non-torch (GiB) | KV cache (GiB) | Max concurrency |
|:---:|:---:|:---:|:---:|:---:|:---:|
| v0.15.0 | 27.52 | 5.64 | 0.13 | 41.94 | 33.55 |
| v0.16.0 | 27.52 | 5.64 | 0.13 | 41.94 | 33.55 |
| **v0.17.0** | 27.52 | **2.23** | 0.13 | 45.34 | 36.27 |
| v0.18.0 | 27.52 | 2.23 | 0.25 | 45.23 | 36.18 |
| v0.19.0† | 27.52 | ~2.21 | 0.25 | ~45.2 | ~36.2 |

†v0.19.0 values extrapolated from Qwen3-8B (same architecture, scaled); Qwen3-14B not directly measured at v0.19.0.

**Key findings**:

- **Activation dropped 60% between v0.16.0 and v0.17.0**: 5.64 GiB → 2.23 GiB. This is the point at which all planner constants became stale.
- **Weight memory is stable across all versions**: model parameters are version-independent.
- **KV cache increased ~3.4 GiB at v0.17.0+**: the freed activation overhead is reallocated to the KV pool, increasing max concurrency by ~8%.
- **Non-torch increased at v0.18.0**: 0.13 → 0.25 GiB; likely new runtime bookkeeping overhead.
- **The planner's Qwen3 constant (5.60 GiB) matches v0.16.0 exactly** — constants were last calibrated against v0.16.0 or earlier.

### Implication for Constant Re-calibration

The activation drop at v0.17.0 is not architecture-specific — it reflects a vLLM-wide change in how activation memory is measured or allocated. All architecture constants in the planner should be re-calibrated against a single vLLM version (recommend v0.19.0) and re-validated whenever vLLM is updated.

| Metric | v0.16.0 | v0.17.0+ | Change |
|--------|:-------:|:--------:|:------:|
| Activation (Qwen3-14B) | 5.64 GiB | 2.23 GiB | −60% |
| KV cache (Qwen3-14B) | 41.94 GiB | ~45.3 GiB | +8% |
| Max concurrency (Qwen3-14B) | 33.55 | ~36.2 | +8% |

---

## Part 3: Parameters Not Yet Modeled

The following vLLM flags affect memory allocation but are not yet accepted as planner inputs.

### `--kv-cache-dtype fp8`

| Model | kv_cache_dtype | Actual KV (GiB) | KV GiB err | Actual tokens | Pred tokens | Token err |
|-------|:--------------:|:---------------:|:----------:|:-------------:|:-----------:|:---------:|
| Qwen2.5-7B-Instruct | auto | 58.53 | -4.21% | 1,096,000 | 1,049,789 | -4.22% |
| Qwen2.5-7B-Instruct | fp8 | 58.53 | -4.21% | 2,192,000 | 1,049,789 | -52.11% |
| Llama-3.1-8B-Instruct | auto | 42.80 | +31.06% | 175,296 | 459,509 | +162.13% |
| Llama-3.1-8B-Instruct | fp8 | 58.11 | -3.47% | 952,032 | 459,509 | -51.73% |

**KV pool size (GiB) is unaffected** — fp8 halves per-token storage, not the pool. **Token count is ~2× too low** because the planner uses the model's native dtype (BF16 = 2 bytes/element) instead of fp8 (1 byte/element). Fix: accept `kv_cache_dtype`; when `fp8`, use 1 byte/token.

### `--dtype` override

| dtype | Actual weight (GiB) | Weight err | Actual KV (GiB) | KV err |
|-------|:-------------------:|:----------:|:---------------:|:------:|
| bfloat16 | 14.99 | -0.22% | 58.11 | -3.47% |
| float16 | 14.99 | -0.22% | 58.11 | -3.47% |
| float32 | 29.98 | -50.11% | 42.80 | +31.06% |

`--dtype float32` doubles weight memory. The planner reads the HF config dtype (BF16) and has no visibility into the vLLM override → −50% weight error, +31% KV error. `--dtype float16` matches the HF config → near-zero error. Fix: accept `dtype` as input.

### Runtime `--quantization fp8`

| Model | Actual weight (GiB) | Weight err | Actual KV (GiB) | KV err |
|-------|:-------------------:|:----------:|:---------------:|:------:|
| Llama-3.1-8B-Instruct | 8.49 | +76.18% | 64.61 | -13.18% |

Runtime `--quantization fp8` compresses weights on-the-fly. vLLM logs the post-compression size (~half of BF16). The planner finds no `quantization_config` in the HF repo and predicts full BF16 weights → ~+76% weight error. Fix: accept `quantization fp8`; apply 1 byte/param.

### Recommendations

| Priority | Action | Expected impact |
|:--------:|--------|:---------------:|
| High | **Re-calibrate activation constants** from v0.19.0 measurements | Removes largest single error source (2–7× over-estimate) |
| High | **`kv_cache_dtype`** — when `fp8`, use 1 byte/token for KV | Fixes ~2× token/concurrency error for fp8-KV runs |
| Medium | **`dtype`** — when `float32`, double bytes-per-param | Fixes −50% weight error for float32 runs |
| Medium | **`quantization fp8` (runtime)** — apply 1 byte/param | Fixes +76% weight error for runtime-fp8 runs |
| Medium | **Re-measure non-torch constants for TP≥2 and PP≥2** | +1–2 GiB KV accuracy for multi-GPU |
| Medium | **Scale activation constant by 1/PP** | Fixes growing activation error at high PP |
| Medium | **`find_possible_tp`: require vocab_size divisibility** — valid TP must divide both `num_attention_heads` and `vocab_size` (vLLM shards the embedding/LM-head across TP ranks). Evidence: Qwen3-14B has 40 heads (tp=5 valid) but vocab_size=151936 (151936 % 5 ≠ 0 → rejected by vLLM). Fix: return divisors of `gcd(num_attention_heads, vocab_size)`. | Prevents planner from suggesting TP values that vLLM will reject |
| Low | **Use physical GPU memory** (79.19 GiB) instead of catalog 80 GiB | +0.77 GiB KV accuracy |

---

## Run Matrix — vLLM v0.19.0 / H100-80GB

**60 successful runs, 15 failed runs.**

Quantization abbreviations: `ct` = compressed-tensors, `gptq` = gptq_marlin, `fp8` = fp8 inline, `mxfp4` = mx-format fp4, `—` = none.

Vision/multi-modal models in this sweep: `moonshotai/Kimi-VL-A3B-Instruct` (vision-language MoE), `ibm-granite/granite-vision-3.3-2b` (vision-language), `google/gemma-3-{4b,12b,27b}-it` (vision-language), `meta-llama/Llama-4-Scout-17B-16E-Instruct` (vision+text MoE), `XiaomiMiMo/MiMo-VL-7B-SFT` (vision-language).

`Qwen/Qwen1.5-MoE-A2.7B` uses the Qwen2Moe architecture (14.3B total, 2.7B active). Its activation memory (2.47 GiB) is much lower than the generic MoE constant (8.0 GiB) used by the planner, similar to the pattern observed for Qwen3Moe, Mixtral, and Llama4.

### Successful Runs

| Model | TP | PP | DP | max_len | dtype | quant | kv_dtype | Weight err | Activation err | Non-torch err | KV cache err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| codellama/CodeLlama-7b-hf | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.1% | +523.4% | -40.0% | -5.1% |
| deepseek-ai/DeepSeek-V2-Lite-Chat | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.6% | +314.5% | -42.3% | -11.5% |
| google/gemma-2-27b-it | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.0% | +50.3% | -42.3% | -4.6% |
| google/gemma-2-2b-it | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.6% | +51.9% | -37.5% | -1.5% |
| google/gemma-2-9b-it | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.0% | +50.7% | -40.0% | -1.8% |
| google/gemma-3-12b-it | 1 | 1 | 1 | 8192 | bf16 | — | auto | -2.6% | +39.6% | -40.0% | -0.1% |
| google/gemma-3-27b-it | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.7% | +37.8% | -42.3% | -1.4% |
| google/gemma-3-4b-it | 1 | 1 | 1 | 8192 | bf16 | — | auto | -6.6% | +41.4% | -40.0% | -0.3% |
| google/gemma-7b | 1 | 1 | 1 | 8192 | bf16 | — | auto | +0.0% | +51.5% | -40.0% | -1.8% |
| ibm-granite/granite-3.1-2b-instruct | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.4% | +633.3% | -67.4% | -5.3% |
| ibm-granite/granite-3.1-8b-instruct | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.2% | +547.1% | -67.4% | -6.0% |
| ibm-granite/granite-3.3-8b-instruct | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.2% | +547.1% | -67.4% | -6.0% |
| ibm-granite/granite-vision-3.3-2b | 1 | 1 | 1 | 8192 | bf16 | — | auto | +0.0% | +216.5% | -40.0% | -1.2% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 8192 | bf16 | — | fp8 | -0.2% | +154.0% | -40.0% | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.2% | +154.0% | -40.0% | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 8192 | bf16 | fp8 | auto | +76.2% | +154.0% | -40.0% | -13.2% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 8192 | f16 | — | auto | -0.2% | +154.0% | -40.0% | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 8192 | f32 | — | auto | -50.1% | +117.2% | -40.0% | +31.1% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 2048 | bf16 | — | auto | -0.2% | +154.0% | -40.0% | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 4096 | bf16 | — | auto | -0.2% | +154.0% | -40.0% | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 2 | 1 | 8192 | bf16 | — | auto | -0.4% | +336.4% | +114.3% | -0.9% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 4 | 1 | 8192 | bf16 | — | auto | -12.2% | +357.1% | +114.3% | +1.6% |
| meta-llama/Llama-3.1-8B-Instruct | 2 | 1 | 1 | 8192 | bf16 | — | auto | -0.4% | +154.0% | -71.0% | +2.8% |
| meta-llama/Llama-3.1-8B-Instruct | 4 | 1 | 1 | 8192 | bf16 | — | auto | -0.8% | +154.0% | -71.8% | +4.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 16384 | bf16 | — | auto | -0.2% | +154.0% | -40.0% | -3.5% |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 1 | 32768 | bf16 | — | auto | -0.2% | +154.0% | -40.0% | -3.5% |
| microsoft/phi-2 | 1 | 1 | 1 | 2048 | f16 | — | auto | -0.2% | +596.2% | -37.5% | -5.5% |
| microsoft/phi-4 | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.3% | +261.8% | -40.0% | -6.6% |
| mistralai/Mistral-Small-3.1-24B-Instruct-2503 | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.1% | +23.2% | -40.0% | +1.5% |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | 2 | 1 | 1 | 8192 | bf16 | — | auto | -0.0% | +561.2% | -71.0% | -1.9% |
| moonshotai/Kimi-Dev-72B | 2 | 1 | 1 | 8192 | bf16 | — | auto | -0.2% | +144.5% | -71.3% | +61.8% |
| moonshotai/Kimi-Dev-72B | 4 | 1 | 1 | 8192 | bf16 | — | auto | -0.5% | +144.5% | -72.9% | +9.3% |
| moonshotai/Kimi-VL-A3B-Instruct | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.6% | +174.0% | -40.0% | -9.8% |
| moonshotai/Kimi-VL-A3B-Instruct | 2 | 1 | 1 | 8192 | bf16 | — | auto | -1.6% | +180.7% | -71.0% | +2.4% |
| openai/gpt-oss-20b | 2 | 1 | 1 | 8192 | bf16 | mxfp4 | auto | -8.6% | +178.7% | -71.0% | +2.8% |
| meta-llama/Llama-4-Scout-17B-16E-Instruct | 4 | 1 | 1 | 8192 | bf16 | — | auto | -4.8% | +150.8% | -72.4% | +36.2% |
| Qwen/Qwen2.5-72B-Instruct | 2 | 1 | 1 | 8192 | bf16 | — | auto | -0.1% | +144.5% | -71.3% | +60.8% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 8192 | bf16 | — | fp8 | -0.5% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.5% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 16384 | bf16 | — | auto | -0.5% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 32768 | bf16 | — | auto | -0.5% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 2048 | bf16 | — | auto | -0.5% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 1 | 1 | 1 | 4096 | bf16 | — | auto | -0.5% | +153.4% | -37.5% | -4.2% |
| Qwen/Qwen2.5-7B-Instruct | 2 | 1 | 1 | 8192 | bf16 | — | auto | -0.4% | +153.4% | -70.9% | +2.6% |
| Qwen/Qwen2.5-7B-Instruct | 4 | 1 | 1 | 8192 | bf16 | — | auto | -0.1% | +153.4% | -71.8% | +4.6% |
| Qwen/Qwen3-30B-A3B | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.0% | +198.5% | -44.4% | -28.8% |
| Qwen/Qwen1.5-MoE-A2.7B | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.0% | +223.9% | -40.0% | -10.1% |
| Qwen/Qwen3-8B | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.1% | +153.4% | -40.0% | -4.4% |
| RedHatAI/Llama-3.3-70B-Instruct-fp8-dynamic | 2 | 1 | 1 | 8192 | bf16 | ct | auto | -0.1% | +144.9% | -71.4% | +5.0% |
| RedHatAI/Llama-3.3-70B-Instruct-fp8-dynamic | 4 | 1 | 1 | 8192 | bf16 | ct | auto | -0.2% | +143.7% | -72.9% | +5.9% |
| redhatai/Llama-3.3-70B-Instruct-quantized.w8a8 | 2 | 1 | 1 | 8192 | bf16 | ct | auto | -0.1% | +144.9% | -71.6% | +5.0% |
| RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16 | 1 | 1 | 1 | 8192 | f16 | gptq | auto | -0.7% | +154.0% | -40.0% | -3.0% |
| RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 | 1 | 1 | 1 | 8192 | f16 | ct | auto | -0.4% | +154.0% | -40.0% | -3.1% |
| RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8 | 1 | 1 | 1 | 8192 | bf16 | ct | auto | -0.2% | +14.7% | -42.3% | +1.2% |
| RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8 | 2 | 1 | 1 | 8192 | bf16 | ct | auto | -0.8% | +23.2% | -71.0% | +5.3% |
| RedHatAI/Qwen2.5-7B-Instruct-fp8-dynamic | 1 | 1 | 1 | 8192 | bf16 | ct | auto | -0.4% | +153.4% | -37.5% | -3.9% |
| RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a8 | 1 | 1 | 1 | 8192 | bf16 | ct | auto | -0.4% | +153.4% | -40.0% | -3.9% |
| XiaomiMiMo/MiMo-VL-7B-SFT | 1 | 1 | 1 | 8192 | bf16 | — | auto | -0.8% | +120.9% | -40.0% | -3.5% |

### Failed Runs

| Model | TP | PP | DP | max_len | Notes |
|---|---|---|---|---|---|
| codellama/CodeLlama-34b-hf | 2 | 1 | 1 | 8192 | GPU contention at runtime |
| meta-llama/Llama-3.1-8B-Instruct | 1 | 1 | 2 | 8192 | DP=2 |
| microsoft/phi-2 | 1 | 1 | 1 | 8192 | max_model_len=8192 > max_position_embeddings=2048; fixed with max_model_len=2048 |
| moonshotai/Kimi-Dev-72B | 4 | 1 | 1 | 8192 | second attempt; tp=2 succeeded |
| openai/gpt-oss-20b | 1 | 1 | 1 | 8192 | sampler warmup OOM (~786 MiB needed, <552 MiB free) |
| openai/gpt-oss-20b | 2 | 1 | 1 | 8192 | sampler warmup OOM at gmu=0.95; succeeded at gmu=0.90 |
| Qwen/Qwen3-14B | 5 | 1 | 1 | 8192 | tp=5 invalid (vocab not divisible by 5) |

### Calibration Decisions

_Document constant changes here: old value → new value, evidence._

---

## Appendix: Measurement Methodology

### Log Extraction

Metrics are extracted from vLLM startup logs by [`accuracy/scripts/parse_logs.py`](scripts/parse_logs.py). All patterns match the first occurrence of the line, which comes from the rank-0 worker (TP0 or the sole GPU). Each regex below is taken directly from `parse_logs.py`.

#### Model / run config

```
Initializing a V1 LLM engine (v0.19.0) with config: model='...', dtype=...,
  max_seq_len=..., tensor_parallel_size=..., pipeline_parallel_size=...,
  data_parallel_size=..., quantization=..., kv_cache_dtype=...
```

Fields extracted: `model`, `dtype`, `max_model_len`, `tp`, `pp`, `dp`, `quantization`, `kv_cache_dtype`.

#### Weight memory per GPU (`weight_memory_gib`)

```
Model loading took 14.99 GiB memory and 14.429680 seconds
```

Regex: `Model loading took ([0-9.]+) GiB memory`

This is the per-GPU weight footprint as loaded by vLLM's model runner.

#### Activation, non-torch, and total non-KV memory

```
Memory profiling takes 17.96 seconds. Total non KV cache memory: 17.12GiB;
  torch peak memory increase: 1.89GiB; non-torch forward increase memory: 0.25GiB;
  weights memory: 14.99GiB.
```

Regex: `Memory profiling takes [0-9.]+ seconds\. Total non KV cache memory: ([0-9.]+)GiB; torch peak memory increase: ([0-9.]+)GiB; non-torch forward increase memory: ([0-9.]+)GiB; weights memory: ([0-9.]+)GiB\.`

Fields extracted:

| CSV column | Log field | Description |
|---|---|---|
| `total_non_kv_cache_gib` | `Total non KV cache memory` | Sum of all non-KV usage per GPU |
| `activation_memory_gib` | `torch peak memory increase` | Peak PyTorch memory from warmup/CUDA graph profiling |
| `non_torch_forward_gib` | `non-torch forward increase memory` | CUDA runtime + NCCL overhead |
| `weights_memory_gib` | `weights memory` | Cross-check of model loading line; should match `weight_memory_gib` |

#### KV cache memory per GPU (`kv_cache_memory_gib`)

```
Available KV cache memory: 58.11 GiB
```

Regex: `Available KV cache memory: ([0-9.]+) GiB`

#### KV cache token count (`kv_cache_tokens`)

```
GPU KV cache size: 476,016 tokens
```

Regex: `GPU KV cache size: ([\d,]+) tokens`

#### Max concurrency (`max_concurrency`)

```
Maximum concurrency for 8,192 tokens per request: 58.11x
```

Regex: `Maximum concurrency for ([\d,]+) tokens per request: ([0-9.]+)x`

This is `kv_cache_tokens / max_model_len`.

#### CUDA graph memory (`cuda_graph_actual_gib`)

```
CUDA graph pool memory: 0.91 GiB (actual), 0.84 GiB (estimated), difference: 0.07 GiB (7.7%).
```

Regex: `CUDA graph pool memory: ([0-9.]+) GiB \(actual\), ([0-9.]+) GiB \(estimated\)`

#### KV cache blocks (`kv_cache_blocks`)

```
Engine 000: vllm cache_config_info with initialization after num_gpu_blocks is: 29751
```

Regex: `num_gpu_blocks is: (\d+)`

---

### Capacity Planner Functions Evaluated

Predictions are generated by [`accuracy/scripts/predict_capacity.py`](scripts/predict_capacity.py), which imports from [`src/planner/capacity_planner.py`](../../src/planner/capacity_planner.py).

| Metric | Log column | Planner function | File:line |
|---|---|---|---|
| Weight memory per GPU | `weight_memory_gib` | `per_gpu_model_memory_required()` → `model_memory_req()` | [`capacity_planner.py:832`](../../src/planner/capacity_planner.py#L832) |
| Activation memory | `activation_memory_gib` | `estimate_vllm_activation_memory()` | [`capacity_planner.py:380`](../../src/planner/capacity_planner.py#L380) |
| Non-torch memory | `non_torch_forward_gib` | `estimate_vllm_non_torch_memory()` | [`capacity_planner.py:351`](../../src/planner/capacity_planner.py#L351) |
| KV cache memory per GPU | `kv_cache_memory_gib` | `allocatable_kv_cache_memory()` ÷ (tp×pp×dp) | [`capacity_planner.py:855`](../../src/planner/capacity_planner.py#L855) |
| KV cache tokens | `kv_cache_tokens` | KV bytes ÷ `KVCacheDetail.per_token_memory_bytes` per GPU | [`capacity_planner.py:81`](../../src/planner/capacity_planner.py#L81) |
| Max concurrency | `max_concurrency` | KV bytes ÷ (per-token bytes × max_model_len) | [`capacity_planner.py:81`](../../src/planner/capacity_planner.py#L81) |
| CUDA graph memory | `cuda_graph_actual_gib` | `estimate_vllm_cuda_graph_memory()` → returns 0.0 | [`capacity_planner.py:366`](../../src/planner/capacity_planner.py#L366) |

**Weight memory** is computed by parsing the model's safetensors index to count parameter bytes per dtype. For quantized models, the quant method is detected from `quantization_config` in the HF config and the appropriate bytes-per-param are applied. See `model_memory_req()` at [`capacity_planner.py:553`](../../src/planner/capacity_planner.py#L553).

**Activation memory** uses a tiered lookup: first checks `VALIDATED_ACTIVATION_PROFILES` (a dict keyed by `architectures[0]`), then falls back to `ACTIVATION_MEMORY_BASE_MOE_GIB` (8.0 GiB) for MoE models, `ACTIVATION_MEMORY_BASE_MULTIMODAL_GIB` (2.5 GiB) for multimodal, and `ACTIVATION_MEMORY_BASE_DENSE_GIB` (5.5 GiB) for all others. See [`capacity_planner.py:44`](../../src/planner/capacity_planner.py#L44) for the constants and [`capacity_planner.py:424`](../../src/planner/capacity_planner.py#L424) for the lookup logic.

**KV cache memory** uses the formula:
```
available = gpu_memory_gib × gpu_util × (tp × pp × dp)
           − model_weights × dp
           − activation × dp
           − cuda_graph (0.0)
           − non_torch
per_gpu_kv = available / (tp × pp × dp)
```
This uses the catalog GPU memory (80 GiB), not the physical VRAM (79.19 GiB); the gap accounts for most of the baseline KV under-estimate.

**Per-token KV bytes** are derived from `KVCacheDetail` at [`capacity_planner.py:81`](../../src/planner/capacity_planner.py#L81):
```
per_token_bytes = 2 × num_kv_heads × head_dim × num_layers × dtype_bytes
```
For MLA (DeepSeek models), the formula uses `kv_lora_rank` and `qk_rope_head_dim` instead. TP sharding: each GPU holds `per_token_bytes / (tp × pp)` per token.
