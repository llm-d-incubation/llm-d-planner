# Accuracy Score Mapping Summary

## Overview
This document summarizes how accuracy scores were mapped from `opensource_all_benchmarks.csv` to `redhat_models_benchmarks.csv` for the NeuralNav POC demo.

**Date:** December 2024  
**Files Updated:** `redhat_models_benchmarks.csv`

---

## Penalty Scale Reference (Research-Based 2024)

Based on recent quantization research:
- **GPTQ (2023)**: W4A16 shows <2% accuracy loss with proper calibration
- **AWQ (2024)**: 4-bit shows <1.5% loss on most benchmarks
- **FP8 (2024)**: <0.5% loss in most cases (negligible)
- **Llama.cpp**: W8A8 shows ~0.5-1% loss

| Quantization | Penalty | Typical Accuracy Loss | Research Reference |
|--------------|---------|----------------------|-------------------|
| Full (FP16/32) | ×1.00 | 0% | Reference baseline |
| FP8-dynamic | ×0.995 | ~0.5% | NVIDIA FP8 (2024) |
| W8A8 (8-bit) | ×0.99 | ~1% | INT8 Quantization |
| W4A16 (4-bit) | ×0.98 | ~2% | GPTQ/AWQ (2024) |
| Version diff (e.g., v2.5→v3) | ×0.92 | ~8% | Model generation diff |

---

## Key Model Mappings

### Kimi K2 (Top Model)

| Model | Base | Penalty | Result |
|-------|------|---------|--------|
| RedHatAI/Kimi-K2-Instruct-quantized.w4a16 | Kimi K2 Thinking | ×0.98 | **83.10% MMLU-Pro, 91.14% τ²-Bench** |

**Key Scores:**
- τ²-Bench: 91.14% (Agentic workflows - HIGHEST)
- LiveCodeBench: 86.24% (Code generation)
- MMLU-Pro: 83.10% (General knowledge)
- GPQA: 82.81% (Scientific reasoning)

### DeepSeek R1

| Model | Base | Penalty | Result |
|-------|------|---------|--------|
| RedHatAI/DeepSeek-R1-0528-quantized.w4a16 | DeepSeek R1 | ×0.98 | **94.67% MMLU-Pro** |

### Other Mappings

| Our Model | Source Model | Penalty | Reason |
|-----------|--------------|---------|--------|
| Qwen/Qwen2.5-7B-Instruct | Qwen3 8B | ×0.92 | Version 2.5 < 3 |
| RedHatAI/Qwen2.5-7B-*-FP8 | Qwen3 8B | ×0.915 | v2.5 + FP8 |
| RedHatAI/Qwen2.5-7B-*-w4a16 | Qwen3 8B | ×0.90 | v2.5 + W4A16 |
| ibm-granite/granite-3.1-8b-* | Granite 3.3 8B | ×1.00 | Same generation |

---

## Use Case Scores (Final - December 2024)

| Use Case | Kimi K2 Score | Top Benchmark |
|----------|---------------|---------------|
| **Chatbot Conversational** | **86.7%** | τ²-Bench + MMLU-Pro |
| **Code Completion** | **87.3%** | LiveCodeBench + τ²-Bench |
| **Code Generation** | **87.5%** | LiveCodeBench + τ²-Bench |
| **Translation** | **86.7%** | τ²-Bench + MMLU-Pro |
| **Content Creation** | **86.7%** | τ²-Bench + MMLU-Pro |
| **Summarization** | **86.7%** | τ²-Bench + MMLU-Pro |
| **Document RAG** | **87.0%** | τ²-Bench (Agentic) |
| **Long Doc Summary** | **87.1%** | τ²-Bench + MMLU-Pro |
| **Research/Legal** | **87.5%** | τ²-Bench (Research) |

---

## Results Summary

| Metric | Value |
|--------|-------|
| Full coverage | **42 (84%)** |
| Partial coverage | **7 (14%)** |
| No coverage | **1 (2%)** |
| Max accuracy (MMLU-Pro) | **94.67%** (DeepSeek R1) |
| Max τ²-Bench | **91.14%** (Kimi K2) |

### Top 5 Accuracy Models

1. **DeepSeek-R1-0528-quantized.w4a16** - 94.67% MMLU-Pro
2. **Kimi-K2-Instruct-quantized.w4a16** - 83.10% MMLU-Pro, **91.14% τ²-Bench**
3. **Llama-4-Maverick-17B-128E-Instruct** - 88.90% MMLU-Pro
4. **Mistral-Small-3.1-24B-Instruct-2503** - 88.30% MMLU-Pro
5. **GPT-OSS-120B** - 80.80% MMLU-Pro

---

## Files

- `accuracy_mapping_summary.csv` - Full mapping table (downloadable)
- `redhat_models_benchmarks.csv` - Updated model scores (50 models)
- `opensource_all_benchmarks.csv` - Source AA benchmark data (206 models)
- `optimized_weights.json` - Use case weighting configuration
