# 61 Experiments Later: What We Learned About LLM Memory Estimation

*GPU memory estimation for LLM deployments is still mostly guesswork. llm-d-planner is designed to solve it. Here's what validating its estimates against 61 experiments across 35 architectures taught us, and why empirical grounding is what separates a useful tool from a confident guess.*

---

You're planning a benchmark suite and need to know how many GPUs each model requires before sizing the cluster. You're launching a serving application and want to avoid over-provisioning by 3x. You're a researcher asking whether two H100s will be enough for a 70B model.

The question is the same: **how much GPU memory will this actually need?**

Memory is the first gate. Either the model fits or it doesn't, and the only way to find out without a memory planning tool is to spin up a vLLM server and see if it OOMs. Tensor parallelism, pipeline parallelism, quantization, and long-context windows all change the footprint in non-obvious ways, which makes trial-and-error expensive. This post is about memory estimation specifically: not throughput, latency, or any other performance metric. Before you can answer any question about performance, you need to answer this one: does the model fit?

[llm-d-planner](https://github.com/llm-d-incubation/llm-d-planner) includes a pip-installable capacity planner that answers this question before you touch a cluster, using only model config files and safetensor headers. To verify it wasn't replacing guesswork with false precision, we ran 61 experiments on H100 GPUs. Here's what we found.

---

## How the Planner Works

Memory breaks into four components: weights, KV cache, activation memory, and non-torch overhead. The theoretical approach is standard: weight memory from parameter counts and dtype, KV cache from attention head dimensions and context length. What makes this different is that we measured actual vLLM behavior across 61 configurations to validate those formulas and catch where theory diverges from reality. No GPU is required at planning time; all constants are grounded in prior empirical measurements on real hardware.

**Current limitations:** Constants are calibrated on H100-80GB; other GPU types may differ. fp8 quantization modes and float32 dtype overrides are not yet modeled — treat estimates for those configurations as an upper bound.

---

## The Experiment

We launched vLLM servers across 61 configurations on H100-80GB GPUs, captured startup logs, and compared measured memory against the planner's estimates. vLLM reports per-component memory usage at initialization, which is what made per-component error measurement possible. The sweep covered:

- **35 model architectures**: Llama, Qwen, Gemma, Granite, Mistral, DeepSeek, Phi, Mixtral, multimodal models (LLaVA, Kimi-VL, MiMo)
- **Tensor parallelism** (TP 1, 2, 4) and **pipeline parallelism** (PP 1, 2, 3, 4)
- **Context lengths** from 2,048 to 32,768 tokens
- **Dtype and quantization**: bfloat16, float16; compressed-tensors, GPTQ
- **vLLM version sensitivity**: Qwen3-14B across v0.15.0-v0.19.0

[Raw logs and run JSON files](https://drive.google.com/drive/folders/1a0y2gdhcpKcFxm4RsqXUKWW40Gpd2Kx5) are published; the analysis is reproducible locally without cluster access.

---

## What We Found

The two components that dominate GPU memory, weights and KV cache, account for over 90% of total usage. Weight estimation came in under 1% mean error across all architectures. KV cache error was low for dense models (typically under 5%) but higher for sparse MoE architectures. Here's how that breaks down by model family:

| Family | Type | Weight error | KV error |
|---|---|:---:|:---:|
| Llama | Dense | <1% | 3–5% |
| Qwen | Dense | <1% | ~4% |
| Gemma 2 | Dense | <1% | 1–5% |
| Gemma 3 | Dense | <7%* | <2% |
| Granite | Dense | <1% | 5–6% |
| Phi | Dense | <1% | 5–7% |
| Mistral | Dense | <1% | <2% |
| Mixtral | MoE | <1% | ~2% |
| Qwen MoE | MoE | <1% | 10–29% |
| DeepSeek MLA | MoE | <1% | ~12% |
| LLaVA, Kimi-VL, MiMo-VL | Multimodal | <1% | 1–10% |

\* One Gemma 3 variant (4B) had 6.65% weight error; all others under 1%.

Weight estimation holds consistently across all architecture types. KV cache error is higher for sparse MoE models — Qwen3-30B-A3B, where only ~10% of parameters are active per token, drives the 29% upper bound. All errors are under-estimates: the planner predicts slightly less memory than vLLM actually uses, which is the safe direction for capacity planning.

**Where we got it wrong.** Activation memory showed +212% mean error, and the reason is a story worth telling. Between v0.16.0 and v0.17.0, vLLM silently reduced activation memory overhead by ~60% (from 5.64 GiB down to 2.23 GiB for Qwen3-14B), and we didn't notice until we ran these experiments. Our constants reflected the older behavior. In absolute terms the impact was ~2.9 GiB on a 79 GiB GPU: bounded, but real, and the kind of drift that's invisible without empirical validation. The right fix isn't to chase vLLM releases with updated constants; it's to derive activation memory from first principles so the estimate doesn't depend on framework internals at all.

For the complete per-model and per-configuration breakdown, see the [full accuracy report](https://github.com/llm-d-incubation/llm-d-planner/blob/main/accuracy/accuracy_report.md). Beyond the numbers, running these experiments gave us a clearer picture of how LLMs actually behave at runtime, and those findings will inform the next round of improvements to the planner's accuracy.

---

## Plan Before You Provision

The goal isn't a perfect number. It's a good enough answer at day 0: before you've committed to hardware, before you've designed your deployment topology, before you've found out that the workload you planned for doesn't fit on the cluster you ordered. Memory estimation is what gives you that answer early: whether the model fits, in what configuration, and what concurrency your hardware can actually support under your expected workload.

- [GitHub: llm-d-incubation/llm-d-planner](https://github.com/llm-d-incubation/llm-d-planner)
- [Full accuracy report with per-model tables](https://github.com/llm-d-incubation/llm-d-planner/blob/main/accuracy/accuracy_report.md)
- [Run the sweep on your own cluster](https://github.com/llm-d-incubation/llm-d-planner/blob/main/accuracy/README.md)

Before these experiments, we had formulas. Now we have evidence and a clearer path to making the planner accurate enough to trust before you touch a cluster.
