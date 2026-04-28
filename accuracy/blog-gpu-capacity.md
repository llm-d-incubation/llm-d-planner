# 58 Experiments Later: What We Learned About LLM Memory Prediction

*GPU memory planning for LLM deployments is still mostly guesswork. Here's what we learned from measuring it empirically across 35 architectures.*

---

You're standing up a benchmark suite and need to know how many GPUs each model configuration requires before sizing the cluster. Or you're launching a serving application and want to plan capacity without over-provisioning by 3x. Or you're a researcher asking whether two H100s will be enough for a 70B model, or whether you need four.

In all of these cases, the question is the same: **how much GPU memory will this actually need?**

Most teams answer it by copying what someone else deployed, or by spinning up the pod, watching it OOM, and doubling the resources. This works, but it gets harder as models grow larger and serving configurations more complex. Tensor parallelism, pipeline parallelism, quantization, and long-context windows all change the memory footprint in non-obvious ways.

[llm-d-planner](https://github.com/llm-d-incubation/llm-d-planner) is an open-source library that guides LLM deployments from concept to production. One of its core submodules is a capacity planner built to answer this question before you deploy. To make sure it wasn't just replacing guesswork with a false sense of precision, we ran 58 experiments on H100 GPUs to validate its predictions against reality. Here's what we found, and why we're asking the community to help make it even better.

---

## What llm-d-planner Does

llm-d-planner guides LLM deployments from concept to production: conversational requirements gathering, SLO-driven model and GPU recommendations, what-if analysis, one-click Kubernetes config generation, and monitoring. The capacity planner is a pip-installable subcomponent that focuses on one question: how much GPU memory will this deployment actually need.

It breaks memory into four components: weights, KV cache, activation memory, and non-torch overhead (CUDA runtime and NCCL buffers for multi-GPU). Each scales differently with tensor parallelism, context length, and quantization, so knowing which component is driving your footprint tells you what to actually change. For each component, the planner anchors to a source of truth wherever one exists: `config.json` and safetensor file headers for weights, vLLM's allocation strategy for KV cache, and empirically measured constants for things that can't be derived analytically, like activation memory. The experiment in this post is how those constants are kept honest.

---

## The Experiment: Trusting but Verifying

Claiming a tool is accurate is easy. Measuring it is harder. We launched vLLM servers across 58 configurations on H100-80GB GPUs, captured the full startup logs for each, and parsed the actual memory measurements reported by vLLM at initialization. We then compared those measurements against llm-d-planner's predictions for every configuration. The sweep covered:

- **35 model architectures**: Llama, Qwen, Gemma, Granite, Mistral, DeepSeek, Phi, Mixtral, and multimodal models including LLaVA, Kimi-VL, and MiMo
- **Tensor parallelism** (TP 1, 2, 4) and **pipeline parallelism** (PP 1, 2, 4)
- **Context lengths** from 2,048 to 32,768 tokens
- **Dtype and quantization variants**: bfloat16, float16; compressed-tensors, GPTQ
- **vLLM version sensitivity**: Qwen3-14B across v0.15.0 through v0.19.0 to track how memory behavior changes across releases

For each run, we compared predicted values against the four measured memory components independently. The [raw results and all run JSON files](https://github.com/llm-d-incubation/llm-d-planner/tree/main/accuracy/results) are committed to the repository, and the analysis is fully reproducible locally without cluster access.

---

## What We Found

### The headline: accurate where it counts most

**Weight memory: 0.89% mean absolute error** across 54 of the 58 runs. (The remaining 4 used parameters the planner doesn't yet model, float32 dtype and runtime fp8 quantization, and are discussed below.) This is the single largest memory component; for a model like Llama-3.1-8B at TP=1, weights consume about 15 GiB of the 79 GiB available. It's also the hardest to get right across a diverse model set.

Weight prediction is harder than it looks: dense, MoE, multi-head latent attention, and vision-language models all organize parameters differently, quantization changes the bytes-per-parameter, and TP sharding depends on how dimensions divide across ranks. The formula handles all of this by reading `config.json` for architecture parameters and safetensor headers for exact tensor shapes, giving precise counts without downloading the full model and making it generalizable to any model on HuggingFace beyond the 35 we explicitly tested. Across dense, MoE, multimodal, and quantized architectures, it held to under 1% error.

**KV cache memory: 0.34% mean error** across all runs. This is the component that matters most for capacity planning, as it determines your maximum concurrent token budget. For Llama-3.1-8B at TP=1 with 8K context, that's roughly 58 GiB of KV pool, and we're within half a GiB across every context length we tested.

One insight worth pausing on: the KV pool size is *independent* of `max_model_len`. vLLM sizes the pool from whatever memory remains after weights and activation are allocated, then figures out how many tokens fit given the per-token KV size for that architecture. This means setting a longer context window doesn't shrink your KV pool; it just means each request can use a larger share of it, leaving less headroom for concurrent requests at the maximum context length. Tools that pre-allocate based on `max_model_len` will over-estimate memory for long-context configs and leave capacity on the table.

These two components together typically account for 90%+ of total GPU memory consumption. Getting them right is what makes the planner useful in practice.

### The honest part: smaller components, real errors

**Activation memory** showed a mean error of +195%. That sounds alarming, so let's ground it in absolute numbers. For Llama-3.1-8B at TP=1, our formula predicted 4.80 GiB; vLLM v0.19.0 actually used 1.89 GiB, an over-estimate of about 2.9 GiB. On a GPU with 79 GiB of VRAM where weights alone consume 15 GiB and the KV pool takes 58 GiB, a 2.9 GiB error in a smaller component is meaningful but bounded.

The root cause is more interesting than the magnitude: **vLLM v0.17.0 quietly reduced activation memory by ~60%, and we didn't notice.**

Our version sensitivity study tells the story clearly:

| vLLM version | Activation (Qwen3-14B) |
|:---:|:---:|
| v0.15.0 | 5.64 GiB |
| v0.16.0 | 5.64 GiB |
| **v0.17.0** | **2.23 GiB** |
| v0.18.0 | 2.23 GiB |
| v0.19.0 | ~2.21 GiB |

The planner's Qwen3 activation constant was 5.60 GiB, a near-exact match for v0.16.0. Our constants had been calibrated against an older vLLM release and were never updated as vLLM evolved. The 60% reduction at v0.17.0 freed memory that vLLM reallocated to the KV cache, actually *improving* serving capacity, but our planner didn't know about it.

This kind of silent drift is precisely why empirical validation matters. We didn't catch it until we ran the experiments, and the fix was straightforward once we knew where to look: re-calibrate every architecture constant against v0.19.0 measurements. That's now done, and the updated constants are in the library.

The planner currently tracks the behavior of the latest supported vLLM release (v0.19.0). It is not version-aware in the sense that it won't automatically adjust for older releases. When vLLM changes memory behavior in a future release, re-running the accuracy sweep and submitting a PR with updated constants is how the library stays current—which is exactly the kind of contribution the community campaign is designed to support.

**Non-torch overhead** (CUDA runtime + NCCL buffers) was under-estimated by 44% on average. At TP=1, this is a small absolute amount (~0.25 GiB actual vs 0.15 GiB predicted). At TP>=2, NCCL all-reduce buffers push actual overhead to ~2.1 GiB per GPU versus our constant of 0.60 GiB, a more meaningful gap. Updated multi-GPU constants are also in.

There are a few configurations the experiment didn't cover that the planner doesn't yet model: fp8 KV cache dtype (halves per-token storage, roughly doubling token capacity), float32 dtype overrides (doubles weight memory), runtime fp8 quantization, and data parallelism. For entirely unknown precision types, the planner raises an error. For these specific gaps, the planner will produce an estimate using the base model configuration without accounting for the override—meaning results may be off for that component without an explicit warning. These are real gaps for anyone running quantized production models today, and they're actively being worked on — contributions are welcome if you need one of these sooner. The sweep also turned up a subtle correctness bug in `find_possible_tp`: it wasn't verifying that TP values divide `vocab_size`, which can cause vLLM to reject a configuration the planner suggests as valid. That's fixed.

---

## Join the Community

We covered 35 architectures. The LLM landscape releases more every week, and vLLM will keep evolving. Accuracy at a point in time isn't enough; what matters is having a community that keeps the constants current as things change.

**If your model isn't covered, or a new architecture ships with memory optimizations** (a new attention variant, a custom KV cache layout, or a novel quantization scheme), llm-d-planner should be where those updated constants land first. The sweep runner in `accuracy/` is fully documented and self-contained; run it against your own cluster, submit the results as a PR, and everyone who installs the library gets the improvement.

**Get started:**

- [GitHub: llm-d-incubation/llm-d-planner](https://github.com/llm-d-incubation/llm-d-planner)
- [Accuracy campaign results and methodology](https://github.com/llm-d-incubation/llm-d-planner/tree/main/accuracy)
- [Run the sweep on your own cluster](https://github.com/llm-d-incubation/llm-d-planner/blob/main/accuracy/README.md)
- Open an issue or PR; contributions welcome

No one should have to guess how many GPUs they need.
