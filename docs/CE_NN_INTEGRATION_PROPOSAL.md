# Proposal: Unifying config_explorer and NeuralNav into a Single Upstream Project

## The Problem

Deploying LLMs in production is a multi-step decision process. Users start with a general use case -- a chatbot, a code assistant, a document summarization service -- and need to work through a series of decisions:

1. **Determine requirements**: What latency, throughput, and quality targets does this use case demand? What traffic patterns should the deployment handle?
2. **Evaluate options**: Which combinations of models and hardware can meet those requirements? What are the trade-offs in cost, accuracy, latency, and deployment complexity across viable configurations?
3. **Make a decision and deploy**: Choose a configuration, generate deployment manifests, apply them to a cluster, and verify the service meets expectations.

**NeuralNav** guides users through this entire process: describe a use case in natural language, and NeuralNav extracts the requirements, evaluates model+GPU configurations against real benchmark data, ranks options across multiple dimensions (cost, quality, latency, complexity), and deploys the chosen configuration to Kubernetes. The limitation is **coverage** -- NeuralNav can only evaluate configurations that have been empirically benchmarked, and currently only supports aggregated vLLM deployments.

**config_explorer** (in llm-d-benchmark) expands what can be evaluated. Its GPU memory analysis determines hardware feasibility for virtually any model, its roofline model produces synthetic performance estimates for unbenchmarked configurations, and its benchmark explorer analyzes llm-d deployments including P/D disaggregation. It provides the breadth that NeuralNav's benchmark-dependent approach lacks, but without the guided workflow, multi-criteria ranking, or deployment automation.

This proposal argues that combining these projects into a single upstream tool in the llm-d organization would produce a system that is meaningfully better than either project alone.

## The Case for Integration

### 1. NeuralNav's guided workflow + config_explorer's expanded reach

NeuralNav already provides a complete end-to-end workflow: a user describes their use case in natural language, NeuralNav extracts the important details (model requirements, traffic profile, SLO targets), produces ranked recommendations backed by real benchmark data, generates deployment manifests, and deploys to Kubernetes. That guided approach is the core value -- users don't need to know what GPU memory their model requires or what tensor parallelism strategy to use.

The limitation is **coverage**. NeuralNav can only recommend configurations that exist in its benchmark database -- specific combinations of models, GPUs, GPU counts, and traffic profiles that have been tested with GuideLLM. If a promising model hasn't been benchmarked on a particular GPU, it doesn't appear as an option. If the user's workload has different token length distributions than the four GuideLLM profiles, there's no data to draw from.

config_explorer removes that ceiling. Its roofline model can estimate performance for virtually any model on any GPU, and its capacity planner can verify memory feasibility before anything is benchmarked. In a unified system:

- NeuralNav's intent extraction produces the structured specification (use case, token lengths, SLO targets) that would otherwise require the user to manually figure out and plug into config_explorer's CLI
- config_explorer's engines expand the recommendation space beyond what's been empirically tested
- The system presents **validated options** from real benchmarks alongside **promising alternatives** that include potentially better models, different GPU configurations, or different data patterns that haven't been benchmarked yet
- Users get a guided experience with broader coverage, rather than having to choose between a guided tool with limited options or a manual tool with unlimited options

### 2. The capabilities are complementary, not competitive

The two projects have remarkably little overlap in their core logic:

| Capability | config_explorer | NeuralNav | Overlap? |
| --- | --- | --- | --- |
| GPU memory estimation (model weights, KV cache, activation) | Yes | No | None |
| Attention-type-aware analysis (MHA/GQA/MQA/MLA) | Yes | No | None |
| Synthetic performance estimation (roofline model) | Yes | No | None |
| P/D disaggregation data model | Yes | No | None |
| llm-d scheduler parameter analysis | Yes | No | None |
| Benchmark sweep visualization (Pareto fronts) | Yes | No | None |
| Natural language intent extraction | No | Yes | None |
| Use case to SLO mapping (9 use cases, 4 traffic profiles) | No | Yes | None |
| Model quality scoring | No | Yes | None |
| Multi-criteria ranking (4 dimensions, 5 ranked views) | No | Yes | None |
| Replica and scaling calculations | No | Yes | None |
| YAML generation (KServe/vLLM/HPA/ServiceMonitor) | No | Yes | None |
| Kubernetes deployment and lifecycle management | No | Yes | None |
| SLO compliance filtering | Yes | Yes | Shared concept, different implementation |
| GPU cost lookups | Yes | Yes | Shared concept, different data sources |

The overlap is limited to two shared concepts (SLO filtering and GPU pricing), both of which would benefit from a single source of truth rather than competing implementations.

### 3. NeuralNav's multi-factor decision matrix becomes the unified ranking layer

config_explorer can tell you which GPU runs a model and what performance to expect, but it has no concept of whether the model is any good at the task you need it for. It ranks GPUs, not deployment decisions.

NeuralNav factors **model accuracy and quality** benchmarks into its recommendation scoring. A model that's fast and cheap but produces low-quality output for your use case is ranked accordingly -- it won't be recommended as "best" just because it fits on a single GPU.

This matters for a unified system because config_explorer's roofline model will surface many more (model, GPU) configurations than currently available. Without quality-aware ranking, users would drown in options with no way to distinguish a capable model from a fast-but-mediocre one. NeuralNav's scoring engine ensures that expanded coverage doesn't come at the cost of recommendation quality.

The decision matrix is also designed to grow. Beyond accuracy and latency, NeuralNav plans to incorporate **model safety scoring** (toxicity, bias, alignment characteristics) and potentially other factors like license compliance and ecosystem maturity. These dimensions apply equally to benchmarked and synthetically estimated configurations, making the unified ranking more valuable as the decision matrix expands.

### 4. Synthetic + empirical performance data is more powerful than either alone

NeuralNav's biggest limitation today is that it can only recommend (model, GPU) configurations that have been empirically benchmarked. If a model hasn't been tested on a specific GPU with GuideLLM, it simply doesn't appear as an option. This creates a cold-start problem: every new model or GPU requires a benchmark run before it can be recommended.

config_explorer's roofline model can produce synthetic estimates for *any* model on *any* GPU, instantly. These estimates are less accurate than real benchmarks but far better than no data at all.

A unified system can present recommendations with **tiered confidence**:

- **High confidence**: Configurations backed by real GuideLLM benchmark data
- **Medium confidence**: Configurations with synthetic roofline estimates, validated by memory feasibility analysis
- **Exploratory**: Configurations from capacity planning alone (feasible but uncharacterized performance)

This gives users a complete picture: proven options alongside promising alternatives worth benchmarking.

### 5. The benchmark feedback loop becomes automatic

Today, benchmark data flows in one direction: llm-d-benchmark produces it, and NeuralNav (or similar tools) consume it. In a unified project:

1. **Benchmarks feed recommendations**: Real benchmark data from GuideLLM runs populates the recommendation database
2. **Recommendations drive benchmarks**: When the system identifies high-value (model, GPU) combos that lack empirical data (only synthetic estimates), it can prioritize those for the next benchmark run
3. **Deployments validate both**: Actual deployment outcomes feed back to calibrate both synthetic estimates and benchmark-based predictions

This closed loop improves recommendation quality over time without manual coordination between separate projects.

### 6. NeuralNav needs P/D disaggregation support

NeuralNav currently only supports aggregated vLLM deployments. Adding llm-d P/D disaggregation support is a stated goal but would require significant schema work:

- Extending `GPUConfig` to represent separate prefill/decode configurations
- Extending `BenchmarkData` to store disaggregated metrics
- Extending the recommendation engine to compare aggregated vs disaggregated topologies
- Building UI for disaggregated deployment visualization

config_explorer's `explorer.py` already has all of this modeled in its 84-column DataFrame schema (`Is_PD`, `P_TP`, `D_TP`, `P_Replicas`, `D_Replicas`, scheduler weights). In a unified project, this becomes the foundation rather than something NeuralNav has to reinvent.

## Conclusion

The strongest argument for integration is not that it reduces engineering effort -- it doesn't in the short term. The argument is that **the unified product is meaningfully better than either project alone**. A tool that can analyze GPU memory feasibility, estimate performance synthetically, rank configurations against real benchmarks with quality and safety awareness, generate deployment manifests, and deploy to Kubernetes -- all in one workflow -- is qualitatively different from two tools that each cover part of that path. As the decision matrix grows to include safety, license compliance, and other factors, the value of a single ranking engine that applies these dimensions to both benchmarked and synthetically estimated configurations compounds.
