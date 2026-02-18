# Comparative Technical Evaluation: config_explorer vs NeuralNav

## Executive Summary

**config_explorer** (llm-d-benchmark) and **NeuralNav** solve related but architecturally distinct problems in the LLM deployment space. config_explorer is a bottom-up infrastructure analysis tool built specifically for the llm-d stack, with prefill/decode (P/D) disaggregation as a first-class concern. NeuralNav is a top-down deployment guidance system that translates business requirements into ranked, SLO-compliant model+GPU configurations using empirical benchmark data -- currently focused on aggregated vLLM deployments, with llm-d support as a stated goal.

Their overlap is real but narrow -- concentrated in the "given a model and workload, which GPU configuration works?" question. The most promising integration path is **offline synthetic benchmark generation**: using config_explorer to pre-compute estimated performance data in NeuralNav's benchmark format, expanding NeuralNav's recommendation space with clearly labeled confidence levels. A full codebase merge would require aligning on shared objectives and operating as a single open-source team.

## 1. Functional Comparison

### What config_explorer solves

config_explorer answers: **"Can this model physically run on this GPU, and what performance can I expect -- especially on llm-d with P/D disaggregation?"**

Three capabilities:

1. **Capacity Planning** (`capacity_planner.py`): Calculates GPU memory breakdown -- model weights, KV cache (MHA/GQA/MQA/MLA-aware), activation memory, CUDA overhead -- to determine whether a model fits on a given GPU with a given parallelism strategy (TP/PP/DP/EP). Reports max concurrent requests. Empirically validated against H100 GPUs. Generic to vLLM, not llm-d-specific.
2. **GPU Recommendation** (`recommender/recommender.py`): Uses BentoML's `llm-optimizer` roofline model to synthetically estimate throughput, TTFT, ITL, and E2E latency across GPU types. Ranks by best throughput, lowest latency, or lowest cost. Operates on aggregated vLLM only -- does **not** model P/D disaggregation.
3. **Benchmark Exploration** (`explorer.py`): Loads llm-d benchmark report files (YAML, schema v0.1/v0.2) into Pandas DataFrames with 84 columns. Supports SLO filtering, Pareto-front analysis, and scenario-based visualization. This module is **deeply llm-d-specific** -- its data model has first-class columns for P/D disaggregation (`P_TP`, `D_TP`, `P_Replicas`, `D_Replicas`, `Is_PD`) and llm-d inference scheduler parameters (`KV_Cache_Scorer_Weight`, `Queue_Scorer_Weight`, `Prefix_Cache_Scorer_*`).

### What NeuralNav solves

NeuralNav answers: **"Given my business needs, what model+GPU deployment should I use, and how do I get it running?"**

Four-stage workflow:

1. **Intent Extraction**: LLM-powered NLU converts natural language ("chatbot for 1000 users") into structured intent (use case, user count, priorities).
2. **Specification Generation**: Maps intent to GuideLLM traffic profile + SLO targets (9 use cases, 4 GuideLLM traffic profiles).
3. **Recommendation Engine**: Queries PostgreSQL for all (model, GPU, TP) combos meeting SLO targets from real benchmark data (produced by running GuideLLM against real models on real GPUs). Scores on 4 dimensions (accuracy, price, latency, complexity), generates 5 ranked views.
4. **Configuration & Deployment**: Generates KServe/vLLM YAML via Jinja2 templates, deploys to Kubernetes, monitors health.

**Important context**: While BLIS simulator benchmarks are currently checked into the repo, NeuralNav's primary focus is on **real benchmarks** produced by running GuideLLM against real models on real GPUs. The reliance on empirical data limits the range of configurations NeuralNav can recommend to only those that have been actually tested.

### Where responsibilities intersect

| Concern | config_explorer | NeuralNav |
| --- | --- | --- |
| "Will this model fit on this GPU?" | Yes (memory analysis, pre-benchmark) | Yes, indirectly (benchmark existence proves feasibility, but only for benchmarked combos) |
| "What latency/throughput will I get?" | Yes (synthetic roofline) | Yes (empirical benchmark lookup) |
| GPU cost comparison | Yes (hourly x GPU count) | Yes (hourly x GPU count x replicas x 730h) |
| SLO compliance filtering | Yes (constraint strings) | Yes (p95 SQL WHERE clauses) |
| Multi-criteria ranking | Partial (best per dimension, no composite) | Yes (4-dimensional weighted scoring + 5 views) |
| Benchmark data exploration | Yes (Pandas + Pareto + visualization) | No (benchmarks are opaque input) |

### Where they are clearly different

| Capability | config_explorer | NeuralNav |
| --- | --- | --- |
| NLU intent extraction | No | Yes (Ollama LLM) |
| Business-context mapping | No | Yes (use case -> traffic -> SLO) |
| Model quality scoring | No | Yes (Artificial Analysis benchmarks) |
| Multi-model comparison | No (single model per run) | Yes (all models meeting SLO, ranked) |
| Replica/scaling calculations | No | Yes (QPS-based planning) |
| YAML generation and K8s deployment | No | Yes (full lifecycle) |
| GPU memory estimation from architecture | Yes (empirically validated) | No |
| Synthetic performance estimation | Yes (BentoML roofline) | No |
| P/D disaggregation modeling | Yes (core to explorer.py data model) | No (aggregated vLLM only; llm-d support is a goal) |
| MoE/MLA/attention-type analysis | Yes | No |
| llm-d scheduler parameter tuning | Yes (KV cache, queue, prefix cache scorer weights) | No |

## 2. Architectural Comparison

### Core abstractions

| Aspect | config_explorer | NeuralNav |
| --- | --- | --- |
| Primary abstraction | `KVCacheDetail` dataclass (memory) + `ColumnProperties` dict (84-column benchmark DataFrame) | `DeploymentRecommendation` Pydantic model (scored config) |
| GPU representation | `GPU_SPECS` (BentoML hardware database) + `db.json` (memory specs) + `gpu_costs.json` (pricing) | `gpu_types` in `model_catalog.json` (pricing + memory) |
| Performance data source | BentoML `PerformanceEstimationResult` (synthetic) + benchmark report YAML (empirical) | PostgreSQL `exported_summaries` (empirical from GuideLLM) |
| User input | CLI args / Streamlit widgets (model ID, token lengths, GPU constraints) | Natural language -> `DeploymentIntent` Pydantic model |
| Scoring model | Per-dimension best (no composite score) | 4-dimension weighted composite with scalability penalty |

### Data models

**config_explorer**:

- `KVCacheDetail` dataclass: attention type, precision, layers, heads, per-token memory, total KV cache
- `PerformanceEstimationParams`/`Result` (BentoML): model, input/output len, GPU, num_gpus, framework, constraints -> throughput, TTFT, ITL, E2E
- Benchmark DataFrame: 84 columns covering run metadata, configuration (incl. P/D disagg + scheduler params), workload, and metrics (all latency percentiles from p0.1 to p99.9)
- `Scenario` dataclass: Streamlit session state
- `CostManager`: GPU name -> hourly cost

**NeuralNav**:

- `DeploymentIntent`: use_case, user_count, priorities, GPU preferences
- `TrafficProfile`: prompt_tokens, output_tokens, expected_qps
- `SLOTargets`: ttft_p95_target_ms, itl_p95_target_ms, e2e_p95_target_ms
- `GPUConfig`: gpu_type, gpu_count, tensor_parallel, replicas
- `ConfigurationScores`: accuracy/price/latency/complexity scores (0-100), balanced_score, slo_status
- `BenchmarkData`: flat row with mean/p90/p95/p99 for TTFT/ITL/E2E/TPS + requests_per_second + estimated flag
- `DeploymentRecommendation`: all above + cost + reasoning

### Benchmark data formats -- compatibility

| Field | llm-d benchmark report | config_explorer DataFrame | NeuralNav BenchmarkData |
| --- | --- | --- | --- |
| Model ID | `scenario.model.name` | `Model` column | `model_hf_repo` |
| GPU type | `scenario.host.accelerator.model` | `GPU` column | `hardware` |
| GPU count | `scenario.host.accelerator.count` | `Num_GPUs` column | `hardware_count` |
| Input tokens | `metrics.requests.input_length` | `ISL` column | `prompt_tokens` |
| Output tokens | `metrics.requests.output_length` | `OSL` column | `output_tokens` |
| TTFT | `metrics.latency.ttft` (Statistics with all percentiles) | `Mean_TTFT_ms` through `P99_TTFT_ms` | `ttft_mean`, `ttft_p90`, `ttft_p95`, `ttft_p99` |
| ITL | `metrics.latency.itl` | Same pattern | Same pattern |
| E2E | `metrics.latency.request_latency` | `Mean_E2EL_ms` through `P99_E2EL_ms` | `e2e_mean` through `e2e_p99` |
| Throughput | `metrics.throughput.*_per_sec` | `Request_Throughput`, `Output_Token_Throughput` | `requests_per_second`, `tokens_per_second` |
| P/D disagg | `host.type` (REPLICA/PREFILL/DECODE) | `Is_PD`, `P_TP`, `D_TP`, `P_Replicas`, `D_Replicas` | **Not supported** |
| Scheduler params | Not in v0.1 schema | `KV_Cache_Scorer_Weight`, etc. | **Not supported** |

**Conversion feasibility**: ~80-90% of core performance fields map directly between formats. P/D disaggregation and scheduler parameters have no NeuralNav equivalent yet. Unit normalization (seconds vs milliseconds) required.

### Extensibility

- **config_explorer**: Add GPUs via `gpu_costs.json` + BentoML `GPU_SPECS`. Add models by pointing at any HuggingFace model ID. Add benchmark analysis via Pandas operations. llm-d benchmark schema is versioned (v0.1, v0.2).
- **NeuralNav**: Add GPUs via `model_catalog.json`. Add models to catalog + run GuideLLM benchmarks + load to PostgreSQL. Add use cases via `slo_templates.json` + Artificial Analysis weighted score CSVs.

### API surface

- **config_explorer**: Python library API + CLI (`config-explorer plan|estimate|start`) + Streamlit UI. No REST API.
- **NeuralNav**: FastAPI REST API (`/api/v1/*`) + Python library API (service classes) + Streamlit UI.

## 3. Overlap Analysis

### Shared concepts

| Concept | config_explorer term | NeuralNav term |
| --- | --- | --- |
| Latency metrics | TTFT, ITL, E2E latency | TTFT p95, ITL p95, E2E p95 |
| Token lengths | input_len, output_len / ISL, OSL | prompt_tokens, output_tokens |
| GPU specification | GPU name + memory + count | gpu_type + gpu_count + tensor_parallel |
| Performance constraints | max_ttft, max_itl, max_latency | SLOTargets |
| Cost calculation | `CostManager.get_cost()` | `ModelCatalog.calculate_gpu_cost()` |
| Tensor parallelism | TP (from `find_possible_tp()`) | tensor_parallel (from benchmark data) |
| Estimated vs real data | (not explicitly tracked) | `estimated` flag in `benchmark_metrics` |

### Similar workflows

Both follow: "take a model + workload + GPU -> estimate/lookup performance -> filter by SLO -> rank by preference." The differences:

- **Data source**: synthetic roofline (config_explorer) vs empirical benchmarks (NeuralNav)
- **Scope**: single model per run (config_explorer) vs all qualifying models ranked (NeuralNav)
- **Deployment topology**: P/D disaggregation + scheduler tuning (config_explorer) vs aggregated only (NeuralNav)

### Redundant capabilities

1. **GPU cost lookups**: Both maintain separate GPU pricing tables. Could share a source of truth.
2. **SLO filtering**: Both filter by latency constraints -- different mechanisms (BentoML constraint strings vs SQL WHERE), same logic.
3. **"Best by metric" queries**: config_explorer's `get_gpu_with_lowest_cost()` etc. overlap with NeuralNav's ranked views, but NeuralNav operates across models and config_explorer across GPUs.

### Competing abstractions

- **Performance data**: `PerformanceEstimationResult` (opaque BentoML object) vs `BenchmarkData` (flat PostgreSQL row). Structurally incompatible.
- **GPU specs**: `GPU_SPECS` (BentoML hardware compute specs) vs `model_catalog.json gpu_types` (pricing-focused). Different schemas, different purposes.
- **Deployment topology**: config_explorer's 84-column DataFrame includes P/D disaggregation as native columns. NeuralNav's `GPUConfig` has no equivalent -- adding it requires schema changes.

## 4. Complementarity Analysis

### How config_explorer could enhance NeuralNav

1. **Fill benchmark gaps with synthetic estimates**: NeuralNav can only recommend (model, GPU) combos that have been empirically benchmarked via GuideLLM. This fundamentally limits the recommendation space. config_explorer's roofline model could produce estimated performance for unbenchmarked combos. These could be presented with **lower confidence** compared to real benchmarks, giving users a broader view of options. This directly addresses the Phase 2 TODO in `src/neuralnav/recommendation/config_finder.py:14-18`.

2. **Offline synthetic benchmark generation**: Rather than calling config_explorer at runtime, use it as an **offline tool** to pre-generate synthetic benchmarks in NeuralNav's `BenchmarkData` format (or `benchmarks_BLIS.json` schema). These would be loaded into PostgreSQL alongside real benchmarks, flagged with `estimated=True`. This avoids runtime dependencies entirely.

3. **Memory-feasibility pre-filtering**: Before querying benchmarks, NeuralNav could use `capacity_planner` to verify that a model physically fits on a GPU -- useful for error messaging and validating benchmark data integrity.

4. **Path to llm-d support**: config_explorer's data model already handles P/D disaggregation natively. As NeuralNav adds llm-d support, config_explorer's `explorer.py` column schema and benchmark report loading could inform NeuralNav's data model evolution -- particularly how to represent disaggregated configurations in `GPUConfig` and `BenchmarkData`.

5. **KV cache and concurrency analysis**: `capacity_planner.max_concurrent_requests()` could improve NeuralNav's replica calculations beyond the current `ceil(required_qps * 1.2 / benchmark_rps)`.

### How NeuralNav could enhance config_explorer

1. **Business-context-aware recommendations**: config_explorer recommends GPUs in isolation. NeuralNav's use-case -> SLO -> multi-model ranking pipeline gives business-aligned recommendations.

2. **Model quality awareness**: config_explorer has no concept of model quality -- it would happily recommend a fast but low-quality model. NeuralNav's Artificial Analysis scoring prevents this.

3. **End-to-end deployment**: config_explorer stops at recommendation. NeuralNav's YAML generation and K8s deployment turns recommendations into running services.

4. **Multi-model fleet comparison**: NeuralNav compares all qualifying models in a single request. config_explorer analyzes one model at a time.

### Integration leverage points

- **`capacity_planner` functions** -> NeuralNav's `ConfigFinder._calculate_required_replicas()` for better replica estimation
- **`GPURecommender`** -> Offline batch synthetic benchmark generation script
- **`explorer.py` data model** -> Reference for extending NeuralNav's schemas to support P/D disaggregation
- **llm-d benchmark report schema** -> Standard format for real benchmark data ingestion into NeuralNav

## 5. Feasibility of Unification

### Codebase compatibility

| Factor | Assessment |
| --- | --- |
| Language | Both Python -- compatible |
| Python version | config_explorer >=3.11, NeuralNav >=3.10 -- compatible |
| Package management | setuptools/pip vs uv -- minor friction, uv can install pip packages |
| Data validation | Both Pydantic >=2 -- compatible |
| UI framework | Both Streamlit -- compatible |
| Backend framework | config_explorer has no REST API; NeuralNav uses FastAPI |
| Key external dependency | `llm-optimizer` (BentoML, git-only, no PyPI) adds supply chain risk |

### Refactoring effort

| Integration type | Effort | Key work |
| --- | --- | --- |
| Offline synthetic benchmark generator | Low | Script to run GPURecommender batch, format output as NeuralNav JSON |
| Use config_explorer as pip dependency | Low-Medium | Add dependency, write adapter for data model translation |
| Embed capacity planning into NeuralNav | Medium | Wrap `capacity_planner` behind service interface, add HF dependency |
| Extend NeuralNav schemas for P/D disagg | Medium-High | Schema changes to GPUConfig, BenchmarkData, UI, PostgreSQL |
| Full merge of codebases | Very High | Reconcile data models, UIs, CLIs, dependencies, governance |

### Risk areas

1. **BentoML `llm-optimizer` dependency**: External git dependency with no PyPI release. Brings scipy, transformers, huggingface_hub. Supply chain and build complexity risk -- especially for offline generation, this may be acceptable since it only runs in dev/CI, not production.
2. **Synthetic vs empirical accuracy**: Roofline estimates are approximations. If mixed with empirical data, users must clearly understand confidence differences. The `estimated` flag in NeuralNav's `benchmark_metrics` exists for this purpose.
3. **HuggingFace API dependency**: config_explorer fetches model configs at runtime. For offline generation this is acceptable; for runtime integration it adds latency and failure modes.
4. **Ownership divergence**: config_explorer lives in `llm-d/llm-d-benchmark`. API changes could break downstream integrations. Offline generation is less sensitive to this since the script can pin versions.
5. **Full merge governance**: Combining codebases requires agreeing on common objectives, shared roadmap, unified release process, and operating as a single OSS team. This is an organizational decision, not just a technical one.

### Long-term maintainability

- **Offline generation** (Option A): Lowest burden. config_explorer is a dev tool, not a runtime dependency. NeuralNav consumes its output (JSON files), not its code.
- **Loose coupling** (Option B): Low burden. Adapter layer absorbs API changes.
- **Full merge** (Option C): Highest burden. Requires sustained alignment between teams on priorities, release cadence, and architecture direction. Only justified if both projects share a single roadmap.

## 6. Proposed Integration Models

### Option A: Offline Synthetic Benchmark Generation (Recommended)

Use config_explorer as an **offline dev/CI tool** to batch-generate synthetic performance estimates in NeuralNav's benchmark data format. No runtime dependency.

**Implementation**:

- Create a standalone script (e.g., `scripts/generate_synthetic_benchmarks.py`) that:
  1. Takes a list of (model_id, gpu_type, gpu_count) combos from NeuralNav's `model_catalog.json`
  2. For each combo, calls `GPURecommender` with the 4 GuideLLM traffic profiles (512->256, 1024->1024, 4096->512, 10240->1536)
  3. Translates `PerformanceEstimationResult` -> NeuralNav `benchmarks_BLIS.json` entry format
  4. Generates synthetic percentiles from mean values (p90 ~ mean x 1.05, p95 ~ mean x 1.10, p99 ~ mean x 1.20 -- or use distribution models)
  5. Writes output as JSON with `estimated: true` flag
  6. Loads into PostgreSQL alongside real benchmarks
- NeuralNav's existing `ConfigFinder` processes synthetic benchmarks identically to real ones -- the `estimated` flag is already in the schema
- UI shows confidence indication (e.g., "Estimated" vs "Benchmarked") on recommendation cards

**Pros**:

- Zero runtime dependency on config_explorer or llm-optimizer
- NeuralNav's production code doesn't change (except optional UI confidence indicator)
- Can run in CI to regenerate estimates when model catalog changes
- Clear separation: offline tool generates data, NeuralNav consumes data
- config_explorer API changes only affect the generation script, not NeuralNav core
- Expands recommendation space to any HuggingFace model, not just benchmarked ones

**Cons**:

- Synthetic percentile generation is heuristic -- no real distribution data
- Must rerun script when adding models or GPUs
- BentoML roofline accuracy is unvalidated against NeuralNav's GuideLLM real benchmarks
- Doesn't help with P/D disaggregation (GPURecommender doesn't model disaggregation)

**Complexity**: Low

**Risk**: Low (isolated to offline tooling)

### Option B: Loose Coupling (Runtime Library Integration)

NeuralNav imports config_explorer as a pip dependency, calls it through a thin adapter as a fallback when empirical benchmarks are unavailable.

**Implementation**:

- Add `config_explorer` to NeuralNav's `pyproject.toml`
- Create `src/neuralnav/adapters/config_explorer_adapter.py`
- Modify `ConfigFinder.plan_all_capacities()` to call adapter for unbenchmarked combos
- Use `capacity_planner` for memory validation

**Pros**:

- Real-time synthetic estimation for any model
- Memory validation before recommendation
- Progressive enhancement: empirical data preferred, synthetic as fallback

**Cons**:

- Brings `llm-optimizer` + `transformers` + `huggingface_hub` into NeuralNav's dependency tree at runtime
- Runtime HuggingFace API calls add latency and failure modes
- Data model impedance mismatch
- No control over config_explorer API stability

**Complexity**: Medium

**Risk**: Medium

### Option C: Full Merge

Merge both projects into a single codebase with unified data models, shared UI, and common governance.

**Implementation**:

- Agree on shared project objectives and roadmap
- Form a single OSS team with shared ownership
- Reconcile all data models (GPUConfig, BenchmarkData to support P/D disaggregation)
- Merge UI pages (NeuralNav's recommendation flow + config_explorer's capacity planning + sweep visualization)
- Unified CLI and REST API
- Single dependency tree

**Pros**:

- Tightest possible integration
- Unified data model supporting both aggregated and disaggregated deployments
- Single user experience: from capacity analysis -> recommendation -> deployment
- No adapter overhead or format translation
- Shared roadmap eliminates divergence

**Cons**:

- Requires organizational alignment: common objectives, shared governance, unified release process
- Very high engineering effort to reconcile data models, UIs, and dependencies
- Risk of scope bloat -- combining a focused tool with a comprehensive system
- BentoML `llm-optimizer` becomes a core dependency of the merged project
- Forks from upstream config_explorer -- future upstream improvements must be manually ported or the merge must be the new upstream

**Complexity**: Very High

**Risk**: High (organizational + technical)

## 7. Proof-of-Concept Plan

### Goal

Validate that config_explorer's synthetic roofline estimates are accurate enough to usefully expand NeuralNav's recommendation space, using **Option A (offline generation)** as the approach.

### Success Criteria

1. **Accuracy**: For (model, GPU, traffic_profile) combos where NeuralNav has real GuideLLM benchmarks, the roofline estimates agree within 30% on TTFT, ITL, and E2E latency for the majority (>60%) of tested combos.
2. **Coverage expansion**: The synthetic generator produces valid estimates for at least 3 model+GPU combos that NeuralNav currently cannot recommend.
3. **End-to-end flow**: Synthetic benchmarks loaded into NeuralNav are scored, ranked, and displayed with an "Estimated" indicator -- no existing recommendation quality degraded.

### Implementation Steps

#### Step 1: Accuracy validation (standalone script, no NeuralNav changes)

- Script reads NeuralNav's `benchmarks_BLIS.json`
- For each (model, GPU, input_tokens, output_tokens) combo, calls `GPURecommender`
- Compares synthetic TTFT/ITL/E2E with empirical values
- Reports: correlation, mean absolute error, % within 30%
- **Output**: Accuracy report determining whether Step 2 is worth pursuing

#### Step 2: Offline synthetic benchmark generator

- Script iterates over all (model, GPU) combos from `model_catalog.json`
- For each of the 4 GuideLLM traffic profiles, runs `GPURecommender`
- Formats output as `benchmarks_BLIS.json` entries with `estimated: true`
- Writes to `data/benchmarks/performance/benchmarks_estimated.json`

#### Step 3: Load and verify

- Load synthetic benchmarks into PostgreSQL alongside real benchmarks
- Run NeuralNav's recommendation endpoint with a use case that benefits from expanded coverage
- Verify synthetic results appear in ranked recommendations with correct scoring
- Verify existing real-benchmark recommendations are unchanged

#### Step 4: UI confidence indicator (optional)

- Add "Estimated" vs "Benchmarked" badge to recommendation cards based on `benchmark_metrics.estimated` flag

### Failure Indicators

- Roofline estimates diverge >50% from real benchmarks for majority of combos -> synthetic data not trustworthy
- `llm-optimizer` cannot be installed alongside NeuralNav's dependencies -> offline generation requires isolated virtualenv (adds friction but not a blocker)
- GPURecommender fails for models in NeuralNav's catalog (gated HF models, unsupported architectures) -> limited coverage expansion

### Decision Framework

| POC Outcome | Recommendation |
| --- | --- |
| Accuracy <30% error for >60% of combos | Proceed with Option A (offline generation) |
| Accuracy 30-50% error | Proceed with caveats -- label estimates as "rough estimates" in UI |
| Accuracy >50% error | Do not integrate -- synthetic data not useful for NeuralNav's ranking |
| Independent of accuracy | Use config_explorer's P/D disaggregation data model as reference for NeuralNav's llm-d support roadmap |
