# Programmatic API User Guide

llm-d Planner provides two programmatic interfaces for building LLM deployment recommendations:

1. **Python Library API** - Import `Planner` class directly in Python programs
2. **REST API** - HTTP endpoints for language-agnostic integration

Both expose the same pipeline with the same schemas. Use the library for in-process Python integration, or the REST API for cross-language/microservice architectures.

---

## Installation

### Core Library

```bash
pip install llm-d-planner
```

This installs the core recommendation engine with no external dependencies. You can generate specifications, recommendations, and deployment YAML with just Python.

### Optional Extras

Install additional features as needed:

```bash
# REST API server
pip install llm-d-planner[server]

# LLM-powered intent extraction (Ollama)
pip install llm-d-planner[llm]

# LLM-powered intent extraction (OpenAI)
pip install llm-d-planner[openai]

# LLM-powered intent extraction (Vertex AI)
pip install llm-d-planner[vertex]

# Kubernetes deployment support
pip install llm-d-planner[kubernetes]

# Roofline estimation for missing benchmarks
pip install llm-d-planner[estimation]

# Streamlit UI
pip install llm-d-planner[ui]

# Quality data sync from Arena/AA APIs
pip install llm-d-planner[quality-sync]

# Development tools (pytest, ruff, mypy)
pip install llm-d-planner[dev]

# All extras
pip install llm-d-planner[server,llm,kubernetes,estimation,ui,quality-sync,dev]
```

---

## Quickstart

### Python Library

```python
from planner import Planner, DeploymentIntent

p = Planner()
p.load_bundled_benchmarks()

spec = p.generate_specification(DeploymentIntent(
    use_case="chatbot_conversational",
    user_count=1000,
))

recs = p.generate_recommendations(spec)

if recs.balanced:
    bundle = p.generate_deployment(recs.balanced[0].configuration)
    for name, yaml in bundle.files.items():
        print(f"--- {name} ---\n{yaml}")
```

### REST API

Requires `pip install llm-d-planner[server]`.

**Start the server:**

```bash
uvicorn planner.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

**Call endpoints:**

```bash
# Generate specification from intent
curl -X POST http://localhost:8000/api/v1/generate-specification \
  -H "Content-Type: application/json" \
  -d '{"use_case": "chatbot_conversational", "user_count": 1000}'

# Generate recommendations from specification
curl -X POST http://localhost:8000/api/v1/generate-recommendations \
  -H "Content-Type: application/json" \
  -d '{"specification": {...}}'
```

See the **REST API** section below for complete endpoint documentation.

---

## Python Library API

### The `Planner` Class

The `Planner` class is the main entry point for the library. It provides methods for each stage of the recommendation pipeline.

#### Initialization

```python
from planner import Planner, PlannerConfig

# Default — bundled config and quality data, empty benchmark DB
p = Planner()

# Keyword shorthand — any PlannerConfig field works as a kwarg
p = Planner(data_dir="/custom/data")
p = Planner(llm_provider="openai", llm_api_key="sk-...")

# Config object — groups all settings, serializable, IDE-friendly
config = PlannerConfig(
    llm_provider="openai",
    llm_api_key="sk-...",
    quality_auto_update=True,
    aa_api_key="aa-...",
)
p = Planner(config)
```

#### `PlannerConfig` Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `data_dir` | `Path \| None` | `None` | Custom data directory (default: bundled package data) |
| `llm_provider` | `str \| None` | `None` | LLM provider: `"ollama"`, `"openai"`, or `"vertex"` |
| `llm_api_key` | `str \| None` | `None` | API key for OpenAI/Vertex provider |
| `llm_base_url` | `str \| None` | `None` | Base URL for OpenAI-compatible endpoints or Ollama host |
| `llm_model` | `str \| None` | `None` | Model name override |
| `quality_auto_update` | `bool` | `False` | Fetch fresh Arena/AA data on init when cache is stale. Cache is written to `quality_cache_dir` (or `.quality_cache/` in CWD). Requires `aa_api_key` for AA data. |
| `quality_cache_dir` | `Path \| None` | `None` | Directory for runtime quality cache. Default: `.quality_cache/` in current working directory, or `LLM_QUALITY_CACHE_DIR` env var. Only used when `quality_auto_update=True`. |
| `aa_api_key` | `str \| None` | `None` | Artificial Analysis API key (for quality data refresh). Falls back to `AA_API_KEY` env var. |
| `hf_token` | `str \| None` | `None` | HuggingFace token (for model config lookups) |
| `model_catalog_url` | `str \| None` | `None` | Model Catalog API URL (for `sync_model_catalog()`) |
| `model_catalog_token` | `str \| None` | `None` | Auth token for Model Catalog API |

All fields are optional. `Planner()` with no arguments uses bundled data and works without any external services.

**Thread safety:** The `Planner` class holds mutable state (SQLite connection, scoring engine). It is not thread-safe. Each thread or async task should use its own `Planner` instance, or callers must synchronize access externally.

---

### Data Loading

Benchmark data is not auto-loaded. Call one of these methods before generating recommendations:

```python
# Load bundled BLIS benchmark data (included in wheel)
p.load_bundled_benchmarks()

# Load custom benchmark file
p.load_benchmarks("/path/to/my_benchmarks.json")

# Sync from model catalog service (fetches benchmarks from external catalog API)
p.sync_model_catalog()
```

Calling `generate_recommendations()` without loading benchmarks raises:

```
PlannerError: No benchmarks loaded. Call load_bundled_benchmarks() or load_benchmarks(path) first.
```

---

### Pipeline Methods

Each method mirrors a REST endpoint and accepts/returns Pydantic models (not plain dicts).

#### `extract_intent(text: str) -> DeploymentIntent`

Extract structured intent from natural language using an LLM.

**Requires:** LLM provider configured during initialization (`llm_provider` parameter)

**Example:**

```python
p = Planner(llm_provider="openai", llm_api_key="sk-...")
intent = p.extract_intent("I need a chatbot for 1000 users, low latency is critical")
```

**Returns:** `DeploymentIntent` with all fields populated (defaults filled in).

---

#### `generate_specification(intent: DeploymentIntent) -> DeploymentSpecification`

Generate a complete deployment specification from structured intent. No LLM required.

**Requires:** Nothing (reads from static config files)

**Example:**

```python
from planner import DeploymentIntent

spec = p.generate_specification(DeploymentIntent(
    use_case="chatbot_conversational",
    user_count=1000,
    latency_priority="high",
))
```

**Returns:** `DeploymentSpecification` with 4 sections:
- `intent` - Original intent with defaults filled in
- `slo_targets` - TTFT, ITL, E2E latency targets with ranges
- `workload_profile` - Prompt/output tokens, expected QPS
- `quality_weights` - Per-use-case category weights for quality scoring (read-only; shows weights in effect)
- `priorities` - Quality/cost/latency priority levels and resolved weights

---

#### `generate_recommendations(spec: DeploymentSpecification, ...) -> RankedRecommendations`

Generate ranked deployment recommendations from a specification.

**Requires:** Benchmark data (call `load_bundled_benchmarks()` or `load_benchmarks()` first)

**Parameters:**

- `spec: DeploymentSpecification` - The specification (from `generate_specification()`, possibly user-modified)
- `enable_estimated: bool = False` - Run roofline estimation for missing benchmarks
- `min_quality: float | None = None` - Minimum quality score filter
- `max_cost: float | None = None` - Maximum cost filter (USD/month)
- `include_near_miss: bool = True` - Include near-miss configurations (slightly exceed SLO targets)

**Example:**

```python
recs = p.generate_recommendations(spec)

# With filters
recs = p.generate_recommendations(
    spec,
    enable_estimated=True,
    min_quality=50.0,
    max_cost=10000.0,
)
```

**Returns:** `RankedRecommendations` with 4 ranked views:
- `balanced` - Sorted by weighted composite score
- `best_quality` - Sorted by model capability
- `lowest_cost` - Sorted by price efficiency
- `lowest_latency` - Sorted by SLO headroom

---

#### `generate_deployment(config: DeploymentConfiguration, ...) -> DeploymentBundle`

Generate Kubernetes deployment files (YAML) for a selected configuration.

**Requires:** Nothing (Jinja2 template rendering)

**Parameters:**

- `config: DeploymentConfiguration` - The configuration (extract from `DeploymentRecommendation.configuration`)
- `namespace: str = "default"` - Kubernetes namespace
- `stack: str = "vllm"` - Deployment stack: `"vllm"` or `"llm-d"`

**Example:**

```python
# Select the top balanced recommendation
if recs.balanced:
    bundle = p.generate_deployment(
        recs.balanced[0].configuration,
        namespace="production",
        stack="vllm",
    )
    
    # Print YAML files
    for name, yaml in bundle.files.items():
        print(f"--- {name} ---\n{yaml}")
```

**Returns:** `DeploymentBundle` with generated YAML files in `files` dict.

---

#### `deploy_bundle_to_cluster(bundle: DeploymentBundle) -> dict`

Deploy a deployment bundle to a Kubernetes cluster.

**Requires:** `kubectl` in PATH and a configured kubeconfig

**Parameters:**

- `bundle: DeploymentBundle` - The bundle (from `generate_deployment()`, possibly with user-modified YAML)

**Example:**

```python
result = p.deploy_bundle_to_cluster(bundle)
print(f"Deployed: {result['deployment_id']}")
```

**Returns:** Dict with `deployment_id` and deployment metadata.

---

### Method Signatures

```python
class Planner:
    def __init__(
        self,
        config: PlannerConfig | None = None,
        **kwargs,  # shorthand for PlannerConfig fields
    ) -> None: ...

    # Data loading
    def load_bundled_benchmarks(self) -> None: ...
    def load_benchmarks(self, path: str | Path) -> None: ...
    def sync_model_catalog(
        self, url: str | None = None, token: str | None = None,
    ) -> dict: ...

    # Pipeline
    def extract_intent(self, text: str) -> DeploymentIntent: ...
    def generate_specification(
        self, intent: DeploymentIntent
    ) -> DeploymentSpecification: ...
    def generate_recommendations(
        self,
        spec: DeploymentSpecification,
        enable_estimated: bool = False,
        min_quality: float | None = None,
        max_cost: float | None = None,
        include_near_miss: bool = True,
    ) -> RankedRecommendations: ...
    def generate_deployment(
        self,
        config: DeploymentConfiguration,
        namespace: str = "default",
        stack: str = "vllm",
    ) -> DeploymentBundle: ...
    def deploy_bundle_to_cluster(self, bundle: DeploymentBundle) -> dict: ...
```

---

### Dependency Requirements Per Method

| Method | Core install | Needs benchmarks | Needs LLM | Needs extra |
|---|---|---|---|---|
| `generate_specification()` | Yes | No | No | — |
| `generate_recommendations()` | Yes | Yes | No | — |
| `generate_deployment()` | Yes | No | No | — |
| `extract_intent()` | — | No | Yes | `[llm]`, `[openai]`, or `[vertex]` |
| `deploy_bundle_to_cluster()` | — | No | No | `kubectl` in PATH |

---

### Complete Usage Examples

#### Minimal — spec to YAML with core install only

```python
from planner import Planner, DeploymentIntent

p = Planner()
p.load_bundled_benchmarks()

spec = p.generate_specification(DeploymentIntent(
    use_case="chatbot_conversational",
    user_count=1000,
))

recs = p.generate_recommendations(spec)

if recs.balanced:
    bundle = p.generate_deployment(recs.balanced[0].configuration)
    for name, yaml in bundle.files.items():
        print(f"--- {name} ---\n{yaml}")
```

---

#### With LLM intent extraction

Requires `pip install llm-d-planner[openai]` (or `llm-d-planner[llm]` for Ollama, `llm-d-planner[vertex]` for Vertex).

```python
from planner import Planner

p = Planner(llm_provider="openai", llm_api_key="sk-...")
p.load_bundled_benchmarks()

intent = p.extract_intent("I need a code completion service for 500 developers")
spec = p.generate_specification(intent)
recs = p.generate_recommendations(spec)

for rec in recs.balanced[:3]:
    print(f"{rec.model_name} on {rec.gpu_config.gpu_type}: ${rec.cost_per_month_usd:.2f}/mo")
```

---

#### With custom benchmarks

```python
from planner import Planner, DeploymentIntent

p = Planner()
p.load_benchmarks("/data/my_custom_benchmarks.json")

spec = p.generate_specification(DeploymentIntent(
    use_case="document_analysis_rag",
    user_count=200,
))

recs = p.generate_recommendations(spec)
```

---

#### Full pipeline with deployment

Requires `pip install llm-d-planner[openai]` and `kubectl` in PATH.

```python
from planner import Planner

p = Planner(llm_provider="openai", llm_api_key="sk-...")
p.load_bundled_benchmarks()

intent = p.extract_intent("I need a chatbot for 1000 users, low latency is critical")
spec = p.generate_specification(intent)
recs = p.generate_recommendations(spec)

if recs.balanced:
    bundle = p.generate_deployment(recs.balanced[0].configuration)
    result = p.deploy_bundle_to_cluster(bundle)
    print(f"Deployed: {result['deployment_id']}")
```

---

#### Exploring multiple ranked views

```python
from planner import Planner, DeploymentIntent

p = Planner()
p.load_bundled_benchmarks()

spec = p.generate_specification(DeploymentIntent(
    use_case="chatbot_conversational",
    user_count=1000,
))

recs = p.generate_recommendations(spec)

print("Best quality:")
for rec in recs.best_quality[:3]:
    print(f"  {rec.model_name}: quality={rec.scores.quality_score:.1f}")

print("\nLowest cost:")
for rec in recs.lowest_cost[:3]:
    print(f"  {rec.model_name}: ${rec.cost_per_month_usd:.2f}/mo")

print("\nLowest latency:")
for rec in recs.lowest_latency[:3]:
    print(f"  {rec.model_name}: TTFT={rec.predicted_ttft_p95_ms}ms")

print("\nBalanced:")
for rec in recs.balanced[:3]:
    print(f"  {rec.model_name}: score={rec.scores.balanced_score:.1f}")
```

---

## REST API

The REST API requires `pip install llm-d-planner[server]`.

**Start the server:**

```bash
uvicorn planner.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

**Interactive API docs:** http://localhost:8000/docs

---

### Pipeline Endpoints

The API is organized as a composable pipeline where each stage's output feeds as input to the next:

```
extract-intent → generate-specification → generate-recommendations → generate-deployment → deploy-bundle-to-cluster
```

Each stage is independent. Users can:
- Enter at any stage by constructing the input object
- Exit at any stage with the output object
- Skip stages entirely
- Modify any object between stages

---

### `POST /api/v1/extract-intent`

Extract structured intent from natural language using an LLM.

**Request:**

```json
{"text": "I need a chatbot for 1000 users, low latency is critical"}
```

**Response:** A `DeploymentIntent` with all fields populated (defaults filled in).

```json
{
  "use_case": "chatbot_conversational",
  "user_count": 1000,
  "domain_specialization": ["general"],
  "preferred_gpu_types": [],
  "preferred_models": [],
  "quality_priority": "medium",
  "cost_priority": "medium",
  "latency_priority": "high"
}
```

**Requires:** LLM (Ollama or configured provider)

---

### `POST /api/v1/generate-specification`

Generate a complete deployment specification from structured intent. No LLM required.

**Request:** A `DeploymentIntent`. Only `use_case` and `user_count` are required; all other fields are filled with defaults.

```json
{
  "use_case": "chatbot_conversational",
  "user_count": 1000,
  "latency_priority": "high"
}
```

**Response:** A `DeploymentSpecification` with all 4 sections populated.

```json
{
  "intent": {
    "use_case": "chatbot_conversational",
    "user_count": 1000,
    "domain_specialization": ["general"],
    "preferred_gpu_types": [],
    "preferred_models": [],
    "quality_priority": "medium",
    "cost_priority": "medium",
    "latency_priority": "high"
  },
  "slo_targets": {
    "ttft_target_ms": 200,
    "itl_target_ms": 24,
    "e2e_target_ms": 6280,
    "percentile": "p95",
    "ttft_range": {"min": 100, "max": 500},
    "itl_range": {"min": 15, "max": 50},
    "e2e_range": {"min": 3940, "max": 13300}
  },
  "workload_profile": {
    "prompt_tokens": 512,
    "output_tokens": 256,
    "expected_qps": 0.87
  },
  "quality_weights": {
    "categories": {
      "overall": 4,
      "instruction_following": 3,
      "multi_turn": 3,
      "creative_writing": 2,
      "hard_prompts": 2
    }
  },
  "priorities": {
    "quality": {"priority": "medium", "weight": 4},
    "cost": {"priority": "medium", "weight": 4},
    "latency": {"priority": "high", "weight": 2}
  }
}
```

Note: `latency_priority` is `"high"`, so SLO defaults use the 25th percentile of each range (tighter targets).

**Requires:** Nothing (reads from static config files)

---

### `POST /api/v1/generate-recommendations`

Generate ranked deployment recommendations from a specification.

**Request:** A `RecommendationRequest` containing a `DeploymentSpecification` (the output of `generate-specification`, with optional user modifications). The weights come from the specification's `priorities` section.

Additional optional fields on the request (not part of the specification):
- `enable_estimated: bool = false` — run roofline estimation for missing benchmarks
- `min_quality: float | null` — minimum quality score filter
- `max_cost: float | null` — maximum cost filter
- `include_near_miss: bool = true` — include near-miss configurations

```json
{
  "specification": {
    "intent": {...},
    "slo_targets": {...},
    "workload_profile": {...},
    "quality_weights": {...},
    "priorities": {...}
  },
  "enable_estimated": true,
  "min_quality": 50.0,
  "max_cost": 10000.0,
  "include_near_miss": true
}
```

**Response:** A `RankedRecommendations` object with 4 ranked views.

```json
{
  "specification": {...},
  "balanced": [
    {
      "model_id": "meta-llama/Llama-3.1-8B-Instruct",
      "model_name": "Llama 3.1 8B Instruct",
      "gpu_config": {
        "gpu_type": "NVIDIA-L4",
        "gpu_count": 1,
        "tensor_parallel": 1,
        "replicas": 1
      },
      "cost_per_month_usd": 350.0,
      "scores": {
        "quality_score": 75.5,
        "price_score": 85,
        "latency_score": 90,
        "balanced_score": 82.3,
        "slo_status": "compliant"
      },
      "configuration": {...}
    }
  ],
  "best_quality": [...],
  "lowest_cost": [...],
  "lowest_latency": [...]
}
```

**Requires:** Benchmark database

---

### `POST /api/v1/generate-deployment`

Generate deployment files (YAML) for a selected configuration.

**Request:**

```json
{
  "configuration": {
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "model_name": "Llama 3.1 8B Instruct",
    "model_uri": "meta-llama/Llama-3.1-8B-Instruct",
    "gpu_config": {
      "gpu_type": "NVIDIA-L4",
      "gpu_count": 1,
      "tensor_parallel": 1,
      "replicas": 1
    },
    "use_case": "chatbot_conversational",
    "expected_qps": 0.87,
    "prompt_tokens": 512,
    "output_tokens": 256,
    "e2e_target_ms": 6280
  },
  "namespace": "default",
  "stack": "vllm"
}
```

Parameters:
- `configuration` — the `DeploymentConfiguration` extracted from the selected recommendation
- `namespace` — Kubernetes namespace (default: `"default"`)
- `stack` — deployment stack: `"vllm"` or `"llm-d"` (default: `"vllm"`)

The `stack` parameter selects which set of Jinja2 templates to use:
- `vllm` — generates KServe `InferenceService` with vLLM runtime, HPA autoscaling
- `llm-d` — generates llm-d-specific resources (EPP, PD split, routing)

**Response:** A `DeploymentBundle` with generated YAML files.

```json
{
  "deployment_id": "chatbot-1000-users-abc123",
  "namespace": "default",
  "stack": "vllm",
  "configuration": {...},
  "files": {
    "inferenceservice": "apiVersion: serving.kserve.io/v1beta1\nkind: InferenceService\n...",
    "autoscaling": "apiVersion: autoscaling/v2\nkind: HorizontalPodAutoscaler\n..."
  }
}
```

**Requires:** Nothing (Jinja2 template rendering)

---

### `POST /api/v1/deploy-bundle-to-cluster`

Deploy a deployment bundle to a Kubernetes cluster. This endpoint accepts a `DeploymentBundle` (from `generate-deployment`, possibly with user-modified YAML) and deploys it directly without re-generating. YAML content from `bundle.files` is applied directly via `kubectl apply -f -` (piped to stdin) — no intermediate files are written to disk.

**Request:**

```json
{
  "deployment_id": "chatbot-1000-users-abc123",
  "namespace": "default",
  "stack": "vllm",
  "configuration": {...},
  "files": {
    "inferenceservice": "apiVersion: serving.kserve.io/v1beta1\n...",
    "autoscaling": "apiVersion: autoscaling/v2\n..."
  }
}
```

**Response:**

```json
{
  "deployment_id": "chatbot-1000-users-abc123",
  "status": "deployed",
  "namespace": "default"
}
```

**Requires:** Kubernetes cluster access

---

## Pipeline Objects

All pipeline stages use named Pydantic models for input and output. These are the same models used by both the Python library and REST API.

### `DeploymentIntent`

The input to `generate-specification` and the output of `extract-intent`.

#### Required fields

| Field | Type | Valid values |
|---|---|---|
| `use_case` | string | `chatbot_conversational`, `code_completion`, `code_generation_detailed`, `translation`, `content_generation`, `summarization_short`, `document_analysis_rag`, `long_document_summarization`, `research_legal_analysis` |
| `user_count` | integer | Any positive integer |

#### Optional fields with constrained values

| Field | Type | Default | Valid values |
|---|---|---|---|
| `quality_priority` | string | `"medium"` | `low`, `medium`, `high` |
| `cost_priority` | string | `"medium"` | `low`, `medium`, `high` |
| `latency_priority` | string | `"medium"` | `low`, `medium`, `high` |

#### Optional fields with known values (open-ended)

| Field | Type | Default | Known values | Notes |
|---|---|---|---|---|
| `preferred_gpu_types` | list | `[]` | `L4`, `A10G`, `A100-40`, `A100-80`, `H100`, `H200`, `B200`, `MI300X`, `L40`, `L20`, `B100` | Normalized via `gpu_normalizer.py`; aliases accepted (e.g., `NVIDIA-H100`). Unknown values are skipped with a warning. Empty list = no GPU preference. |
| `preferred_models` | list | `[]` | 47 models in catalog | HuggingFace format (e.g., `meta-llama/Llama-3.1-8B-Instruct`). Can be catalog models or arbitrary HF repo IDs. Empty list = no model preference. |
| `domain_specialization` | list | `["general"]` | `general`, `code`, `enterprise`, `multilingual`, `reasoning`, `vision` | Not currently used by the recommendation engine, but will be used in the future to influence quality benchmark weight selection. |

#### GPU count limits

Each entry in `preferred_gpu_types` can optionally include a maximum GPU count:

```json
{
  "preferred_gpu_types": [
    "L4",
    {"gpu_type": "H100", "max_count": 4},
    {"gpu_type": "H200", "max_count": 2}
  ]
}
```

Plain strings mean no GPU count limit. Objects with `max_count` set a ceiling — configurations requiring more GPUs of that type are filtered out.

---

### `DeploymentSpecification`

The output of `generate-specification` and the input to `generate-recommendations`. Contains 4 sections plus the original intent.

#### Fields

| Field | Type | Description |
|---|---|---|
| `intent` | `DeploymentIntent` | The original intent (echoed back with defaults filled in) |
| `slo_targets` | `SLOTargets` | Latency targets at a given percentile |
| `workload_profile` | `WorkloadProfile` | Token counts, expected QPS |
| `quality_weights` | `QualityWeights` | Per-use-case category weights for quality scoring (read-only) |
| `priorities` | `Priorities` | Quality/cost/latency priority levels and resolved weights |

#### `SLOTargets` fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `ttft_target_ms` | integer | Derived from use case + latency priority | Time to First Token target. User can enter any positive value; the default is within the recommended range. |
| `itl_target_ms` | integer | Derived from use case + latency priority | Inter-Token Latency target |
| `e2e_target_ms` | integer | Derived from use case + latency priority | End-to-end latency target |
| `percentile` | string | `"p95"` | Which percentile the targets apply to. Valid values: `p90`, `p95`, `p99`. **Note:** only `p95` is currently supported by the backend benchmark data. |
| `ttft_range` | object | From use case template | `{"min": int, "max": int}` — the recommended range. Informational; not enforced. |
| `itl_range` | object | From use case template | `{"min": int, "max": int}` — the recommended range. Informational; not enforced. |
| `e2e_range` | object | From use case template | `{"min": int, "max": int}` — the recommended range. Informational; not enforced. |

**SLO default value selection:** The default target value is derived from the recommended range using the `latency_priority` from the intent:

| Latency priority | Range percentile | Effect |
|---|---|---|
| `high` | 25th percentile | Tighter target, closer to min (aggressive) |
| `medium` | 50th percentile | Middle of range (balanced) |
| `low` | 75th percentile | Relaxed target, closer to max (permissive) |

#### `WorkloadProfile` fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt_tokens` | integer | From use case template | Mean input token length per request (GuideLLM traffic profile) |
| `output_tokens` | integer | From use case template | Mean output token length per request |
| `expected_qps` | float | Calculated from `user_count` + per-use-case workload parameters | Expected queries per second. Includes peak capacity buffer. |

#### `QualityWeights` fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `categories` | dict[string, int] | From `quality_weights.json` by use case | **Read-only.** Per-category weights used for quality scoring. Shows the weights in effect; changes are not currently honored. Keys are category names (e.g., `overall`, `coding`, `math`), values are relative integer weights. |

#### `Priorities` fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `quality` | `PriorityEntry` | From intent's `quality_priority` | Priority level and resolved weight for model quality |
| `cost` | `PriorityEntry` | From intent's `cost_priority` | Priority level and resolved weight for cost efficiency |
| `latency` | `PriorityEntry` | From intent's `latency_priority` | Priority level and resolved weight for latency/SLO headroom |

Each `PriorityEntry` contains:

```json
{"priority": "low" | "medium" | "high", "weight": int}
```

---

### `RankedRecommendations`

The output of `generate-recommendations`. Contains 4 ranked views of deployment configurations.

| Field | Type | Notes |
|---|---|---|
| `specification` | `DeploymentSpecification` | The specification used to generate recommendations (echoed back) |
| `balanced` | list of `DeploymentRecommendation` | Sorted by weighted composite score |
| `best_quality` | list of `DeploymentRecommendation` | Sorted by model capability |
| `lowest_cost` | list of `DeploymentRecommendation` | Sorted by price efficiency |
| `lowest_latency` | list of `DeploymentRecommendation` | Sorted by SLO headroom |
| `total_configs_evaluated` | integer | Total configurations considered |
| `configs_after_filters` | integer | Configurations after SLO/quality/cost filters |

---

### `DeploymentRecommendation`

A single recommended configuration. Selected from `RankedRecommendations` for user review.

| Field | Type | Notes |
|---|---|---|
| `model_id` | string or null | Recommended model identifier |
| `model_name` | string or null | Human-readable model name |
| `model_uri` | string or null | Model artifact URI |
| `gpu_config` | `GPUConfig` or null | GPU type, count, tensor parallelism |
| `predicted_ttft_p95_ms` | integer or null | Predicted TTFT |
| `predicted_itl_p95_ms` | integer or null | Predicted ITL |
| `predicted_e2e_p95_ms` | integer or null | Predicted E2E latency |
| `predicted_throughput_qps` | float or null | Predicted throughput |
| `benchmark_metrics` | dict or null | All percentile metrics from benchmark |
| `cost_per_hour_usd` | float or null | Hourly cost estimate |
| `cost_per_month_usd` | float or null | Monthly cost estimate |
| `meets_slo` | boolean | Whether configuration meets SLO targets |
| `reasoning` | string | Explanation of recommendation |
| `scores` | `ConfigurationScores` or null | Multi-criteria scores for ranking (see below) |
| `configuration` | `DeploymentConfiguration` or null | Embedded deployment parameters for YAML generation. Extract and pass to `generate-deployment`. |

`ConfigurationScores` fields:

| Field | Type | Notes |
|---|---|---|
| `quality_score` | float | Model quality/capability score (0–100) |
| `price_score` | int | Cost efficiency score (0–100) |
| `latency_score` | int | SLO headroom score (0–100) |
| `balanced_score` | float | Weighted composite score (0–100) |
| `slo_status` | string | `"compliant"`, `"near_miss"`, or `"exceeds"` |

---

### `DeploymentConfiguration`

A slim model containing only the fields needed for YAML generation. Extracted from `DeploymentRecommendation` and passed to `generate-deployment`.

| Field | Type | Notes |
|---|---|---|
| `model_id` | string | Model identifier (e.g., `meta-llama/Llama-3.1-8B-Instruct`) |
| `model_name` | string | Human-readable model name (e.g., `Llama 3.1 8B Instruct`) |
| `model_uri` | string | Model artifact URI |
| `gpu_config` | `GPUConfig` | GPU type, count, tensor parallelism |
| `use_case` | string | Use case (e.g., `chatbot_conversational`) |
| `expected_qps` | float | Expected queries per second |
| `prompt_tokens` | integer | Mean input token length |
| `output_tokens` | integer | Mean output token length |
| `e2e_target_ms` | integer | End-to-end latency target |

---

### `DeploymentBundle`

The output of `generate-deployment` and the input to `deploy-bundle-to-cluster`.

| Field | Type | Notes |
|---|---|---|
| `deployment_id` | string | Unique deployment identifier |
| `namespace` | string | Kubernetes namespace |
| `stack` | string | `"vllm"` or `"llm-d"` |
| `configuration` | `DeploymentConfiguration` | The configuration used to generate these files |
| `files` | dict[string, string] | Map of filename to YAML content (e.g., `{"inferenceservice": "...", "autoscaling": "..."}`). Files are applied in iteration order by `deploy-bundle-to-cluster`; order matters if resources have dependencies. |

---

## Reference: Demo Scripts

### `scripts/wheel_demo.py` — Python Library Demo

Interactive demo of the Planner library API (no server required). Walks through the full pipeline using the `Planner` class directly:

```bash
# From the repo (development)
uv run python scripts/wheel_demo.py

# Or from an installed wheel (isolated testing)
python wheel_demo.py
```

See the comments at the top of the script for setup instructions for isolated wheel testing. Once `llm-d-planner` is published to PyPI, just `pip install llm-d-planner` and run the script. Build a local wheel (`uv build`) when testing unpublished changes.

### `scripts/pipeline_demo.py` — REST API Demo

Interactive demo of the REST API pipeline. Requires a running backend server:

```bash
cd /path/to/llm-d-planner
uv run python scripts/pipeline_demo.py
```

Both scripts demonstrate the same pipeline stages with the same inputs — use `wheel_demo.py` for library integration examples and `pipeline_demo.py` for REST API integration examples.

---

## Error Handling

### Custom Exception

```python
from planner import PlannerError
```

Used for:
- No benchmarks loaded
- No LLM configured
- Invalid use case
- Missing required fields

### Import Guards

Modules importing optional dependencies raise helpful errors when called without the required extra:

**LLM providers** — error raised when `extract_intent()` is called:

```
ImportError: Install llm-d-planner[openai] for OpenAI support: pip install llm-d-planner[openai]
```

**Kubernetes** — error raised when `deploy_bundle_to_cluster()` is called:

```
PlannerError: Failed to apply inferenceservice: kubectl not found in PATH
```

**llm-optimizer** — warning logged, estimation skipped (not an error):

```
WARNING: llm-optimizer not installed. Roofline estimation disabled.
Install from: pip install llm-optimizer@git+https://github.com/bentoml/llm-optimizer.git
```

### No LLM Configured

If `extract_intent()` is called without `llm_provider` set:

```
PlannerError: No LLM provider configured. Pass llm_provider to Planner(), e.g.:
  Planner(llm_provider="openai", llm_api_key="sk-...")
```

---

## Data Override

The `data_dir` field on `PlannerConfig` (or kwarg on `Planner()`) allows overriding bundled configuration data files:

```python
p = Planner(data_dir="/custom/data")
# or
p = Planner(PlannerConfig(data_dir="/custom/data"))
```

If provided, config files (`configuration/`) load from the custom directory instead of bundled defaults. Quality scoring data (Arena/AA benchmarks) is bundled with the `quality_scoring` package separately and is not affected by `data_dir`.

**Note:** `load_bundled_benchmarks()` always loads from bundled wheel data regardless of `data_dir` — it's a known, fixed dataset. Custom benchmarks are loaded via `load_benchmarks(path)` from any path.

Expected directory structure for custom data:

```
/custom/data/
  configuration/
    model_catalog.json
    gpu_catalog.json
    usecase_slo_workload.json
    quality_weights.json
    priority_weights.json
```
