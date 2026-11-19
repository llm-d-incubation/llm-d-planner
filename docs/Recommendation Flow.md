# Planner Recommendation Flow

This document describes the end-to-end recommendation flow implemented in the Planner codebase. It traces data from user input through to ranked deployment recommendations.

## Overview

The recommendation flow is **SLO-driven**: the system searches all (model, GPU) configurations in the benchmark database that meet the user's SLO targets, then scores each on three criteria (quality, price, latency). This means the search is driven by requirements, not by a pre-selected model.

```
User describes use case
    ↓
Intent Extraction (LLM or form input)
    ↓
Generate Specification
    - Traffic profile (prompt/output tokens, expected QPS)
    - SLO targets (TTFT, ITL, E2E ranges + defaults)
    - Quality weights (per-use-case category importance)
    - Scoring priorities (quality, cost, latency weights)
    ↓
User reviews and edits specification
    ↓
Query benchmark database for SLO-compliant configurations
    ↓
Score each configuration (quality, price, latency)
    ↓
Return 4 ranked lists (balanced, quality, cost, latency)
    ↓
User selects a configuration
    ↓
Generate deployment manifests (YAML)
```

---

## API Endpoints

| Endpoint | Purpose | Returns |
|----------|---------|---------|
| `POST /api/v1/extract-intent` | Extract structured intent from natural language | `DeploymentIntent` |
| `POST /api/v1/generate-specification` | Generate specification from intent | `DeploymentSpecification` |
| `POST /api/v1/generate-recommendations` | Multi-criteria ranking from specification | 4 ranked lists |
| `POST /api/v1/generate-deployment` | Generate Kubernetes YAML from configuration | `DeploymentBundle` |
| `POST /api/v1/deploy-bundle-to-cluster` | Deploy bundle to Kubernetes cluster | Deployment result |

**Entry Point**: [src/planner/api/routes/](../src/planner/api/routes/)

---

## Step-by-Step Flow

### Step 1: Intent Extraction

**File**: [src/planner/intent_extraction/extractor.py](../src/planner/intent_extraction/extractor.py)

The `IntentExtractor` uses an LLM (Ollama qwen2.5:7b) to parse the user's natural language request into structured deployment intent.

**Input**: User message (e.g., "I need a chatbot for 1000 users, low latency is critical")

**Output**: `DeploymentIntent` object containing:
- `use_case`: Mapped to one of 9 supported use cases
- `user_count`: Number of concurrent users
- `quality_priority`: high, medium, low
- `cost_priority`: high, medium, low
- `latency_priority`: high, medium, low
- `preferred_gpu_types`: Optional list of GPU type preferences
- `preferred_models`: Optional list of preferred model IDs
- `domain_specialization`: Optional list of domains

**Key Function**:
```python
intent = intent_extractor.extract_intent(user_message, conversation_history)
intent = intent_extractor.infer_missing_fields(intent)
```

---

### Step 2: Traffic Profile Generation

**File**: [src/planner/specification/traffic_profile.py](../src/planner/specification/traffic_profile.py)

The `TrafficProfileGenerator` maps the use case to a GuideLLM traffic profile and calculates SLO targets.

**Input**: `DeploymentIntent`

**Output**:
- `TrafficProfile`: prompt_tokens, output_tokens, expected_qps
- `SLOTargets`: ttft_target_ms, itl_target_ms, e2e_target_ms

**Data Source**: [src/planner/data/configuration/usecase_slo_workload.json](../src/planner/data/configuration/usecase_slo_workload.json)

**Traffic Profiles** (aligned with GuideLLM):
| Use Case | Prompt Tokens | Output Tokens |
|----------|--------------|---------------|
| chatbot_conversational | 512 | 256 |
| code_completion | 512 | 256 |
| code_generation_detailed | 1024 | 1024 |
| translation | 512 | 256 |
| content_generation | 512 | 256 |
| summarization_short | 4096 | 512 |
| document_analysis_rag | 4096 | 512 |
| long_document_summarization | 10240 | 1536 |
| research_legal_analysis | 10240 | 1536 |

**Key Functions**:
```python
traffic_profile = traffic_generator.generate_profile(intent)
slo_targets = traffic_generator.generate_slo_targets(intent)
```

---

### Step 3: Benchmark Query (Database)

**File**: [src/planner/knowledge_base/benchmarks.py](../src/planner/knowledge_base/benchmarks.py)

The `BenchmarkRepository` queries the database for all (model, GPU, tensor_parallel) configurations that meet SLO targets for the traffic profile.

**Input**: Traffic profile and SLO targets

**Output**: List of `BenchmarkData` objects with latency/throughput metrics

**Key Query**: `find_configurations_meeting_slo()`
- Matches exact traffic profile (prompt_tokens, output_tokens)
- Filters by p95 SLO targets (TTFT, ITL, E2E)
- Uses window functions to select highest QPS per configuration
- Returns one benchmark per unique (model, hardware, hardware_count) combination

**Data Source**: `exported_summaries` table in the database (loaded from [data/benchmarks/performance/benchmarks_BLIS.json](../data/benchmarks/performance/benchmarks_BLIS.json))

**Near-Miss Tolerance**: When `include_near_miss=True`, SLO thresholds are relaxed by 20% to include configurations that nearly meet targets.

---

### Step 4: Capacity Planning and Scoring

**File**: [src/planner/recommendation/config_finder.py](../src/planner/recommendation/config_finder.py)

The `ConfigFinder.plan_all_capacities()` method processes each benchmark configuration and calculates three scores.

**Input**:
- Traffic profile and SLO targets
- Deployment intent
- ScoringEngine (for quality scoring)

**Output**: List of `DeploymentRecommendation` objects with `ConfigurationScores`

**For each benchmark configuration**:

1. **Calculate replicas** needed to handle expected QPS (with 20% headroom)
2. **Build GPU config** (tensor_parallel from benchmark, replicas from QPS calculation)
3. **Calculate cost** from GPU type and count
4. **Score on 3 dimensions** (see below)
5. **Create DeploymentRecommendation** with scores attached

**Key Function**:
```python
all_configs = capacity_planner.plan_all_capacities(
    traffic_profile=traffic_profile,
    slo_targets=slo_targets,
    intent=intent,
    model_evaluator=model_evaluator,
    include_near_miss=True,
)
```

---

### Step 5: Multi-Criteria Scoring

**Files**:
- [src/planner/recommendation/scorer.py](../src/planner/recommendation/scorer.py) - Calculates 3 scores
- [src/planner/recommendation/quality/usecase_scorer.py](../src/planner/recommendation/quality/usecase_scorer.py) - Benchmark-based quality scoring

#### 5.1 Accuracy Score (0-100)

**Primary Source**: Use-case specific quality scores from Artificial Analysis benchmarks

**Data Files**: [data/benchmarks/accuracy/weighted_scores/*.csv](../data/benchmarks/accuracy/weighted_scores/)

The `UseCaseQualityScorer` loads pre-calculated weighted scores for each use case. Each use case has different benchmark weights (e.g., code_completion weights LiveCodeBench 35%, SciCode 30%).

**Fallback**: If model not found in benchmark data, uses `ModelEvaluator.score_model()` which considers:
- Use case quality match (50 points)
- Domain specialization (15 points)
- Latency-appropriate model size (20 points)
- Budget-appropriate model size (10 points)
- Context length (5 points)

**Key Function**:
```python
accuracy_score = model_evaluator.score_model(model, intent)
```

#### 5.2 Price Score (0-100)

**Formula**: `100 * (max_cost - config_cost) / (max_cost - min_cost)`

Normalized inverse cost across all viable configurations. Cheapest = 100, most expensive = 0.

**Key Function**:
```python
price_score = scorer.score_price(cost_per_month, min_cost, max_cost)
```

#### 5.3 Latency Score (0-100)

Based on ratio of predicted latency to SLO target (worst metric determines status):

| Ratio | Score | SLO Status |
|-------|-------|------------|
| ≤ 1.0 | 90-100 | `compliant` (bonus for headroom) |
| 1.0-1.2 | 70-89 | `near_miss` |
| > 1.2 | 0-69 | `exceeds` |

**Key Function**:
```python
latency_score, slo_status = scorer.score_latency(
    predicted_ttft, predicted_itl, predicted_e2e,
    target_ttft, target_itl, target_e2e
)
```

#### 5.4 Balanced Score

Weighted composite of all three scores:

```python
balanced_score = (
    accuracy_score * 0.45 +
    price_score * 0.45 +
    latency_score * 0.10
)
```

Custom weights can be provided via API (0-10 scale, normalized to percentages).

---

### Step 6: Ranking and Filtering

**File**: [src/planner/recommendation/analyzer.py](../src/planner/recommendation/analyzer.py)

The `Analyzer` generates 4 ranked lists from scored configurations.

**Input**: List of scored DeploymentRecommendations, optional filters

**Output**: Dict with 4 keys, each containing top 10 configurations

**Filters Applied**:
- `min_accuracy`: Exclude configs with accuracy < threshold
- `max_cost`: Exclude configs with monthly cost > ceiling

**4 Ranked Views**:
| View | Sorted By |
|------|-----------|
| `best_accuracy` | Accuracy score (descending) |
| `lowest_cost` | Price score (descending) |
| `lowest_latency` | Latency score (descending) |
| `balanced` | Weighted composite score (descending) |

**Key Function**:
```python
ranked_lists = ranking_service.generate_ranked_lists(
    configurations=all_configs,
    min_accuracy=70,
    max_cost=5000,
    top_n=10,
    weights={"accuracy": 4.5, "price": 4.5, "latency": 1}
)
```

---

### Step 7: Response Generation

**File**: [src/planner/orchestration/workflow.py](../src/planner/orchestration/workflow.py)

The `RecommendationWorkflow` orchestrates all steps and returns the appropriate response.

**For `/api/v1/generate-recommendations`**:
- Returns `RankedRecommendations` with all 4 ranked lists
- Includes specification (intent, traffic_profile, slo_targets)
- Reports total configs evaluated and configs after filters

---

## Data Files Summary

| File | Description | Used By |
|------|-------------|---------|
| [src/planner/data/configuration/usecase_slo_workload.json](../src/planner/data/configuration/usecase_slo_workload.json) | 9 use case definitions (traffic profiles, SLO ranges, workload params) | UseCaseRepository, TrafficProfileGenerator |
| [src/planner/data/configuration/model_catalog.json](../src/planner/data/configuration/model_catalog.json) | 47 curated models with metadata | ModelCatalog |
| [src/planner/data/performance/benchmarks_BLIS.json](../src/planner/data/performance/benchmarks_BLIS.json) | Latency benchmarks (loaded to database) | BenchmarkRepository |
| [src/quality_scoring/data/](../src/quality_scoring/data/) | Model quality scores (Arena + Artificial Analysis) | ScoringEngine |

---

## Key Classes and Their Responsibilities

| Class | File | Responsibility |
|-------|------|----------------|
| `RecommendationWorkflow` | orchestration/workflow.py | Orchestrate end-to-end flow |
| `IntentExtractor` | intent_extraction/extractor.py | Parse user message to intent |
| `TrafficProfileGenerator` | specification/traffic_profile.py | Generate traffic profile and SLO targets |
| `BenchmarkRepository` | knowledge_base/benchmarks.py | Query database for benchmarks |
| `ConfigFinder` | recommendation/config_finder.py | Find viable configs, calculate scores |
| `Scorer` | recommendation/scorer.py | Calculate 3 scores |
| `ScoringEngine` | quality_scoring/engine.py | Dual-source quality scores (Arena + AA) |
| `Analyzer` | recommendation/analyzer.py | Filter and sort into 4 ranked lists |
| `ModelCatalog` | knowledge_base/model_catalog.py | Model metadata and GPU pricing |

---

## Sequence Diagram

```
Full pipeline (each stage has its own API endpoint):

POST /api/v1/extract-intent
     │  User message → DeploymentIntent (LLM-powered)
     ▼
POST /api/v1/generate-specification
     │  DeploymentIntent → DeploymentSpecification
     │    ├── TrafficProfileGenerator.generate_profile()
     │    │         └── UseCaseRepository (usecase_slo_workload.json)
     │    ├── TrafficProfileGenerator.generate_slo_targets()
     │    └── Quality weights + priorities from config files
     ▼
User reviews and edits specification (UI Spec Editor)
     ▼
POST /api/v1/generate-recommendations
     │  DeploymentSpecification → RankedRecommendations
     │
     │  ┌─────────────────────────────────────────────┐
     │  │ ConfigFinder.find_configurations()           │
     │  │   ├── BenchmarkRepository                    │
     │  │   │     .find_configurations_meeting_slo()   │
     │  │   │                                          │
     │  │   ├── For each config:                       │
     │  │   │     ├── ScoringEngine.score() (quality)  │
     │  │   │     ├── Scorer.score_latency()           │
     │  │   │     └── Scorer.score_price()             │
     │  │   │                                          │
     │  │   └── Scorer.score_balanced()                │
     │  └─────────────────────────────────────────────┘
     │
     │  ┌─────────────────────────────────────────────┐
     │  │ Analyzer.generate_ranked_lists()             │
     │  │   ├── Apply filters (min_quality, max_cost)  │
     │  │   └── Sort into 4 views (balanced, quality,  │
     │  │       cost, latency)                         │
     │  └─────────────────────────────────────────────┘
     ▼
User selects a configuration
     ▼
POST /api/v1/generate-deployment
     │  DeploymentConfiguration → DeploymentBundle (YAML files)
     ▼
POST /api/v1/deploy-bundle-to-cluster
     │  DeploymentBundle → deployed to Kubernetes
     ▼
Done
```

---

## Example Request/Response

**Request**:
```json
{
  "message": "I need a chatbot for 1000 users with low latency",
  "min_accuracy": 50,
  "max_cost": 5000,
  "include_near_miss": true
}
```

**Extracted Intent**:
- use_case: chatbot_conversational
- user_count: 1000
- latency_requirement: high
- experience_class: conversational

**Generated Profile**:
- prompt_tokens: 512
- output_tokens: 256
- expected_qps: 9

**SLO Targets (p95)**:
- TTFT: 150ms
- ITL: 25ms
- E2E: 7000ms

**Example Configuration Scores**:
```
Granite 3.1 8B on 1x H100:
  Accuracy: 72  (from weighted_scores CSV)
  Price: 85     (relatively inexpensive)
  Latency: 95   (well under SLO)
  Balanced: 80.15
```

---

## Future Enhancements

1. **Parametric performance models**: Train regression models to predict latency for arbitrary traffic profiles (not just the 4 GuideLLM profiles)

2. **Multi-QPS benchmark selection**: Currently selects highest QPS meeting SLO; future versions may offer QPS-specific recommendations

3. **GPU availability scoring**: Factor in procurement constraints and lead times

4. **Feedback loop**: Use actual deployment performance to improve future recommendations
