# NeuralNav Recommendation Flow

This document describes the end-to-end recommendation flow implemented in the NeuralNav codebase. It traces data from user input through to ranked deployment recommendations.

## Overview

The recommendation flow follows a **configuration-first approach**: rather than pre-filtering models, the system queries all (model, GPU) configurations that meet SLO targets from the benchmark database, then scores each on four criteria.

```
User Message
    ↓
Intent Extraction (LLM)
    ↓
Traffic Profile + SLO Targets (from templates)
    ↓
Query PostgreSQL for SLO-compliant configurations
    ↓
Score each configuration (accuracy, price, latency, complexity)
    ↓
Generate 5 ranked lists
    ↓
Return best recommendation or all ranked lists
```

---

## API Endpoints

| Endpoint | Purpose | Returns |
|----------|---------|---------|
| `POST /api/recommend` | Simple recommendation | Single best config with YAML |
| `POST /api/ranked-recommend` | Multi-criteria ranking | 5 ranked lists (10 configs each) |
| `POST /api/re-recommend` | Re-run with edited specs | Single best config |
| `POST /api/regenerate-and-recommend` | Regenerate profile from intent | Single best config |

**Entry Point**: [backend/src/api/routes.py](../backend/src/api/routes.py)

---

## Step-by-Step Flow

### Step 1: Intent Extraction

**File**: [backend/src/context_intent/extractor.py](../backend/src/context_intent/extractor.py)

The `IntentExtractor` uses an LLM (Ollama qwen2.5:7b) to parse the user's natural language request into structured deployment intent.

**Input**: User message (e.g., "I need a chatbot for 1000 users, low latency is critical")

**Output**: `DeploymentIntent` object containing:
- `use_case`: Mapped to one of 9 supported use cases
- `user_count`: Number of concurrent users
- `latency_requirement`: very_high, high, medium, low
- `budget_constraint`: strict, moderate, flexible, none
- `domain_specialization`: Optional list of domains
- `experience_class`: instant, conversational, interactive, deferred, batch

**Key Function**:
```python
intent = intent_extractor.extract_intent(user_message, conversation_history)
intent = intent_extractor.infer_missing_fields(intent)
```

---

### Step 2: Traffic Profile Generation

**File**: [backend/src/context_intent/traffic_profile.py](../backend/src/context_intent/traffic_profile.py)

The `TrafficProfileGenerator` maps the use case to a GuideLLM traffic profile and calculates SLO targets.

**Input**: `DeploymentIntent`

**Output**:
- `TrafficProfile`: prompt_tokens, output_tokens, expected_qps
- `SLOTargets`: ttft_p95_target_ms, itl_p95_target_ms, e2e_p95_target_ms

**Data Source**: [data/slo_templates.json](../data/slo_templates.json)

**Traffic Profiles** (aligned with GuideLLM):
| Use Case | Prompt Tokens | Output Tokens |
|----------|--------------|---------------|
| chatbot_conversational | 512 | 256 |
| code_completion | 512 | 256 |
| code_generation_detailed | 1024 | 1024 |
| translation | 1024 | 1024 |
| content_generation | 512 | 256 |
| summarization_short | 512 | 256 |
| document_analysis_rag | 4096 | 512 |
| long_document_summarization | 10240 | 1536 |
| research_legal_analysis | 10240 | 1536 |

**Key Functions**:
```python
traffic_profile = traffic_generator.generate_profile(intent)
slo_targets = traffic_generator.generate_slo_targets(intent)
```

---

### Step 3: Benchmark Query (PostgreSQL)

**File**: [backend/src/knowledge_base/benchmarks.py](../backend/src/knowledge_base/benchmarks.py)

The `BenchmarkRepository` queries PostgreSQL for all (model, GPU, tensor_parallel) configurations that meet SLO targets for the traffic profile.

**Input**: Traffic profile and SLO targets

**Output**: List of `BenchmarkData` objects with latency/throughput metrics

**Key Query**: `find_configurations_meeting_slo()`
- Matches exact traffic profile (prompt_tokens, output_tokens)
- Filters by p95 SLO targets (TTFT, ITL, E2E)
- Uses window functions to select highest QPS per configuration
- Returns one benchmark per unique (model, hardware, hardware_count) combination

**Data Source**: `exported_summaries` table in PostgreSQL (loaded from [data/benchmarks_BLIS.json](../data/benchmarks_BLIS.json))

**Near-Miss Tolerance**: When `include_near_miss=True`, SLO thresholds are relaxed by 20% to include configurations that nearly meet targets.

---

### Step 4: Capacity Planning and Scoring

**File**: [backend/src/recommendation/capacity_planner.py](../backend/src/recommendation/capacity_planner.py)

The `CapacityPlanner.plan_all_capacities()` method processes each benchmark configuration and calculates four scores.

**Input**:
- Traffic profile and SLO targets
- Deployment intent
- Model evaluator (for accuracy scoring)

**Output**: List of `DeploymentRecommendation` objects with `ConfigurationScores`

**For each benchmark configuration**:

1. **Calculate replicas** needed to handle expected QPS (with 20% headroom)
2. **Build GPU config** (tensor_parallel from benchmark, replicas from QPS calculation)
3. **Calculate cost** from GPU type and count
4. **Score on 4 dimensions** (see below)
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
- [backend/src/recommendation/solution_scorer.py](../backend/src/recommendation/solution_scorer.py) - Calculates 4 scores
- [backend/src/recommendation/model_evaluator.py](../backend/src/recommendation/model_evaluator.py) - Accuracy scoring
- [backend/src/recommendation/usecase_quality_scorer.py](../backend/src/recommendation/usecase_quality_scorer.py) - Benchmark-based quality

#### 5.1 Accuracy Score (0-100)

**Primary Source**: Use-case specific quality scores from Artificial Analysis benchmarks

**Data Files**: [data/business_context/use_case/weighted_scores/*.csv](../data/business_context/use_case/weighted_scores/)

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

#### 5.4 Complexity Score (0-100)

Based on total GPU count:

| GPU Count | Score |
|-----------|-------|
| 1 | 100 |
| 2 | 90 |
| 4 | 75 |
| 8 | 60 |
| >8 | Linear decay (min 40) |

**Key Function**:
```python
complexity_score = scorer.score_complexity(gpu_count)
```

#### 5.5 Balanced Score

Weighted composite of all four scores:

```python
balanced_score = (
    accuracy_score * 0.40 +
    price_score * 0.40 +
    latency_score * 0.10 +
    complexity_score * 0.10
)
```

Custom weights can be provided via API (0-10 scale, normalized to percentages).

---

### Step 6: Ranking and Filtering

**File**: [backend/src/recommendation/ranking_service.py](../backend/src/recommendation/ranking_service.py)

The `RankingService` generates 5 ranked lists from scored configurations.

**Input**: List of scored DeploymentRecommendations, optional filters

**Output**: Dict with 5 keys, each containing top 10 configurations

**Filters Applied**:
- `min_accuracy`: Exclude configs with accuracy < threshold
- `max_cost`: Exclude configs with monthly cost > ceiling

**5 Ranked Views**:
| View | Sorted By |
|------|-----------|
| `best_accuracy` | Accuracy score (descending) |
| `lowest_cost` | Price score (descending) |
| `lowest_latency` | Latency score (descending) |
| `simplest` | Complexity score (descending) |
| `balanced` | Weighted composite score (descending) |

**Key Function**:
```python
ranked_lists = ranking_service.generate_ranked_lists(
    configurations=all_configs,
    min_accuracy=70,
    max_cost=5000,
    top_n=10,
    weights={"accuracy": 4, "price": 4, "latency": 1, "complexity": 1}
)
```

---

### Step 7: Response Generation

**File**: [backend/src/orchestration/workflow.py](../backend/src/orchestration/workflow.py)

The `RecommendationWorkflow` orchestrates all steps and returns the appropriate response.

**For `/api/recommend`**:
- Returns single best configuration (highest balanced score)
- Includes top 3 alternatives
- Auto-generates YAML files

**For `/api/ranked-recommend`**:
- Returns `RankedRecommendationsResponse` with all 5 ranked lists
- Includes specification (intent, traffic_profile, slo_targets)
- Reports total configs evaluated and configs after filters

---

## Data Files Summary

| File | Description | Used By |
|------|-------------|---------|
| [data/slo_templates.json](../data/slo_templates.json) | 9 use case templates with SLO targets | TrafficProfileGenerator |
| [data/model_catalog.json](../data/model_catalog.json) | 47 curated models with metadata | ModelCatalog, ModelEvaluator |
| [data/benchmarks_BLIS.json](../data/benchmarks_BLIS.json) | Latency benchmarks (loaded to PostgreSQL) | BenchmarkRepository |
| [data/business_context/use_case/weighted_scores/*.csv](../data/business_context/use_case/weighted_scores/) | 9 use-case quality score files | UseCaseQualityScorer |
| [data/business_context/use_case/configs/usecase_weights.json](../data/business_context/use_case/configs/usecase_weights.json) | Benchmark weight definitions per use case | Documentation |

---

## Key Classes and Their Responsibilities

| Class | File | Responsibility |
|-------|------|----------------|
| `RecommendationWorkflow` | orchestration/workflow.py | Orchestrate end-to-end flow |
| `IntentExtractor` | context_intent/extractor.py | Parse user message to intent |
| `TrafficProfileGenerator` | context_intent/traffic_profile.py | Generate traffic profile and SLO targets |
| `BenchmarkRepository` | knowledge_base/benchmarks.py | Query PostgreSQL for benchmarks |
| `CapacityPlanner` | recommendation/capacity_planner.py | Find viable configs, calculate scores |
| `SolutionScorer` | recommendation/solution_scorer.py | Calculate 4 scores |
| `ModelEvaluator` | recommendation/model_evaluator.py | Accuracy scoring (use-case fit) |
| `UseCaseQualityScorer` | recommendation/usecase_quality_scorer.py | Benchmark-based quality scores |
| `RankingService` | recommendation/ranking_service.py | Filter and sort into 5 ranked lists |
| `ModelCatalog` | knowledge_base/model_catalog.py | Model metadata and GPU pricing |

---

## Sequence Diagram

```
User Request
     │
     ▼
┌─────────────────────┐
│  API Route Handler  │ (/api/recommend or /api/ranked-recommend)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ RecommendationWorkflow │
│ .generate_recommendation() or │
│ .generate_ranked_recommendations() │
└──────────┬──────────┘
           │
           ├──► IntentExtractor.extract_intent()
           │         └──► Ollama LLM (qwen2.5:7b)
           │
           ├──► TrafficProfileGenerator.generate_profile()
           │         └──► SLOTemplateRepository (slo_templates.json)
           │
           ├──► TrafficProfileGenerator.generate_slo_targets()
           │
           ▼
┌─────────────────────┐
│  CapacityPlanner    │
│  .plan_all_capacities() │
└──────────┬──────────┘
           │
           ├──► BenchmarkRepository.find_configurations_meeting_slo()
           │         └──► PostgreSQL (exported_summaries table)
           │
           ├──► For each config:
           │       ├──► ModelCatalog.get_model() (lookup metadata)
           │       ├──► ModelEvaluator.score_model() (accuracy)
           │       │         └──► UseCaseQualityScorer (Artificial Analysis data)
           │       ├──► SolutionScorer.score_latency()
           │       ├──► SolutionScorer.score_complexity()
           │       └──► ModelCatalog.calculate_gpu_cost()
           │
           └──► SolutionScorer.score_price() (after min/max known)
                SolutionScorer.score_balanced()
           │
           ▼
┌─────────────────────┐
│   RankingService    │
│ .generate_ranked_lists() │
└──────────┬──────────┘
           │
           ├──► Apply filters (min_accuracy, max_cost)
           ├──► Recalculate balanced scores (if custom weights)
           └──► Sort into 5 ranked views
           │
           ▼
┌─────────────────────┐
│     API Response    │
│ (DeploymentRecommendation or │
│  RankedRecommendationsResponse) │
└─────────────────────┘
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
  Complexity: 100 (single GPU)
  Balanced: 82.5
```

---

## Future Enhancements

1. **Parametric performance models**: Train regression models to predict latency for arbitrary traffic profiles (not just the 4 GuideLLM profiles)

2. **Multi-QPS benchmark selection**: Currently selects highest QPS meeting SLO; future versions may offer QPS-specific recommendations

3. **GPU availability scoring**: Factor in procurement constraints and lead times

4. **Feedback loop**: Use actual deployment performance to improve future recommendations
