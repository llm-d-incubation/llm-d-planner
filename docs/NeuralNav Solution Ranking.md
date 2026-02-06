## Ranking and Presenting NeuralNav Recommendations

### Overview

NeuralNav recommends optimal LLM and GPU configurations that satisfy users' specific use cases and performance requirements. This document describes the implemented multi-criteria ranking system that scores and presents configuration options based on four optimization criteria.

**Key Design Principle**: Configuration-first approach. The system queries all (model, GPU) configurations meeting SLO targets, then scores each on four criteria. This ensures no viable configurations are missed due to model pre-filtering.

**Priority Hierarchy:**
- **Primary Criteria**: Accuracy and Price (40% weight each in balanced score)
- **Secondary Criteria**: Latency Performance and Operational Complexity (10% weight each)
- **Note**: All recommendations are pre-filtered to meet SLO requirements, so latency becomes a secondary differentiator for headroom beyond compliance

***

### The Four Optimization Criteria

#### 1. Accuracy (Model Quality)

**Definition**: How well the model fits the intended use case.

**Scoring Method**: The `score_model()` function in `ModelEvaluator` calculates accuracy (0-100) based on:
- **Use case match (40 points)**: Model's `recommended_for` field matches the deployment use case
- **Domain specialization (20 points)**: Overlap between model's domains and intent domains
- **Latency appropriateness (20 points)**: Model size vs. latency requirement (smaller models for low-latency needs)
- **Budget fit (10 points)**: Model size vs. budget constraint
- **Context length (10 points)**: Bonus for long-context models in summarization/QA use cases

**Catalog Integration**: Models in the catalog receive accuracy scores from `score_model()`. Models not in the catalog (but present in benchmarks) receive accuracy score = 0 but are still included as valid options.

***

#### 2. Price (Total Cost of Ownership)

**Definition**: The monthly cost of deploying the configuration.

**Scoring Method**: Normalized inverse cost formula:
```
price_score = 100 * (max_cost - config_cost) / (max_cost - min_cost)
```

Where `min_cost` and `max_cost` are computed across all viable configurations for the given request.

**Cost Calculation**: Based on GPU hourly rates from the model catalog, multiplied by GPU count and 730 hours/month. GPU types support aliases (e.g., "A100-40" and "NVIDIA-A100-40GB" resolve to the same GPU with the same pricing).

***

#### 3. Latency Performance

**Definition**: SLO compliance and headroom, measured via three p95 metrics:
- **TTFT** (Time to First Token) - Critical for interactive applications
- **ITL** (Inter-Token Latency) - Important for streaming responses
- **E2E** (End-to-End Latency) - Overall response time

**Scoring Method**: Based on ratio of predicted latency to SLO target (worst metric determines status):

| Ratio | Score Range | SLO Status |
|-------|-------------|------------|
| ≤ 1.0 | 90-100 | `compliant` (bonus for headroom) |
| 1.0-1.2 | 70-89 | `near_miss` |
| > 1.2 | 0-69 | `exceeds` |

**Role in Ranking**: Since all configurations already meet SLO targets, latency scoring rewards additional headroom (e.g., 120ms TTFT vs. 150ms target earns a higher score).

***

#### 4. Operational Complexity

**Definition**: The difficulty of deploying and managing the configuration.

**Scoring Method**: Based on total GPU count:

| GPU Count | Score |
|-----------|-------|
| 1 | 100 |
| 2 | 90 |
| 3 | 82 |
| 4 | 75 |
| 5 | 70 |
| 6 | 65 |
| 7 | 62 |
| 8 | 60 |
| >8 | Linear decay (60 - 2*(n-8)), minimum 40 |

**Rationale**: Fewer GPUs = simpler networking, easier troubleshooting, lower coordination overhead. A 2x H100 deployment may be preferred over 8x L4 even at higher cost due to reduced operational burden.

***

### Configuration Variants

#### Quantization Options

Quantization (FP16, FP8, INT8, INT4) creates distinct benchmark entries for each model-GPU combination. These appear as separate configurations with their own scores.

**Impact on Scores**:
- **Price**: Reduced memory → fewer GPUs needed → lower cost
- **Latency**: Faster inference from smaller compute requirements
- **Complexity**: Smaller memory footprint enables single-GPU deployments
- **Accuracy**: Generally preserved for FP8; slight reduction for INT4

Quantized variants are identified by model ID suffixes (e.g., `-fp8-dynamic`, `-quantized.w4a16`).

***

### Balanced Score Calculation

The balanced score combines all four criteria with configurable weights:

```python
balanced_score = (
    accuracy_score * weight_accuracy +
    price_score * weight_price +
    latency_score * weight_latency +
    complexity_score * weight_complexity
)
```

**Default Weights**:
- Accuracy: 40%
- Price: 40%
- Latency: 10%
- Complexity: 10%

**Custom Weights**: The API accepts a `weights` parameter (0-10 scale per criterion) that normalizes to percentages. This allows users to adjust priorities without changing code.

***

### Filtering and Ranking

#### Hard Filters (Applied Before Ranking)

- **SLO Compliance**: Only configurations meeting p95 targets (or within near-miss tolerance)
- **Minimum Accuracy**: Optional threshold (e.g., ≥70) to exclude low-quality models
- **Maximum Cost**: Optional ceiling (e.g., ≤$500/month) for budget constraints

#### Five Ranked Views

Each view presents up to 10 configurations sorted by the relevant criterion:

1. **Best Accuracy**: Sorted by accuracy score (descending)
2. **Lowest Cost**: Sorted by price score (descending)
3. **Lowest Latency**: Sorted by latency score (descending)
4. **Simplest**: Sorted by complexity score (descending)
5. **Balanced**: Sorted by weighted composite score (descending)

#### Near-SLO Configurations

Configurations within 20% of SLO targets are included with clear warnings:
- Marked with `slo_status: "near_miss"`
- Display ⚠️ indicators in UI
- Useful for finding cost savings when exact SLO compliance isn't critical

***

### Implementation Architecture

#### Key Components

| Component | Location | Responsibility |
|-----------|----------|----------------|
| `Scorer` | `backend/src/recommendation/scorer.py` | Calculate 4 scores (0-100 scale) |
| `UseCaseQualityScorer` | `backend/src/recommendation/quality/usecase_scorer.py` | Accuracy scoring based on use case fit |
| `ConfigFinder.plan_all_capacities()` | `backend/src/recommendation/config_finder.py` | Query benchmarks, score all configs |
| `Analyzer` | `backend/src/recommendation/analyzer.py` | Filter and sort into 5 ranked lists |
| `RecommendationWorkflow` | `backend/src/orchestration/workflow.py` | Orchestrate end-to-end flow |

#### Data Flow

```
User Request
    ↓
Intent Extraction → Traffic Profile → SLO Targets
    ↓
plan_all_capacities()
    ├── Query PostgreSQL: find_configurations_meeting_slo()
    ├── For each (model, GPU) config:
    │   ├── Look up model in catalog
    │   ├── Calculate accuracy via score_model()
    │   ├── Calculate latency score and SLO status
    │   ├── Calculate complexity score
    │   └── Build DeploymentRecommendation with scores
    └── Calculate price scores (after min/max known)
    ↓
RankingService.generate_ranked_lists()
    ├── Apply filters (min_accuracy, max_cost)
    ├── Recalculate balanced scores if custom weights
    └── Sort into 5 ranked views
    ↓
RankedRecommendationsResponse
```

#### API Endpoints

- **`POST /api/v1/recommend`**: Returns single best recommendation (balanced score)
- **`POST /api/v1/ranked-recommend-from-spec`**: Returns all 5 ranked lists with filter support

***

### Example Output

```
🎯 Lowest Cost (Top 5):

1. ✅ Llama-3.1-8B-FP8 on 1x L4     - $584/month  (Scores: A=60, P=100, L=92, C=100)
2. ✅ Mistral-7B on 1x L4           - $584/month  (Scores: A=55, P=100, L=90, C=100)
3. ✅ Qwen-2.5-7B-FP8 on 1x L4      - $584/month  (Scores: A=55, P=100, L=91, C=100)
4. ⚠️ Llama-3.1-70B-FP8 on 2x A100  - $2,555/month (Scores: A=85, P=72, L=78, C=90) [TTFT +15% over SLO]
5. ✅ Llama-3.1-70B on 2x H100      - $6,205/month (Scores: A=85, P=45, L=95, C=90)

🏆 Best Accuracy (Top 5):

1. ✅ Llama-3.1-70B on 2x H100      - $6,205/month (Scores: A=85, P=45, L=95, C=90)
2. ⚠️ Llama-3.1-70B-FP8 on 2x A100  - $2,555/month (Scores: A=85, P=72, L=78, C=90) [Near-miss]
3. ✅ Mixtral-8x7B on 4x A100       - $5,110/month (Scores: A=78, P=50, L=92, C=75)
4. ✅ Llama-3.1-8B on 1x L4         - $584/month   (Scores: A=60, P=100, L=92, C=100)
...
```

***

### Possible Future Enhancements

The following capabilities are documented for future implementation:

- **Benchmark-based accuracy scores**: Replace model size tiers with actual benchmark results (MMLU, HumanEval, MT-Bench)
- **Pareto frontier visualization**: Graphical 2D/3D charts showing optimal tradeoff regions
- **GPU availability scoring**: Factor in procurement constraints and lead times
- **Multi-cost model support**: Compare cloud vs. on-premise vs. existing infrastructure pricing
