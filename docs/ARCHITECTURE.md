# System-Level Architecture for NeuralNav

## Overview

This document defines the system architecture for **NeuralNav**, which streamlines the path from concept to production-ready endpoint for LLM deployments on Kubernetes.

## Project Context

**Vision**: A world where every organization adopts AI at scale, with a personalized AI stack optimized for its business needs.

**Mission**: Ensuring our customers fulfill their AI goals and maximize ROI of their GenAI investments through deep visibility and personalized recommendations.

**Problem Statement**: Deploying LLMs in production environments remains complex and error-prone:
- Application Developers struggle to translate business needs into infrastructure choices
- Cluster Admins face overprovisioning risks without guidance on aligning workloads to budgets and SLAs
- Trial-and-error is the default, extending time-to-production and inflating costs

## Development Phases

This architecture has been implemented across three phases:

**Phase 1: POC (Complete)**
Proof-of-concept implementation demonstrating end-to-end workflow with simplified data models:
- Core components implemented in Python
- Synthetic benchmark data (JSON files)
- Point estimates for traffic (average prompt/output length)
- p90 SLO metrics (TTFT, TPOT, E2E)
- Streamlit UI with conversational interface
- Full deployment automation with KServe/vLLM
- vLLM simulator for GPU-free development
- KIND cluster support with basic monitoring

**Phase 2: MVP (In Progress - PostgreSQL Migration)**
Production-grade implementation with robust data management:
- **PostgreSQL database** for all benchmark and configuration data
- **Traffic profile framework**: 4 GuideLLM benchmark configurations
  - (512→256), (1024→1024), (4096→512), (10240→1536)
- **Experience-driven SLOs**: Mapping use cases to latency targets
- **p95 percentiles** for more conservative SLO guarantees (changed from p90)
- **ITL (Inter-Token Latency)** terminology (changed from TPOT)
- **Exact traffic matching**: No fuzzy matching on token lengths
- **Pre-calculated E2E latency** from benchmark data
- Enhanced recommendation engine with SLO filtering

**Phase 3+: Future Options**
Advanced features for scale and optimization:
- Full statistical distributions for traffic profiles (not just point estimates)
- Multi-dimensional benchmarks (concurrency, batching, KV cache effects)
- Advanced what-if analysis and simulation (SimPy, Monte Carlo)
- Continuous learning from deployment outcomes
- Advanced observability and feedback loops
- Cost modeling for owned hardware and TCO analysis

## Solution Approach

NeuralNav follows a **4-stage workflow**:

1. **Intent & Specification** - Transform conversation into structured deployment specifications with traffic profiles and SLO targets
2. **Recommendation** - Generate ranked model + GPU configurations that meet SLO requirements
3. **Interactive Exploration** - Review, edit specifications, and explore what-if scenarios with cost/latency trade-offs
4. **Deployment & Monitoring** - Generate Kubernetes configs and deploy to production with health monitoring

---

## Architecture Components

NeuralNav is structured as a layered architecture with a presentation layer, four core engines, and a shared data layer:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          UI Layer (Presentation)                          │
│                   Streamlit (current) → React (future)                    │
│  Conversational Interface | Spec Editor | Visualizer | Dashboard | Test   │
└───────────────────────────────────────────────────────────────────────────┘
                            ↕ API Gateway (FastAPI)
┌───────────────────────────────────────────────────────────────────────────┐
│                            Core Engines                                   │
├──────────────────┬──────────────────┬──────────────────┬──────────────────┤
│     Intent &     │                  │                  │                  │
│  Specification   │  Recommendation  │    Deployment    │  Observability   │
│      Engine      │      Engine      │      Engine      │      Engine      │
│                  │                  │                  │                  │
│   Conversation   │  Specification   │  Recommendation  │ Deployed Service │
│        ↓         │        ↓         │        ↓         │        ↓         │
│    Structured    │      Ranked      │     Deployed     │   Monitoring +   │
│    Deployment    │   Model + GPU    │    Kubernetes    │   Performance    │
│  Specification   │ Recommendations  │     Service      │     Insights     │
│                  │                  │                  │                  │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┘
                            ↕ Data Access Layer
┌───────────────────────────────────────────────────────────────────────────┐
│                    Knowledge Base (Data Layer)                            │
│        PostgreSQL: Benchmarks, Deployment Outcomes                        │
│        JSON: SLO Templates, Model Catalog, Hardware Profiles              │
└───────────────────────────────────────────────────────────────────────────┘
```

### UI Layer (Presentation)

**Purpose**: Provide unified user interface across all system capabilities

**Technology**:
- Streamlit (current implementation - rapid development)
- React + WebSocket backend (future for production scale)

**Responsibilities**:
- **Conversational Interface**: Natural language chat for capturing user intent
- **Specification Editor**: Review and modify auto-generated deployment specifications
- **Recommendation Visualizer**: Display model/GPU options with trade-off analysis and what-if scenarios
- **Deployment Dashboard**: Trigger and manage Kubernetes deployments
- **Monitoring & Testing UI**: View deployment status, health checks, and inference testing

**Key Features**:
- Specification review & editing with inline controls
- Side-by-side scenario comparison (future)
- What-if analysis with cost/latency/quality trade-offs (future)
- Real-time deployment status monitoring
- Interactive inference testing

**Code Location**: `ui/` (Streamlit application)

---

### Engine 1: Intent & Specification Engine

**Purpose**: Transform natural language conversation into structured, complete deployment specifications

**Input**: User conversation (use case, requirements, constraints)
**Output**: `DeploymentIntent` specification with traffic profile, SLO targets, and constraints

**Responsibilities**:
1. **Conversation Management**: Multi-turn dialogue with context tracking
2. **Intent Extraction**: Parse user inputs into structured data (LLM-powered)
3. **Use Case Mapping**: Map user's description to standard use case templates
4. **Traffic Profile Determination**: Generate traffic characteristics (prompt_tokens, output_tokens, QPS)
5. **SLO Target Suggestion**: Recommend latency and throughput targets based on use case
6. **Specification Validation**: Ensure completeness before passing to Recommendation Engine

**Sub-Components**:
- **LLM Backend**: Powers conversational AI (Ollama with llama3.1:8b currently)
- **Intent Parser**: Structured extraction from natural language
- **Traffic Profile Generator**: Maps use case → traffic characteristics
- **Specification Builder**: Assembles complete `DeploymentIntent` object

**Data Dependencies**:
- `slo_templates.json`: Maps use cases to traffic profiles and SLO targets
- `model_catalog.json`: Available models for validation
- Historical deployment outcomes (future): Learn from actual traffic patterns

**Key Insight**: This engine produces the complete "what" (specification), while the Recommendation Engine determines the "how" (which model + GPU configuration).

**Code Location**: `backend/src/context_intent/`, `backend/src/llm/`

---

### Engine 2: Recommendation Engine

**Purpose**: Find optimal model + GPU configurations that meet specification requirements

**Input**: `DeploymentIntent` specification (from Intent & Specification Engine)
**Output**: Ranked list of `DeploymentRecommendation` options

**Responsibilities**:
1. **Benchmark Query**: Find benchmark data matching traffic profile
2. **SLO Compliance Filtering**: Eliminate configurations that violate latency/throughput targets
3. **Capacity Calculation**: Determine GPU count and deployment topology
4. **Cost Estimation**: Calculate hourly/monthly costs
5. **Ranking by Priority**: Sort options by user preferences (cost vs latency vs quality)
6. **Alternative Filtering**: Remove dominated options that offer no meaningful value proposition
7. **Trade-off Analysis**: Explain why each option was recommended

**Sub-Components**:
- **Model Recommender**: Filter and rank models by task compatibility
- **Capacity Planner**: Calculate required GPU resources and deployment strategy
- **Cost Calculator**: Estimate operational costs
- **Ranker**: Sort recommendations by user priorities

**Recommendation Logic**:
```
1. Query benchmarks: (model, gpu_type, traffic_profile) → performance metrics
2. Filter by SLO compliance:
   - TTFT p95 ≤ target
   - ITL p95 ≤ target
   - E2E p95 ≤ target
3. Calculate replicas: required_QPS / benchmark_QPS
4. Estimate cost: gpu_count × hourly_rate
5. Rank by priority: cost-optimal vs latency-optimal vs balanced
6. Filter alternatives: Remove strictly dominated options that provide no distinct value
   - Always include different models (user choice)
   - Always include different GPU types (e.g., H100 vs H200)
   - Include if 50%+ throughput improvement for <30% cost increase
   - Include if 20%+ latency improvement for similar cost (≤20% increase)
   - Skip configurations with higher cost and same/worse performance
```

**Data Dependencies**:
- PostgreSQL benchmarks: Performance data for (model, GPU, traffic profile) combinations
- `model_catalog.json`: Available models and their capabilities
- Hardware profiles: GPU specifications and pricing
- Model accuracy benchmarks (future): Quality metrics for model selection

**Code Location**: `backend/src/recommendation/`

---

### Engine 3: Deployment Engine

**Purpose**: Generate and deploy production-ready Kubernetes configurations

**Input**: Selected `DeploymentRecommendation`
**Output**: Running KServe InferenceService on Kubernetes

**Responsibilities**:
1. **YAML Generation**: Create KServe/vLLM manifests from Jinja2 templates
2. **Configuration Validation**: Verify resource requests, security settings
3. **Kubernetes Deployment**: Apply manifests to cluster via K8s API
4. **Lifecycle Management**: Update, scale, or delete existing deployments
5. **External Access Setup**: Generate Ingress configurations (future)
6. **Observability Hooks**: Configure Prometheus ServiceMonitors

**Deployment Artifacts**:
- KServe InferenceService YAML
- vLLM runtime configuration
- Horizontal Pod Autoscaler (HPA) policies
- Prometheus ServiceMonitor for metrics collection
- Ingress/Route for external access (future)

**Deployment Modes**:
- **Simulator Mode**: GPU-free development with vLLM simulator (current default)
- **Real vLLM Mode**: Production deployment with actual GPU inference

**External Integrations**:
- **Kubernetes API**: Cluster deployment and management
- **KServe**: Model serving platform
- **Model Registry**: HuggingFace Hub for model downloads
- **Container Registry**: Docker image storage

**Code Location**: `backend/src/deployment/`

---

### Engine 4: Observability Engine

**Purpose**: Monitor deployed inference services and provide performance insights

**Input**: Deployed InferenceService metadata
**Output**: Health status, performance metrics, and optimization recommendations

**Current Capabilities** (Phase 2):
1. **Deployment Inventory**: Track which services are deployed
2. **Health Monitoring**: Pod status, service readiness via K8s API
3. **Inference Testing**: Send test prompts and verify responses
4. **Basic Metrics Display**: Show deployment status in UI

**Future Capabilities** (Phase 3+):
1. **SLO Compliance Tracking**: Monitor TTFT, ITL, E2E latency percentiles
2. **Resource Utilization**: GPU usage, memory, batch efficiency
3. **Cost Tracking**: Actual spend vs predicted, cost per 1k tokens
4. **Traffic Pattern Analysis**: Actual vs predicted prompt/output lengths
5. **Feedback Loop**: Store actual performance → Knowledge Base → improve future recommendations
6. **Alerting**: Proactive notifications on SLO violations

**Feedback Loop** (Future):
```
Observability Engine collects actual metrics
        ↓
Stores in Knowledge Base (deployment_outcomes table)
        ↓
Intent & Specification Engine queries historical data
        ↓
Refines use case mapping, traffic profiles, SLO targets
        ↓
Improved recommendations for current deployment or similar future deployments
```

**External Integrations**:
- **Kubernetes API**: Pod and service status
- **Prometheus + Grafana** (future): Advanced metrics and dashboards
- **vLLM metrics endpoint** (future): Detailed inference statistics
- **OpenTelemetry** (future): Distributed tracing

**Code Location**: Monitoring UI in `ui/`, observability module (future: `backend/src/observability/`)

---

### Infrastructure Components (Not Numbered)

These are critical infrastructure that enable the engines but are not domain components:

#### API Gateway
**Purpose**: Coordinate multi-engine workflows and provide REST API

**Technology**: FastAPI

**Responsibilities**:
- Expose HTTP endpoints for UI to call engines
- Orchestrate multi-step flows (conversation → recommendation → deployment)
- Manage workflow state (session-based currently, persistent store future)
- Handle authentication and authorization (future)

**Code Location**: `backend/src/api/`, `backend/src/orchestration/`

#### Knowledge Base (Data Layer)
**Purpose**: Store performance data, configuration, and deployment history

**Technology**: Hybrid storage approach
- **PostgreSQL**: High-volume query data (benchmarks, deployment outcomes)
- **JSON files**: Configuration as code (SLO templates, model catalog, hardware profiles)

**Collections**:
1. **Model Benchmarks** (PostgreSQL): Performance metrics for (model, GPU, traffic profile) combinations
2. **Use Case SLO Templates** (JSON): Default targets for 9 standard use cases
3. **Model Catalog** (JSON): 40 curated models with task compatibility metadata
4. **Hardware Profiles** (JSON/code): GPU specifications and pricing
5. **Cost Data** (JSON/code): Cloud pricing, operational cost models
6. **Deployment Outcomes** (PostgreSQL, future): Actual performance data for continuous learning

**Code Location**: `backend/src/knowledge_base/`, `data/` (JSON files), `scripts/schema.sql` (PostgreSQL schema)

---

---

## Detailed Engine Specifications

The following sections provide in-depth technical details for each engine.

---

### Intent & Specification Engine (Detailed)
**Purpose**: Extract structured intent from conversational input and generate comprehensive deployment specifications

**Technology Options**:
- **LangChain with structured output** - Built-in prompt engineering for extraction
- **Guardrails AI** - Schema validation and structured extraction from LLMs
- **Custom LLM + Pydantic validators** - Full control with type-safe extraction

**Key Functions**:
- Parse user inputs into structured data (task definition, expected load, priorities)
- Auto-generate traffic characteristics from high-level descriptions
- Map use cases to appropriate default SLOs
- Validate completeness before proceeding to recommendations
- Store conversation context for iterative refinement
- Present editable specification to user for review/modification

**Data Schema**:

*Note: The Phase 1 POC uses a simplified version of this schema with enum-based fields for easier LLM extraction (see `backend/src/context_intent/schema.py`). In future phases, we may want to support the user providing a more detailed intent that may include some of the following information:*

```python
class DeploymentIntent:
    # High-level intent (user-provided)
    task_type: str  # "chatbot", "summarization", "code-generation"
    use_case_description: str  # Natural language description
    subject_matter: Optional[str]  # May influence model selection

    # Workload characteristics (auto-inferred or user-specified)
    expected_qps: Optional[float]
    concurrent_users: Optional[int]

    # Traffic profile (auto-generated from use case, editable by user)
    prompt_length_avg: Optional[int]  # Phase 1: point estimates
    prompt_length_dist: Optional[Distribution]  # Phase 2: full distribution
    generation_length_avg: Optional[int]
    generation_length_dist: Optional[Distribution]
    concurrency_steady_state: Optional[int]
    concurrency_peak: Optional[int]
    burstiness_pattern: Optional[str]  # "steady", "moderate", "high", "diurnal"
    request_heterogeneity: Optional[dict]  # mix of short/long, streaming vs non-streaming

    # Constraints
    priority: str  # "accuracy", "latency", "cost", "balanced"
    budget: Optional[BudgetConstraint]

    # SLOs (auto-suggested from use case template, user-editable)
    target_ttft_p95_ms: Optional[float]  # Time to First Token
    target_itl_p95_ms: Optional[float]  # Inter-Token Latency
    target_e2e_latency_p95_ms: Optional[float]  # End-to-End latency
    target_throughput_rps: Optional[float]  # Requests per second
    quality_threshold: Optional[str]  # "standard", "high", "very_high"
    reliability_target: Optional[float]  # e.g., 99.9% uptime

    # Optional user preferences
    preferred_models: Optional[List[str]]  # User can specify models to consider
    available_gpus: Optional[List[str]]  # User can specify available GPU types
    gpu_constraints: Optional[dict]  # max count, tensor parallelism preferences

class BudgetConstraint:
    max_monthly_cost_usd: Optional[float]
    max_hourly_cost_usd: Optional[float]
    cost_per_1k_tokens: Optional[float]
```

**Use Case → SLO Mapping**:
The engine maintains templates for common use cases:
```python
USE_CASE_TEMPLATES = {
    "chatbot_conversational": {
        "traffic_profile": {
            "prompt_tokens": 512,
            "output_tokens": 256
        },
        "experience_class": "conversational",
        "default_slos": {
            "ttft_p95_ms": 150,
            "itl_p95_ms": 25,
            "e2e_latency_p95_ms": 7000
        }
    },
    "code_completion": {
        "traffic_profile": {
            "prompt_tokens": 512,
            "output_tokens": 256
        },
        "experience_class": "instant",
        "default_slos": {
            "ttft_p95_ms": 100,
            "itl_p95_ms": 20,
            "e2e_latency_p95_ms": 5000
        }
    },
    "document_analysis_rag": {
        "traffic_profile": {
            "prompt_tokens": 4096,
            "output_tokens": 512
        },
        "experience_class": "interactive",
        "default_slos": {
            "ttft_p95_ms": 600,
            "itl_p95_ms": 30,
            "e2e_latency_p95_ms": 18000
        }
    }
    # See data/slo_templates.json for all 9 use cases
}
```

---

### Recommendation Engine (Detailed)
**Purpose**: Find optimal model and GPU configurations that meet the deployment specification

**Input**: Complete `DeploymentIntent` specification from Intent & Specification Engine
- Use case and task type
- **Traffic profile** (prompt_tokens, output_tokens, QPS) - already determined
- SLO targets (TTFT, ITL, E2E latency, throughput)
- Constraints (budget, preferred GPUs, quality threshold)

**Output**: Ranked list of viable deployment options with cost/performance trade-offs

**Technology Options**:
- **Rule-based decision tree + LLM augmentation** - Hybrid approach with explainability (current)
- **Vector similarity search (FAISS/Pinecone) + retrieval** - Match to known deployment patterns (future)
- **Custom ML model** - Train on historical deployment data (future)

**Sub-Components**:

#### Model Recommender
Filters and ranks models based on task compatibility and user constraints.

**Inputs**:
- Deployment specification (from Intent & Specification Engine)
- Traffic profile (already in specification)
- SLO targets (already in specification)
- User preferences (optional preferred models, subject matter domain)

**Recommendation Logic**:
1. Filter models by task compatibility and subject matter domain
2. Consider user-specified model preferences (if any)
3. Apply curated model constraints (approved model catalog)
4. Evaluate performance vs accuracy trade-offs against SLOs
5. Query benchmark data for each candidate model
6. Rank by user priority (cost vs latency vs accuracy)
7. Generate explanation with rationale for each recommendation

**Outputs**:
- Ranked list of model recommendations with detailed rationale
- Performance projections for each model
- Trade-off analysis (accuracy vs cost vs latency)

#### Capacity Planner
Determines GPU requirements (type, count, configuration) for each model recommendation.

**Inputs**:
- Selected model (or candidate model for evaluation)
- Traffic profile
- SLO targets
- Available GPU types (optional user constraint)
- Budget constraints

**Capacity Calculation Logic**:
```python
def calculate_capacity(
    model: Model,
    gpu_type: GPUType,
    traffic_profile: TrafficProfile,
    slo_targets: SLOTargets
) -> CapacityPlan:
    """
    Returns:
    - num_gpus_required: int
    - deployment_strategy: "independent" | "tensor_parallel"
    - tensor_parallel_degree: Optional[int]
    - expected_throughput_rps: float
    - predicted_slo_compliance: Dict[str, bool]  # TTFT, TPOT, E2E
    - estimated_cost: CostEstimate

    Implementation Details (see backend/src/recommendation/capacity_planner.py):

    1. Query benchmarks: (model, gpu_type, tensor_parallel) -> performance metrics

    2. SLO compliance check:
       - Filter benchmarks where TTFT_p95 > slo.ttft_target (reject)
       - Filter benchmarks where ITL_p95 > slo.itl_target (reject)
       - Use pre-calculated E2E_p95 from benchmark data
       - Filter benchmarks where E2E_p95 > slo.e2e_target (reject)

    3. Replica calculation:
       - required_capacity = traffic_qps × 1.2  (20% headroom for safety)
       - replicas = ceil(required_capacity / benchmark.max_qps)

    4. GPU configuration:
       - gpu_count = tensor_parallel × replicas
       - deployment_strategy = "independent" if replicas > 1, else "single"

    5. Cost estimation:
       - hourly_cost = gpu_count × gpu_hourly_rate
       - monthly_cost = hourly_cost × 730 hours

    6. Select best configuration:
       - Sort viable configs by cost
       - Apply budget constraint (strict → cheapest, flexible → mid-range, none → best performance)
    """

    # Phase 2 enhancements:
    # - Consider auto-scaling (min/max GPU count)
    # Phase 3+ enhancements
    # - Optimize batch size for throughput/latency trade-off
    # - Advanced tensor parallelism optimization
    # - Queueing theory models for latency under load
```

**Benchmark Data Requirements**:
```json
{
  "model": "llama-3-70b",
  "gpu_type": "A100-80GB",
  "tensor_parallel_degree": 4,
  "metrics": {
    "throughput_tokens_per_sec": 3500,
    "ttft_p50_ms": 95,
    "ttft_p90_ms": 120,
    "ttft_p99_ms": 180,
    "tpot_p50_ms": 12,
    "tpot_p90_ms": 15,
    "tpot_p99_ms": 22,
    "memory_usage_gb": 280
  },
  "note": "Benchmarks collected using vLLM defaults (dynamic batching enabled)"
}
```

**Outputs**:
- GPU configuration (type, count, parallelism strategy)
- Deployment topology (independent replicas vs tensor parallelism)
- Cost estimates (hourly, monthly, per-1k-tokens)
- SLO compliance predictions

**Overall Recommendation Engine Outputs**:
- Ranked recommendations combining model + GPU configuration
- Example: "Llama-3-8B on 2x L4 GPUs: $X/month, meets TTFT<200ms, ITL<50ms"
- Detailed trade-off analysis for top 3-5 options

---

### Deployment Engine (Detailed)
**Purpose**: Generate production-ready Kubernetes configurations and manage deployment lifecycle

**Current Implementation**:

**Technology Stack**:
- **Jinja2 templating** - Generate KServe/vLLM YAML from templates
- **Kubernetes Python client** - Direct K8s API interaction for deployment
- **Python** - All deployment automation logic

**Outputs**:
- KServe InferenceService YAML
- vLLM runtime configuration
- Autoscaling policies (HPA/KEDA configs)
- Observability setup (Prometheus ServiceMonitors, Grafana dashboards)

**Integration Points**:
- KServe API (for model serving)
- OpenShift/Kubernetes cluster
- Model registry (HuggingFace Hub, custom registry)

**Deployment Flow**:
1. Validate configuration against cluster resources
2. Generate KServe InferenceService manifest
3. Generate vLLM runtime parameters
4. Create autoscaling policies
5. Deploy to cluster
6. Configure observability hooks
7. Return deployment ID and monitoring links

**External Access Patterns (Phase 2 Production Requirement)**:

The POC uses kubectl port-forward for testing individual deployments. Production deployments require permanent external access to support multiple simultaneous services (e.g., different departments with different use cases).

**Recommended Approaches**:

1. **Ingress (Recommended for Production)**
   - Single external endpoint with path-based or host-based routing
   - Centralized TLS, authentication, and rate limiting
   - Scales to hundreds of services without port conflicts
   - Example patterns:
     - Path-based: `https://inference.mycompany.com/customer-service`
     - Host-based: `https://customer-service.inference.mycompany.com`
   - Requires: Ingress controller (NGINX, Istio, Kong, etc.)
   - Phase 2 Enhancement: Generate Ingress/IngressRoute YAML alongside InferenceService

2. **LoadBalancer Services**
   - Each service gets unique external IP address
   - Simple direct access with no ingress layer
   - Drawback: One external IP per service (costly in cloud environments)
   - Use case: Small number of high-priority services needing dedicated IPs

3. **NodePort Services**
   - Expose services on static ports (30000-32767) across all nodes
   - Access via `<node-ip>:<nodeport>`
   - Drawback: Limited port range, requires exposing node IPs
   - Use case: On-premise deployments with direct node access

4. **Service Mesh (Advanced)**
   - Istio/Linkerd for advanced traffic management, mutual TLS, and observability
   - Supports A/B testing, canary deployments, circuit breaking
   - Use case: Complex multi-tenant environments with sophisticated routing needs

**Multi-Tenant Considerations**:
- Each department/team gets dedicated InferenceService with isolated resources
- Ingress provides unified entry point with path/host routing to appropriate service
- Authentication/authorization enforced at ingress layer (OAuth, JWT, API keys)
- Resource quotas and network policies for tenant isolation
- Separate namespaces per tenant for strong isolation (optional)

---

### Knowledge Base (Data Layer - Detailed)
**Purpose**: Store performance data, industry standards, deployment patterns, and use case templates

**Technology Implementation**: Hybrid storage approach
- **PostgreSQL**: High-volume query data (benchmarks, deployment outcomes)
- **JSON files**: Configuration as code (SLO templates, model catalog, hardware profiles)

**Rationale for Hybrid Approach**:
- **PostgreSQL for benchmarks**: Efficient querying on (model, GPU, traffic profile) combinations
  - psycopg2 with connection pooling for efficient query execution
  - Schema defined in `scripts/schema.sql`
  - Exact traffic matching on (prompt_tokens, output_tokens)
  - Pre-calculated p95 metrics (TTFT, ITL, E2E) from benchmark data
  - Implemented in `backend/src/knowledge_base/benchmarks.py`
- **JSON for configuration**: Version control, easy curation, human-readable
  - Files in `data/` directory
  - Easy to edit and review in pull requests
  - No database setup required for configuration changes

**Future Enhancement**: PostgreSQL + pgvector for embedding-based similarity search

**Data Collections**:

#### 6a. Model Benchmarks
Comprehensive performance metrics for model + GPU combinations.

**Phase 1 POC**: Synthetic JSON data with point estimates (deprecated).

**Phase 2 MVP (Current)**: PostgreSQL database with structured benchmark data:
- **Traffic Profile Framework**: Benchmarks organized around 4 GuideLLM standard profiles:
  - **(512 → 256)**: Medium input, short output - chatbots, Q&A, code completion
  - **(1024 → 1024)**: Long input, long output - content generation, translation
  - **(4096 → 512)**: Very long input, short output - summarization, document analysis
  - **(10240 → 1536)**: Extra-long input, medium output - multi-document analysis
- **Exact Matching**: Queries require exact (prompt_tokens, output_tokens) match - no fuzzy matching
- **p95 Metrics**: TTFT, ITL, E2E latency stored as 95th percentile for conservative SLO guarantees
- **Pre-calculated E2E**: E2E latency stored directly from benchmarks (not dynamically calculated)
- Benchmarks collected using vLLM default configuration with dynamic batching enabled
- Database schema: `scripts/schema.sql` (exported_summaries table)
- See `docs/traffic_and_slos.md` for full traffic profile framework

**Phase 3+ Enhancement**: Multi-dimensional benchmarks capturing:
- Concurrency levels and traffic patterns (bursty vs steady affects tail latencies)
- KV cache efficiency (prefix caching reduces TTFT for similar prompts)
- Different vLLM configuration tunings for specific workload patterns
- Parametric models for interpolation when exact traffic matches aren't available

**Benchmark Schema** (PostgreSQL):
```sql
-- From scripts/schema.sql
CREATE TABLE exported_summaries (
    id SERIAL PRIMARY KEY,
    model_hf_repo VARCHAR(255) NOT NULL,
    hardware VARCHAR(100) NOT NULL,
    hardware_count INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL,        -- Exact traffic profile input
    output_tokens INTEGER NOT NULL,        -- Exact traffic profile output
    ttft_p95 NUMERIC(10, 2) NOT NULL,     -- Time to First Token (p95) in ms
    itl_p95 NUMERIC(10, 2) NOT NULL,      -- Inter-Token Latency (p95) in ms/token
    e2e_p95 NUMERIC(10, 2) NOT NULL,      -- End-to-End latency (p95) in ms
    requests_per_second NUMERIC(10, 2),   -- Throughput (QPS)
    -- Additional metadata fields...
);

-- Unique constraint ensures one benchmark per configuration + traffic profile
CREATE UNIQUE INDEX idx_benchmark_config ON exported_summaries(
    model_hf_repo, hardware, hardware_count,
    prompt_tokens, output_tokens
);
```

**Example Benchmark Data**:
```json
{
  "model_hf_repo": "meta-llama/Llama-3.1-8B-Instruct",
  "hardware": "H100",
  "hardware_count": 1,
  "prompt_tokens": 512,
  "output_tokens": 256,
  "ttft_p95": 45.2,
  "itl_p95": 12.8,
  "e2e_p95": 3325.0,
  "requests_per_second": 25.4,
  "mean_input_tokens": 512,
  "mean_output_tokens": 256
}
```

**Important Design Decision - E2E Latency Storage**:

**Phase 1**: E2E latency was dynamically calculated from TTFT + (tokens × TPOT) because it varied by workload.

**Phase 2 (Current)**: E2E latency is **PRE-CALCULATED and stored** in benchmark data:
- Benchmarks use fixed traffic profiles (512→256, 1024→1024, etc.)
- Each benchmark includes measured `e2e_p95` for that specific traffic pattern
- This provides **actual measured E2E** under realistic load conditions
- No need for dynamic calculation - query returns complete metrics

**Why This Changed**:
- **Accuracy**: Measured E2E includes real-world effects (batching, queueing, scheduling)
- **Simplicity**: Direct lookup instead of estimation formulas
- **Consistency**: All metrics (TTFT, ITL, E2E) at same percentile (p95)
- **Traffic Profiles**: Fixed GuideLLM profiles mean E2E is consistent per configuration

**Query Pattern** (exact matching):
```python
benchmark = repo.get_benchmark(
    model_hf_repo="meta-llama/Llama-3.1-8B-Instruct",
    hardware="H100",
    hardware_count=1,
    prompt_tokens=512,      # Must match exactly
    output_tokens=256       # Must match exactly
)
# Returns complete metrics: ttft_p95, itl_p95, e2e_p95
```

**Phase 3+ Enhancement**: Support traffic profiles outside the 4 GuideLLM standards:
- Parametric models or interpolation for custom (prompt, output) combinations
- Account for workload-specific factors (streaming, concurrency, queueing)

#### 6b. Hardware Profiles
GPU specifications, availability, and pricing.

**Phase 1 Implementation Note**: GPU specifications and pricing are currently embedded in code for rapid prototyping:
- GPU pricing: `DeploymentGenerator.GPU_PRICING` dict in `backend/src/deployment/generator.py`
- GPU metadata: `GPUType` class in `backend/src/knowledge_base/model_catalog.py`

This allows quick iteration but requires code changes to update pricing. Future phases should extract to separate data files for easier maintenance and support for external data source integration (see Future Enhancements section).

**Phase 2 Target Schema**:
```json
{
  "gpu_type": "NVIDIA-A100-80GB",
  "vram_gb": 80,
  "tflops_fp16": 312,
  "memory_bandwidth_gbps": 2039,
  "manufacturer": "NVIDIA",
  "architecture": "Ampere",
  "available": true,
  "supported_parallelism": ["tensor", "pipeline"],
  "max_tensor_parallel": 8
}
```

#### 6c. Cost Data
Pricing information for different GPU types.

**Phase 1 Implementation Note**: Cost data is embedded in `DeploymentGenerator.GPU_PRICING` for simplicity. Currently assumes cloud rental pricing model only. Future phases should externalize and expand to support multiple deployment models (see Future Enhancements section).

**Phase 2 Target Schema**:
```json
{
  "gpu_type": "NVIDIA-A100-80GB",
  "cloud_provider": "AWS",
  "instance_type": "p4d.24xlarge",
  "hourly_rate_usd": 32.77,
  "monthly_rate_usd": 23914.00,
  "region": "us-east-1",
  "effective_date": "2025-01-01"
}
```

#### 6d. Use Case SLO Templates
Maps use cases to traffic profiles and experience-driven SLO targets.

**Phase 2 (Current)**: 9 use cases mapped to 4 GuideLLM traffic profiles with experience classes:

```json
{
  "use_case": "chatbot_conversational",
  "description": "Conversational assistants, customer support, Q&A",
  "traffic_profile": {
    "prompt_tokens": 512,
    "output_tokens": 256
  },
  "experience_class": "conversational",
  "slo_targets": {
    "ttft_p95_target_ms": 150,
    "itl_p95_target_ms": 25,
    "e2e_p95_target_ms": 7000
  },
  "rationale": "Highly interactive; perceived responsiveness drives satisfaction"
}
```

**All 9 Use Cases** (see `data/slo_templates.json` and `docs/traffic_and_slos.md`):
| Use Case | Traffic Profile | Experience Class | TTFT p95 | ITL p95 | E2E p95 |
|----------|----------------|------------------|----------|---------|---------|
| chatbot_conversational | 512→256 | Conversational | ≤150ms | ≤25ms | ≤7s |
| code_completion | 512→256 | Instant | ≤100ms | ≤20ms | ≤5s |
| code_generation_detailed | 1024→1024 | Interactive | ≤300ms | ≤30ms | ≤25s |
| translation | 1024→1024 | Deferred | ≤400ms | ≤35ms | ≤35s |
| content_generation | 1024→1024 | Deferred | ≤500ms | ≤35ms | ≤40s |
| summarization_short | 4096→512 | Interactive | ≤600ms | ≤30ms | ≤18s |
| document_analysis_rag | 4096→512 | Interactive | ≤600ms | ≤30ms | ≤18s |
| long_document_summarization | 10240→1536 | Deferred | ≤1000ms | ≤40ms | ≤60s |
| research_legal_analysis | 10240→1536 | Batch | ≤2000ms | ≤45ms | ≤90s |

**Key Concepts**:
- **Traffic Profile**: Defines computational load shape (prompt→output tokens)
- **Experience Class**: Defines UX expectations (instant, conversational, interactive, deferred, batch)
- **SLO Targets**: Experience-driven latency requirements (p95 percentiles)
- Same traffic profile can have different SLOs based on UX needs (e.g., code completion vs chatbot)
- See `docs/traffic_and_slos.md` for detailed framework explanation

#### 6e. Model Catalog
Curated list of approved models with metadata.

```json
{
  "model_id": "meta-llama/Llama-3-8b-instruct",
  "model_family": "Llama-3",
  "size_parameters": "8B",
  "context_window": 8192,
  "supported_tasks": ["chatbot", "summarization", "qa", "code-generation"],
  "subject_matter_domains": ["general", "technical"],
  "quality_tier": "high",
  "license": "Meta Community License",
  "approved_for_production": true,
  "minimum_vram_gb": 16,
  "supports_quantization": true
}
```

#### 6f. Deployment Templates
Kubernetes YAML templates for generating deployment configurations.

Deployment templates are implemented as **Jinja2 YAML templates** (`backend/src/deployment/templates/*.yaml.j2`). This approach provides:

- Direct YAML generation with proper formatting and indentation
- Conditional logic for simulator mode vs. real vLLM deployments
- Easy maintenance aligned with Kubernetes ecosystem standards (Helm, KServe)
- Human-readable templates that operators can review and customize

**Current Templates**:
- `kserve-inferenceservice.yaml.j2`: KServe InferenceService with vLLM runtime
- `vllm-config.yaml.j2`: vLLM server configuration (model loading, parallelism)
- `autoscaling.yaml.j2`: Horizontal Pod Autoscaler based on inference metrics
- `servicemonitor.yaml.j2`: Prometheus ServiceMonitor for observability

**Template Context** (passed from `DeploymentRecommendation`):
```python
{
  "deployment_id": "chatbot-llama-3-8b-20251011140322",
  "model_id": "meta-llama/Llama-3-8b-instruct",
  "gpu_type": "NVIDIA-L4",
  "gpu_count": 2,
  "tensor_parallel": 1,
  "min_replicas": 1,
  "max_replicas": 4,
  "simulator_mode": False,
  "vllm_config": {
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.9,
    "max_num_seqs": 256
  },
  "slo_targets": {...},
  "cost_estimate": {...}
}
```

**Example Template Snippet** (from `kserve-inferenceservice.yaml.j2`):
```yaml
{% if simulator_mode %}
image: vllm-simulator:latest
resources: {}  # No GPU required
{% else %}
image: vllm/vllm-openai:{{ vllm_version }}
resources:
  limits:
    nvidia.com/gpu: {{ gpus_per_replica }}
{% endif %}
```

#### 6g. Deployment Outcomes (Phase 3+)
Historical data from real deployments for continuous learning.

```json
{
  "deployment_id": "deploy-abc123",
  "template_id": "chatbot-low-latency-medium-scale",
  "model_id": "meta-llama/Llama-3-8b-instruct",
  "gpu_type": "NVIDIA-L4",
  "gpu_count": 2,
  "actual_metrics": {
    "ttft_p90_ms": 185,
    "tpot_p90_ms": 42,
    "throughput_avg_rps": 125,
    "cost_per_hour_usd": 4.20
  },
  "slo_compliance": {
    "ttft": true,
    "tpot": true,
    "e2e_latency": true,
    "availability": 99.95
  },
  "traffic_observed": {
    "prompt_length_avg": 165,
    "generation_length_avg": 220,
    "peak_qps": 95
  },
  "deployed_at": "2025-01-10T00:00:00Z",
  "observation_period_days": 7
}
```

---

### Observability Engine (Detailed)
**Purpose**: Monitor deployed inference services, track performance, and provide insights for optimization

**Technology Choices**:
- **Phase 2 (Current)**: Kubernetes API, direct HTTP requests, Streamlit dashboard
- **Phase 3+ (Future)**: Prometheus + Grafana, vLLM metrics endpoint, OpenTelemetry, advanced SLO tracking

**Key Responsibilities**:

#### Deployment Status Monitoring
Show what instances have been deployed:
- **Cluster connectivity**: Verify Kubernetes cluster is accessible
- **Pod status**: Show running/pending/failed inference services
- **Service endpoints**: Display accessible inference URLs
- **Basic health checks**: Verify pods are Ready and Serving

#### Basic Inference Testing
Enable sanity testing by sending queries:
- **Manual query interface**: Send test prompts to deployed models
- **Response validation**: Verify model generates reasonable outputs
- **Simple latency measurement**: Show basic response times (not statistical SLO tracking)
- **Error detection**: Display failures or timeout errors

**Dashboard Example** (Current Implementation):
```
┌─────────────────────────────────────────────────────────┐
│ Deployed Models                                         │
├─────────────────────────────────────────────────────────┤
│ Name: llama3-8b-chatbot                                 │
│ Status: ✓ Ready (2/2 pods running)                      │
│ Endpoint: http://llama3-8b-chatbot.default.svc          │
│                                                         │
│ Test Inference:                                         │
│ Prompt: "What is the capital of France?"                │
│ Response: "The capital of France is Paris..."           │
│ Latency: ~450ms                                         │
└─────────────────────────────────────────────────────────┘
```

**Phase 3+ Future Enhancements** (Advanced SLO Monitoring):

The following features are planned for future phases but not currently implemented:

#### SLO Compliance Monitoring
- **TTFT/ITL/E2E Latency**: p50, p95, p99 percentile tracking (Phase 2 uses p95 targets)
- **Throughput metrics**: Actual requests/sec and tokens/sec vs predicted
- **Reliability**: Uptime, error rates, timeout rates
- **Alerting**: Proactive alerts when SLOs are violated

#### Resource Utilization Metrics
- **GPU utilization**: Compute usage per GPU (target: >80% for cost efficiency)
- **GPU memory usage**: VRAM consumption vs available
- **Batch efficiency**: Average batch size achieved
- **Request queue depth**: Backlog indicating capacity constraints

#### Cost Tracking
- **Actual cost per hour/month**: Compare to predicted
- **Cost per 1k tokens**: Unit economics
- **Idle GPU time**: Identify overprovisioning

#### Traffic Pattern Analysis
- **Prompt/output length distribution**: Actual vs predicted
- **QPS patterns**: Steady-state, peak, burstiness
- **Temporal patterns**: Diurnal cycles, weekday/weekend differences

#### Feedback Loop to Knowledge Base
- Store actual performance metrics in Deployment Outcomes
- Compare predicted vs actual TTFT, ITL, throughput, cost
- Identify systematic biases in predictions
- Update benchmark data and recommendation logic
- Enable continuous improvement of capacity planning accuracy

**Technology for Phase 3+**:
- **OpenTelemetry + Prometheus + Grafana** - Standard observability stack for metrics
- **vLLM metrics endpoint** - Native Prometheus metrics for inference stats
- **KServe ServiceMonitor** - Kubernetes-native metrics collection

---

## End-to-End Data Flow

```
User: "I need a chatbot for 1000 users, low latency is critical"
    ↓
┌─────────────────────────────────────────────────────────────┐
│                    UI Layer                                 │
│         Conversational Interface (Streamlit)                │
└───────────────────────┬─────────────────────────────────────┘
                        │ Natural Language Input
                        ↓
┌──────────────────────────────────────────────────────────────┐
│          Intent & Specification Engine                       │
│  • Extract: task_type, users, priorities, constraints        │
│  • Map use case → traffic profile (prompt/output tokens)     │
│  • Lookup SLO template → default latency targets             │
│  • Generate complete DeploymentIntent specification          │
└───────────────────────┬──────────────────────────────────────┘
                        │ DeploymentIntent Specification
                        │ (includes traffic profile + SLO targets)
                        ↓
┌─────────────────────────────────────────────────────────────┐
│                    UI Layer                                 │
│         Specification Review & Editor                       │
│  • Display auto-generated spec (traffic, SLOs, budget)      │
│  • Allow inline editing                                     │
│  • Re-trigger Intent Engine if changes                      │
└───────────────────────┬─────────────────────────────────────┘
                        │ Specification
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              Recommendation Engine                          │
│  ┌──────────────────────────────────────────┐               │
│  │ Model Selector                           │               │
│  │  → Filter by task compatibility          │               │
│  │  → Rank by user priority                 │               │
│  └──────────────────────────────────────────┘               │
│  ┌──────────────────────────────────────────┐               │
│  │ Capacity Planner                         │               │
│  │  → Query benchmarks for traffic profile  │               │
│  │  → Filter by SLO compliance              │               │
│  │  → Calculate GPU count & cost            │               │
│  └──────────────────────────────────────────┘               │
│  ┌──────────────────────────────────────────┐               │
│  │ Solution Recommender                     │               │
│  │  → Rank by user priorities               │               │
│  └──────────────────────────────────────────┘               │
│              ↕ Query Knowledge Base                         │
│        (PostgreSQL benchmarks, JSON configs)                │
└───────────────────────┬─────────────────────────────────────┘
                        │ Ranked Recommendations
                        ↓
┌──────────────────────────────────────────────────────────────┐
│                    UI Layer                                  │
│         Recommendation Visualizer                            │
│  Present options with trade-offs:                            │
│  "Llama-3-8B on 2x L4: $800/mo, TTFT p95<200ms, ITL p95<50ms"│
│  "Llama-3-70B on 4x A100: $2400/mo, better quality"          │
│  • Compare scenarios (future)                                │
│  • What-if analysis (future)                                 │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ↓
          User selects configuration
                        │
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              Deployment Engine                              │
│  → Generate KServe/vLLM YAML                                │
│  → Validate configuration                                   │
│  → Deploy to Kubernetes                                     │
│  → Configure observability                                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓
      Return: deployment ID + monitoring dashboard link
                        │
                        ↓
┌──────────────────────────────────────────────────────────────────┐
│              Observability Engine                                │
│  Monitor: health, basic metrics (Phase 2)                        │
│  → Collect actual TTFT/ITL/E2E, traffic profile, cost (Phase 3+) │
└───────────────────────┬──────────────────────────────────────────┘
                        │ (Future: Store actual metrics)
                        ↓
┌─────────────────────────────────────────────────────────────┐
│         Knowledge Base (deployment_outcomes)                │
│  → Intent & Specification Engine learns from history        │
│  → Refines use case mapping, traffic profiles, SLOs         │
│  → Improves future recommendations                          │
└─────────────────────────────────────────────────────────────┘
```

**Optional Feedback Loops:**
- 📝 **Spec Editor → Intent Engine**: User edits specifications, re-triggers intent extraction
- 🔄 **Recommendation Visualizer → Intent Engine**: User relaxes SLOs or changes priorities
- 🔮 **Knowledge Base → Intent Engine** (Phase 3+): Learn from actual deployment performance

---

## Technology Choices by Phase

| Component | Phase 2 (Current) | Phase 3+ (Future) |
|-----------|-------------------|-------------------|
| **UI Layer** | Streamlit | React + WebSocket backend |
| **Intent & Specification Engine** | Ollama (llama3.1:8b) + Pydantic | Enhanced structured extraction frameworks |
| **Recommendation Engine** | Rule-based + SLO filtering | ML-based with historical learning |
| **Deployment Engine** | Jinja2 + Kubernetes Python client | Possible Go migration for operators |
| **Observability Engine** | Kubernetes API + Streamlit | Prometheus + Grafana + vLLM metrics |
| **Knowledge Base** | **PostgreSQL + JSON files** | PostgreSQL + pgvector |
| **API Gateway** | FastAPI + in-memory state | FastAPI + persistent state (LangGraph/Temporal) |

**Key Architecture Features** (Current):
- **Hybrid storage**: PostgreSQL for benchmarks, JSON for configuration
- **p95 SLO targets**: Conservative guarantees for TTFT, ITL, E2E latency
- **ITL (Inter-Token Latency)**: Standard terminology for per-token generation time
- **Pre-calculated E2E latency**: Stored in benchmarks from actual measurements
- **4 GuideLLM traffic profiles**: Standard benchmark configurations (512→256, 1024→1024, 4096→512, 10240→1536)
- **9 use cases with experience classes**: Map business needs to latency requirements
- **Traffic profile in specification**: Intent & Specification Engine determines traffic, Recommendation Engine consumes it

---

## Key Design Principles

1. **Conversational-first**: All complexity hidden behind natural language (no forms)
2. **Benchmarking-driven**: Recommendations grounded in real performance data
3. **Explainable**: Users understand *why* each recommendation is made
4. **Iterative**: Support what-if exploration before committing resources
5. **One-click deployment**: Minimize manual YAML editing
6. **Observability by default**: Every deployment includes monitoring hooks
7. **Experience-driven SLOs**: Latency targets based on UX requirements, not just hardware capabilities (Phase 2)

---

## Traffic Profiles and Experience-Driven SLOs (Phase 2 Framework)

Phase 2 introduces a unified framework for mapping use cases to both traffic profiles and SLO targets. This framework is documented in detail in `docs/traffic_and_slos.md`.

### Core Concepts

**Traffic Profiles** define the computational load shape:
- **Input tokens (prompt)**: Drives prefill cost → affects TTFT
- **Output tokens (completion)**: Drives generation time → affects ITL and E2E
- 4 GuideLLM standard profiles: (512→256), (1024→1024), (4096→512), (10240→1536)

**Experience Classes** define user expectations:
- **Instant** (e.g., code completion): TTFT ≤150ms, ITL ≤25ms - feels real-time
- **Conversational** (e.g., chatbot): TTFT ≤300ms, ITL ≤30ms - natural dialogue
- **Interactive** (e.g., RAG): TTFT ≤500ms, ITL ≤35ms - some waiting acceptable
- **Deferred** (e.g., translation): TTFT ≤1s, ITL ≤40ms - spinner acceptable
- **Batch** (e.g., research): TTFT ≤2s, ITL ≤45ms - asynchronous processing

**Key Insight**: Same traffic profile can have different SLOs based on UX needs:
- Code completion (512→256, instant): TTFT ≤100ms, ITL ≤20ms
- Chatbot (512→256, conversational): TTFT ≤150ms, ITL ≤25ms

### Hardware Selection Strategy

Experience classes guide hardware tier recommendations:

| Experience Class | Hardware Tier | Batching | Priority |
|-----------------|---------------|----------|----------|
| Instant / Conversational | Premium (A100/H100) | Small (≤4) | Low latency |
| Interactive | Balanced (L40S/A10G) | Medium (8-16) | Balance |
| Deferred / Batch | Cost-optimized (A10/T4) | Large (≥16) | Throughput |

### Throughput Scaling

If required QPS exceeds single GPU capacity:
- **Horizontal scaling**: Replicate instances of same type
- **Per-instance SLO**: Each instance must meet SLO targets independently
- Example: 100 QPS required, single GPU handles 25 QPS → deploy 4 replicas

For complete framework details, see `docs/traffic_and_slos.md`.

---

## Development Tools

### vLLM Simulator

The vLLM Simulator is a critical development tool that enables GPU-free development, testing, and demonstrations without requiring expensive GPU hardware.

**Purpose**: Provide a drop-in replacement for vLLM during development and testing

**Technology Stack**:
- **FastAPI** - Implements OpenAI-compatible API endpoints
- **Docker** - Single containerized image for all models
- **Python** - Benchmark-driven latency simulation

**Key Features**:

#### API Compatibility
Provides OpenAI-compatible endpoints that match vLLM:
- **`/v1/completions`** - Standard OpenAI completions endpoint
- **`/v1/chat/completions`** - Chat completions endpoint
- **`/health`** - Health check endpoint
- **`/metrics`** - Prometheus-compatible metrics endpoint

#### Realistic Performance Simulation
Uses actual benchmark data to simulate production behavior:
- **TTFT simulation**: Sleep for benchmark-derived time-to-first-token
- **TPOT simulation**: Sleep per output token based on benchmark data
- **Token counting**: Accurate prompt and completion token counts
- **Latency variance**: Add realistic jitter to simulated latencies

#### Response Generation
Pattern-based canned responses for different use cases:
- **Code generation**: Return sample Python/JavaScript code snippets
- **Chat**: Return conversational responses
- **Summarization**: Return condensed text summaries
- **Q&A**: Return factual answers
- **Generic**: Return sensible default text

#### Configuration
Single Docker image configured via environment variables:
- **`MODEL_NAME`**: Which model to simulate (e.g., "mistralai/Mistral-7B-Instruct-v0.3")
- **`GPU_TYPE`**: GPU type for benchmark lookup (e.g., "NVIDIA-L4")
- **`TENSOR_PARALLEL_SIZE`**: Number of GPUs for benchmark lookup
- **`BENCHMARKS_PATH`**: Path to benchmark data JSON file

**Deployment Modes**:

The Deployment Automation Engine supports toggling between simulator and real vLLM:

1. **Simulator Mode** (default for POC):
   - Uses `vllm-simulator:latest` Docker image
   - No GPU resources requested
   - Fast startup (~10-15 seconds to Ready)
   - Runs on CPU-only Kubernetes clusters (KIND, minikube)
   - Controlled via `DeploymentGenerator(simulator_mode=True)`

2. **Real vLLM Mode** (production):
   - Uses `vllm/vllm-openai:v0.6.2` Docker image
   - Requests GPU resources (nvidia.com/gpu)
   - Downloads models from HuggingFace
   - Actual inference with real GPUs
   - Controlled via `DeploymentGenerator(simulator_mode=False)`

**Benefits**:
- **No GPU required**: Developers can test full deployment workflows on laptops
- **Fast feedback**: Quick iteration without waiting for GPU provisioning
- **Consistent behavior**: Predictable responses for demos and testing
- **Cost savings**: No GPU costs during development
- **CI/CD friendly**: Automated tests without GPU infrastructure

**Integration**:
- Deployed via same KServe InferenceService YAML as real vLLM
- Monitored via same Kubernetes API as production deployments
- Testable via Inference Testing UI in Streamlit
- Uses actual benchmark data for realistic latency simulation

---

## Integration Points

### With Kubernetes Ecosystem
- **KServe**: Primary model serving platform
- **vLLM**: LLM runtime for inference
- **Kubernetes**: Container orchestration platform
- **InstructLab**: Model fine-tuning and serving
- **Observability Stack**: Metrics and monitoring infrastructure (Prometheus/Grafana)

### External Dependencies
- Model registries (HuggingFace Hub, custom)
- GPU hardware inventory systems
- Cost management APIs
- Identity/authentication (SSO)

---

## Security Considerations

1. **Input Validation**: Sanitize all user inputs before LLM processing
2. **Configuration Validation**: Verify generated YAML before Kubernetes deployment
3. **RBAC**: Respect cluster-level permissions for deployments
4. **Secrets Management**: Never expose credentials in configs or logs
5. **Audit Trail**: Log all deployment actions with user attribution

---

## Success Metrics

### User Experience
- Time from conversation start to successful deployment (target: < 10 minutes)
- Recommendation acceptance rate (target: > 70%)
- User satisfaction score (target: > 4/5)

### System Performance
- Conversation response latency (target: < 2 seconds)
- Deployment success rate (target: > 95%)
- System uptime (target: 99.9%)

### Business Impact
- Increase in AI adoption (measure deployments/month)
- Reduction in overprovisioning (measure GPU utilization)
- Customer retention (measure engagement with assistant)

---

## Open Questions for Refinement

1. **Multi-tenancy**: How do we isolate conversations/deployments across users?
2. **Security**: How do we validate generated configs before K8s deployment?
3. **Feedback loop**: How do we capture post-deployment outcomes to improve recommendations?
4. **Model catalog integration**: Do we sync with HuggingFace, or maintain a curated list?
5. **Versioning**: How do we handle updates to recommendations as benchmarks improve?
6. **Rollback**: What's the UX for reverting a deployment if it doesn't meet expectations?
7. **Conversational clarification flow** (Future Phase): How do we handle insufficient information gracefully?
   - When user provides minimal details, system should ask targeted follow-up questions
   - More detailed information → more precise recommendations with narrower option set
   - Limited information → broader recommendation set with clear pros/cons for each
   - System should gracefully handle "I don't know" responses and still provide useful recommendations
   - Balance between gathering information and not overwhelming user with too many questions

---

## Possible Future Enhancements (Post-Phase 1)

### Conversational Interface Layer
- **Alternative UI Frameworks**: Chainlit (Python-first conversational UI) or Custom React + WebSocket backend for production deployments
- Enhanced session management and multi-user support

### Context & Intent Engine
- **Full statistical distributions** for traffic profiles (not just point estimates)
  - Prompt length distributions (mean, variance, percentiles)
  - Generation length distributions
  - Concurrency patterns and burstiness modeling

### Traffic Modeling (Phase 2)
- Full statistical distributions (not just point estimates)
- Advanced burstiness pattern recognition
- Request heterogeneity modeling (mix of short/long requests)
- Streaming vs non-streaming workload separation
- Industry benchmark lookup from Knowledge Base for use case templates
- Pattern learning from historical deployment data

### Capacity Planning (Phase 2)
- **Advanced E2E latency calculation**:
  - Queueing delays under high load
  - Concurrency effects on tail latencies
  - Streaming-aware user perception (first chunk vs full completion)
  - Non-linear scaling effects at saturation
- **Optimization algorithms**:
  - Optimize batch size for throughput/latency trade-off
  - Advanced tensor parallelism optimization
  - Pipeline parallelism support
  - Auto-scaling policy generation (min/max replicas)
  - Queueing theory models for latency under load

### Conversational Interface Layer
- **Advanced what-if analysis and simulation**:
  - **Side-by-side scenario comparison**: Compare multiple configurations (different models, GPU types, SLO targets) with visual diff showing cost/latency/quality trade-offs
  - **Interactive cost/latency sliders**: Real-time updates as users adjust SLO targets or traffic parameters
  - **Discrete event simulation (SimPy)**: More accurate workload modeling accounting for:
    - Arrival patterns (Poisson, bursty, diurnal cycles)
    - Queueing delays under load
    - Auto-scaling behavior (scale-up/scale-down latency)
    - Cold start penalties for new replicas
  - **Monte Carlo simulation**: Uncertainty quantification for cost and latency predictions
    - Account for variance in traffic patterns
    - Confidence intervals for SLO compliance (e.g., "95% confident TTFT p90 < 200ms")
    - Risk analysis for budget overruns
  - **Historical performance extrapolation**: Use real benchmark data interpolation for non-exact traffic matches
  - **Resource utilization projections**: GPU memory, compute efficiency, batch size optimization
  - **What-if scenario builder**:
    - "What if traffic doubles?" → show new GPU requirements and costs
    - "What if we relax TTFT from 200ms to 500ms?" → show cost savings
    - "What if we switch to quantized models?" → show latency/quality trade-offs
  - **Saved scenarios**: Allow users to save and revisit different configuration explorations

### Deployment Automation Engine
- **Migration to Go** (optional, for advanced K8s integration):
  - Custom Kubernetes operators for deployment lifecycle management
  - Advanced watch patterns for real-time cluster state monitoring
  - Performance optimization for large-scale deployments
  - Native K8s ecosystem integration (smaller container footprint)
- **External access patterns** (production requirement):
  - Ingress controller integration (NGINX, Traefik, Istio)
  - Generate Ingress/IngressRoute YAML alongside InferenceService
  - TLS/certificate management automation
  - Load balancer configuration

### Day-2 Operations
- Proactive optimization recommendations for running deployments
- Automated quantization suggestions based on latency requirements
- Speculative decoding automation
- Cost anomaly detection and alerts
- Drift detection (actual vs predicted SLOs)

### Advanced Features
- Multi-model deployments (model ensembles)
- A/B testing automation
- Fine-tuning pipeline integration
- Capacity planning for MaaS deployments
- Continuous learning from deployment outcomes

### Cost Modeling & Deployment Models (Phase 2+)

**Current Limitation**: Phase 1 assumes cloud GPU rental pricing model only.

**Future Support for Multiple Deployment Models**:

1. **Cloud Rental** (current):
   - GPU hourly/monthly rates from cloud providers
   - Multi-cloud support (AWS, GCP, Azure, OCI)
   - Regional pricing variations
   - Spot vs on-demand vs reserved instance pricing

2. **Owned Hardware**:
   - User already owns GPUs, only interested in optimal model/GPU combination
   - Optional operational cost modeling (power, cooling, datacenter space)
   - If user provides power rates: calculate monthly OpEx (power_watts × hours × $/kWh × PUE)
   - If user doesn't provide rates: assume costs are already accounted for, focus on minimizing GPU count while meeting SLOs
   - Recommendation strategy: Prefer configurations with fewest GPUs (maximize utilization of existing hardware)

3. **GPU Purchase + On-Premises**:
   - Total Cost of Ownership (TCO) analysis over amortization period (e.g., 36 months)
   - CapEx: GPU purchase price
   - OpEx: Power, cooling (PUE factor), rack space, maintenance
   - Break-even analysis vs. cloud rental
   - Recommendation threshold: Suggest purchase if sustained usage exceeds break-even period

**Required Schema Extensions**:
```python
class DeploymentIntent:
    deployment_model: Literal["cloud_rental", "owned_hardware", "purchase"] = "cloud_rental"
    datacenter_context: Optional[DatacenterContext] = None

class DatacenterContext:
    power_cost_per_kwh: Optional[float] = None  # e.g., 0.12 ($/kWh)
    cooling_overhead_pue: float = 1.3  # Power Usage Effectiveness
    rack_space_cost_per_month: Optional[float] = None
    amortization_period_months: int = 36  # For purchase scenarios
```

**Hardware Profile Extensions**:
```json
{
  "gpu_type": "NVIDIA-A100-80GB",
  "pricing": {
    "cloud_rental_per_hour": 4.10,
    "purchase_price_usd": 15000,
    "tdp_watts": 400
  }
}
```

**Cost Calculation Examples**:
- **Owned (4x A100-80GB)**: Power: 400W × 4 × 730hrs × $0.12/kWh × 1.3 PUE = $182/month
- **Purchase (4x A100-80GB)**: CapEx $60k amortized over 36mo = $1,667/mo + $182 OpEx = $1,849/mo
- **Cloud Rental (4x A100-80GB)**: $4.10/hr × 4 GPUs × 730hrs = $11,984/month
- **Break-even**: Purchase pays off after ~41 months of sustained usage

**Implementation Strategy**:
- Owned hardware: Skip cost display or show operational costs only (power/cooling) if user provides rates
- Purchase: Show TCO comparison with cloud rental, highlight break-even point
- Cloud rental: Current implementation (hourly/monthly costs)

### Data Source Management (Phase 2+)

**Current Limitation**: Phase 1 uses manually curated JSON files for all data (benchmarks, model catalog, GPU pricing, SLO templates).

**Future Support for Multiple Data Sources**:

1. **Model Catalog**:
   - Manual database (current): Curated list in `data/model_catalog.json`
   - HuggingFace Hub API: Auto-discover available models, metadata, licenses
   - Private model registries: Custom endpoints for enterprise-approved models
   - Public MaaS providers: OpenAI, Anthropic, Cohere APIs for pricing/metadata
   - Private MaaS: Self-hosted model serving platforms

2. **GPU Pricing**:
   - Manual (current): `DeploymentGenerator.GPU_PRICING` dict
   - Cloud provider APIs: AWS Pricing API, GCP Cloud Billing API, Azure Retail Prices API
   - Spot pricing feeds: Real-time spot instance pricing
   - GPU vendor pricing: Scrape NVIDIA, AMD manufacturer suggested pricing for on-prem purchases
   - Third-party aggregators: ML infrastructure marketplaces

3. **Benchmarks**:
   - Manual (current): Curated `data/benchmarks.json`
   - Automated benchmark pipelines: Continuous benchmarking of new model/GPU combinations
   - Community-contributed benchmarks: Crowdsourced performance data with validation
   - Vendor benchmarks: Import official vLLM/HuggingFace/NVIDIA performance data

**Data Validation & Cross-Reference Requirements**:
- **Consistency Check**: Ensure models exist in catalog, have benchmarks, and have pricing for viable GPUs
- **Staleness Detection**: Flag outdated pricing (effective_date) or benchmarks (vLLM version mismatch)
- **Data Health API**: `/api/data/health` endpoint reporting missing cross-references and stale data
- **Startup Validation**: Log warnings for incomplete data during system initialization

**Example Validation Output**:
```
⚠️  Data Consistency Issues:
- Model "meta-llama/Llama-3.1-405B-Instruct" in catalog but no benchmarks available
- GPU "NVIDIA-H100" has benchmarks but missing cloud pricing data
- Benchmark "mistralai/Mistral-7B + NVIDIA-T4" references GPU not in hardware profiles
- Pricing data for "NVIDIA-L4" is 45 days old (last updated: 2024-11-27)
```

**Implementation Priority**:
- Phase 1.5: Add data validation utility and health check endpoint
- Phase 2: Support cloud provider pricing APIs (AWS, GCP, Azure)
- Phase 2+: HuggingFace Hub integration for model discovery
- Phase 3: Automated benchmark pipelines and community contributions
- Quality metric integration (BLEU scores, accuracy benchmarks)
