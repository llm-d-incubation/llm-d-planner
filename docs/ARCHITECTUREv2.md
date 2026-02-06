# NeuralNav Architecture v2

> **Note**: This document is a work in progress. The original
> [ARCHITECTURE.md](ARCHITECTURE.md) captured our initial design thinking.
> This revision reflects the production architecture vision that emerged from
> completing the POC. Once finalized, this document will replace
> ARCHITECTURE.md.

## Overview

### Vision

Accelerate AI adoption by making LLM deployment accessible to every
organization:

- **Reduce time-to-production** for LLM deployments through intelligent
  automation
- **Lower the barrier to entry** for customers new to GPU and model
  infrastructure
- **Simplify Kubernetes** for LLM-based solutions with guided configuration

### Mission

Ensure our customers fulfill their AI goals and maximize ROI of their GenAI
investments through deep visibility and personalized recommendations.

### Problem Statement

Deploying LLMs in production environments remains complex and error-prone:

- **Application Developers** struggle to translate business needs into
  infrastructure choices (model selection, GPU type, SLO targets)
- **Cluster Admins** face overprovisioning risks without guidance on aligning
  workloads to budgets and SLAs
- **Trial-and-error is the default**, extending time-to-production and inflating
  costs

### Solution Approach

NeuralNav streamlines the path from concept to production-ready LLM deployments
through:

1. **Conversational requirements gathering** - Natural language input eliminates
   need for infrastructure expertise
2. **SLO-driven capacity planning** - Translate business needs into technical
   specifications automatically
3. **Multi-criteria recommendation ranking** - Present optimal trade-offs
   between accuracy, cost, latency, and complexity
4. **Ready-to-use configurations** - Generate production-ready Kubernetes
   manifests for immediate deployment

---

## Architecture

NeuralNav consists of **five major components**:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          UI Layer (Presentation)                          │
│                   Streamlit (current) → React (future)                    │
│  Conversational Interface | Spec Editor | Recommendation Viewer | Config  │
└───────────────────────────────────────────────────────────────────────────┘
                            ↕ API Gateway (FastAPI)
┌───────────────────────────────────────────────────────────────────────────┐
│                            Backend Services                               │
├────────────────────┬────────────────────┬────────────────────┬────────────┤
│       Intent       │                    │                    │            │
│     Extraction     │   Specification    │  Recommendation    │   Config   │
│      Service       │      Service       │     Service        │  Service   │
│                    │                    │                    │            │
│   Conversation     │   User Intent      │   Specification    │  Selected  │
│        ↓           │        ↓           │        ↓           │ Recommend  │
│    Structured      │     Complete       │      Ranked        │     ↓      │
│      Intent        │   Specification    │  Recommendations   │   YAML     │
│                    │                    │                    │   Files    │
└────────────────────┴────────────────────┴────────────────────┴────────────┘
                            ↕ Data Access Layer
┌───────────────────────────────────────────────────────────────────────────┐
│                    Knowledge Base (Data Layer)                            │
│     Embedded DB: Performance Benchmarks, Accuracy Benchmarks              │
│     JSON Files: SLO Templates, Model Catalog, Hardware Profiles           │
└───────────────────────────────────────────────────────────────────────────┘
```

### 1. User Interface (Presentation Layer)

```text
┌───────────────────────────────────────────────────────────────────┐
│                    User Interface (Presentation)                  │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────┐ │
│  │Conversational│  │Specification │  │Recommendation│  │ Config │ │
│  │  Interface   │  │   Editor     │  │Viewer/Select │  │ Viewer │ │
│  │              │  │              │  │              │  │        │ │
│  │ Natural lang │  │ Edit SLOs,   │  │ 5 ranked     │  │ Display│ │
│  │ input, show  │  │ priorities,  │  │ views, trade │  │ YAML,  │ │
│  │ intent       │  │ thresholds   │  │ off analysis │  │download│ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────┘ │
│                                                                   │
│  User Conversation ────────────────────────────▶ Config Files     │
└───────────────────────────────────────────────────────────────────┘
```

**Current Implementation**: Streamlit **Future Migration**: React + WebSocket
backend

Provides unified interface across all system capabilities through four
integrated views:

#### a. Conversational Interface

- Accepts natural language description of use case
- Displays extracted intent for user review
- Allows clarification and refinement through multi-turn dialogue
- Obtains user approval before proceeding to specification

#### b. Specification Editor

- Displays auto-generated deployment specification
- Allows editing of multiple fields:
  - **SLO targets**: TTFT, ITL, E2E latency thresholds and percentile
    (p90/p95/p99)
  - **Quality thresholds**: Minimum accuracy score, maximum cost
  - **User priorities**: Weights for accuracy, cost, latency, complexity in
    balanced ranking
  - **Future**: Editing of traffic profile parameters (prompt/output tokens,
    QPS), power/cooling budgets
- Obtains user approval before sending to Recommendation Service

#### c. Recommendation Viewer/Selector

- Displays ranked recommendations in multiple views:
  - **List View**: Top recommendations in each category (Best Accuracy, Lowest
    Cost, Lowest Latency, Simplest, Balanced)
  - **Future**: Pareto frontier charts for two-dimensional trade-offs (e.g.,
    cost vs. accuracy)
  - **Future**: Multi-dimensional visualization (>2 dimensions)
- Allows user to select preferred configuration
- Provides detailed explanations for each recommendation

#### d. Configuration Section

- Displays generated configuration files for selected solution
- Allows download or copy of configuration files
- **Current Target**: KServe on Kubernetes (InferenceService YAML)
- **Future Options**: Direct Linux server deployment, other orchestration
  platforms

---

### Backend Components

Backend components interact with the UI via **FastAPI** REST endpoints.

---

### 2. Intent Extraction Service

```text
┌───────────────────────────────────────────────────────────────────┐
│                    Intent Extraction Service                      │
│                                                                   │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐  │
│  │  LLM Processor  │   │ Schema Validator│   │ Intent Builder  │  │
│  │                 │   │                 │   │                 │  │
│  │ Ollama/qwen2.5  │──▶│ Pydantic schema │──▶│ Structured      │  │
│  │ Engineered      │   │ validation      │   │ intent object   │  │
│  │ prompts         │   │                 │   │                 │  │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘  │
│                                                                   │
│  User Conversation ─────────────────────────▶ Structured Intent   │
└───────────────────────────────────────────────────────────────────┘
```

**Purpose**: Transform natural language conversation into structured user intent

**Input**: User's conversational description of use case and requirements

**Technology**:

- Small LLM for structured extraction (currently Ollama with qwen2.5:7b)
- Pydantic schema for intent validation
- Engineered prompts for consistent extraction

**Processing**:

- Extract use case type (chatbot, code completion, summarization, etc.)
- Identify user count and scale requirements
- Determine user priorities (accuracy, cost, latency, complexity)
- Capture domain specialization needs (e.g., medical, legal, financial)

**Output**: Structured intent object with:

- Use case classification
- Expected user scale
- Priority preferences (accuracy vs. cost vs. latency)
- Subject matter domain (optional)

**Interface**: REST endpoint `/api/v1/extract`

**Notes**:

- May require LLM fine-tuning for improved extraction accuracy
- Alternative approaches to LLM-based extraction may be considered

---

### 3. Specification Service

```text
┌───────────────────────────────────────────────────────────────────┐
│                      Specification Service                        │
│                                                                   │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐  │
│  │ Traffic Profile │   │  SLO Target     │   │    Benchmark    │  │
│  │   Generator     │   │    Mapper       │   │    Selector     │  │
│  │                 │   │                 │   │                 │  │
│  │ Use case →      │──▶│ Use case →      │──▶│ Use case →      │  │
│  │ tokens, QPS     │   │ TTFT, ITL, E2E  │   │ accuracy metrics│  │
│  │                 │   │ targets         │   │                 │  │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘  │
│                                                                   │
│  Structured Intent ───────────────────────▶ Complete Specification│
└───────────────────────────────────────────────────────────────────┘
```

**Purpose**: Generate complete deployment specification from user intent

**Input**: Structured intent from Intent Extraction Service

**Processing**:

- **Traffic Profile Generation**: Map use case → computational load shape
  - Prompt token length (e.g., 512, 1024, 4096)
  - Output token length (e.g., 256, 512, 1024)
  - Expected queries per second (QPS) based on user count
- **SLO Target Mapping**: Assign latency requirements based on use case
  - TTFT (Time to First Token) target
  - ITL (Inter-Token Latency) target
  - E2E (End-to-End) target
  - Percentile selection (p90, p95, or p99)
  - Throughput (requests per second) requirement
- **Accuracy Benchmark Selection**: Identify relevant quality metrics based on
  use case and subject matter

**Output**: Complete deployment specification including:

- **Performance Requirements**: TTFT, ITL, E2E latency targets, RPS throughput
- **Workload Profile**: Mean input/output tokens, distribution characteristics
- **Accuracy Benchmarks**: Use-case specific quality evaluation criteria (for
  scoring)
- **Quality Thresholds**: Minimum accuracy score, maximum cost (for filtering)
- **User Priorities**: Weights for accuracy, cost, latency, complexity
  trade-offs
- **Future**: Power budget, cooling budget thresholds

**Interface**: Called internally after Intent Extraction; specification
displayed in UI Spec Editor for user review and modification

**Design Principle**: Specifications are **always editable** - users can adjust
SLO targets, priorities, and constraints before proceeding to recommendations

---

### 4. Recommendation Service

```text
┌───────────────────────────────────────────────────────────────────┐
│                      Recommendation Service                       │
│                                                                   │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐  │
│  │  Config Finder  │   │     Scorer      │   │    Analyzer     │  │
│  │                 │   │                 │   │                 │  │
│  │ Query benchmarks│──▶│ Score configs   │──▶│ Generate        │  │
│  │ Filter by SLOs  │   │ on 5 dimensions:│   │ ranked views:   │  │
│  │ Calculate       │   │ - Accuracy      │   │ - Best Accuracy │  │
│  │ replicas        │   │ - Cost          │   │ - Lowest Cost   │  │
│  │                 │   │ - Latency       │   │ - Lowest Latency│  │
│  │                 │   │ - Complexity    │   │ - Simplest      │  │
│  │                 │   │ - Balanced      │   │ - Balanced      │  │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘  │
│                                                                   │
│  Specification ─────────────────────────────────▶ Ranked Options  │
└───────────────────────────────────────────────────────────────────┘
```

**Purpose**: Find optimal model + GPU configurations that meet specification
requirements

**Input**: User-approved specification from UI (potentially modified from
auto-generated version)

**Major Sub-Components**:

#### a. Config Finder

**Role**: Query performance database and generate viable configurations

**Processing**:

1. Query performance database for configurations meeting specification
   requirements:
   - Filter by exact traffic profile match (prompt tokens → output tokens)
   - Enforce SLO compliance (TTFT p95 ≤ target, ITL p95 ≤ target, E2E p95 ≤
     target)
   - Enforce quality thresholds (accuracy ≥ minimum, cost ≤ maximum)
   - Apply optional user constraints (included/excluded hardware, preferred
     models)
2. Calculate replicas required to meet throughput:
   - `replicas = ceil((required_QPS × 1.2) / QPS_per_GPU_config)`
   - 20% safety headroom for production stability
3. Generate complete solution:
   - Hardware type (e.g., NVIDIA L4, A100-80GB, H100)
   - Tensor parallelism degree (GPUs per instance)
   - Number of replicas (independent instances)
   - Expected latency metrics (TTFT, ITL, E2E)
   - Predicted throughput (RPS)

**Future Enhancement**: Use simulator or analytical models to generate options
without real benchmark data

#### b. Scorer

**Role**: Evaluate each configuration on multiple criteria

**Four Scoring Dimensions** (each 0-100 scale):

1. **Accuracy** (Primary Factor)
   - Use-case specific quality scores from benchmark databases
   - Fallback: Parameter count heuristic (7B → 55, 70B → 85, 405B → 95)
   - Source: Artificial Analysis benchmarks, weighted by use case

2. **Cost** (Primary Factor)
   - Monthly operational cost (GPU hourly rate × GPU count × 730 hours)
   - Non-linear scoring to create meaningful differentiation
   - Inverse relationship: lower cost = higher score

3. **Latency** (Secondary Factor)
   - SLO compliance headroom: how much better than target
   - Important for user experience but all configs already meet SLOs
   - Extra credit for exceeding requirements, but diminishing returns

4. **Complexity** (Secondary Factor)
   - Deployment and operational simplicity
   - Function of GPU count, tensor parallelism, replica count
   - Fewer resources = higher score (easier to manage)

**Future Primary Factors**:

- Power requirements (datacenter constraints)
- Cooling requirements (thermal budget)

**Note on Scoring Strategy**:

- All four factors scored equally (0-100) in current implementation
- **Balanced composite** uses weighted combination with higher weights on
  primary factors (accuracy, cost)
- Future may formalize Primary/Secondary distinction with explicit weight tiers

#### c. Analyzer

**Role**: Rank solutions and present trade-offs

**Processing**:

1. Generate **five ranked views** of recommendations:
   - **Best Accuracy**: Sorted by quality/capability
   - **Lowest Cost**: Sorted by price efficiency
   - **Lowest Latency**: Sorted by SLO headroom
   - **Simplest**: Sorted by deployment complexity (fewest GPUs)
   - **Balanced**: Sorted by weighted composite score (user priorities)

2. Provide detailed rationale for each recommendation:
   - Why this configuration was selected
   - Trade-offs vs. alternatives
   - SLO compliance margin

**Output**: Ranked list of deployment recommendations with:

- Model + GPU configuration details
- Predicted performance metrics
- Cost estimates (hourly, monthly)
- Quality/complexity/latency scores
- Trade-off analysis vs. alternatives

**Interface**: REST endpoint `/api/v1/ranked-recommend-from-spec`

---

### 5. Configuration Service

```text
┌───────────────────────────────────────────────────────────────────┐
│                      Configuration Service                        │
│                                                                   │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐  │
│  │    Template     │   │    Renderer     │   │    Validator    │  │
│  │    Library      │   │                 │   │                 │  │
│  │                 │   │                 │   │                 │  │
│  │ KServe, vLLM,   │──▶│ Jinja2 template │──▶│ Resource limits,│  │
│  │ HPA, Service-   │   │ processing      │   │ security checks │  │
│  │ Monitor         │   │                 │   │                 │  │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘  │
│                                                                   │
│  Selected Recommendation ───────────────────────▶ YAML Files      │
└───────────────────────────────────────────────────────────────────┘
```

**Purpose**: Generate production-ready Kubernetes configuration files

**Input**: User-selected recommendation from UI

**Processing**:

- Generate Kubernetes manifests using Jinja2 templates:
  - KServe InferenceService YAML
  - vLLM runtime configuration
  - Horizontal Pod Autoscaler (HPA) policies
  - Prometheus ServiceMonitor for observability
- Validate resource requests and security settings

**Output**:

- Configuration files ready for user to deploy via `kubectl apply -f`
- Files available for download or copy from UI

**Interface**: REST endpoint `/api/v1/deploy`

**Initial Target**: KServe on Kubernetes **Future Options**: Direct Linux server
deployment, other orchestration platforms

---

### Infrastructure Components

These are critical infrastructure that enable the services but are not domain
components:

#### API Gateway

**Purpose**: Coordinate multi-service workflows and provide REST API

**Technology**: FastAPI

**Responsibilities**:

- Expose HTTP endpoints for UI to call services
- Orchestrate multi-step flows (conversation → recommendation → deployment)
- Manage workflow state (session-based currently, persistent store future)
- Handle authentication and authorization (future)

#### Knowledge Base (Data Layer)

**Purpose**: Store performance data, configuration, and deployment history

**Technology**: Hybrid storage approach

- **Embedded Database**: Benchmark data and deployment outcomes (currently
  PostgreSQL, migrating to DuckDB or SQLite for simpler deployment)
- **JSON files**: Configuration as code (SLO templates, model catalog, hardware
  profiles)

**Collections**:

- **Performance Benchmarks**: Performance metrics for (model, GPU, traffic
  profile) combinations
- **Accuracy Benchmarks**: Accuracy metrics for multiple use cases
- **Use Case SLO Templates**: Default targets for 9 standard use cases
- **Model Catalog**: Curated models with task compatibility metadata
- **Hardware Profiles**: GPU specifications and pricing
- **Deployment Outcomes** (future): Actual performance data for continuous
  learning

---

## Data Flow

### Primary Path: Conversational Requirements (Current)

```
1. User Message
     ↓
2. Intent Extraction Service
   - Natural language → structured intent
   - Extract: use case, user count, priorities, domain
     ↓
3. Specification Service
   - Intent → traffic profile (prompt/output tokens, QPS)
   - Use case → SLO targets (TTFT, ITL, E2E)
   - Select accuracy benchmarks
     ↓
4. UI Specification Editor
   - Display auto-generated specification
   - User reviews and edits (SLOs, priorities, constraints)
   - Obtain approval
     ↓
5. Recommendation Service
   a. Config Finder: Query benchmarks, calculate replicas
   b. Scorer: Evaluate on 4 dimensions (accuracy, cost, latency, complexity)
   c. Analyzer: Generate 5 ranked views
     ↓
6. UI Recommendation Viewer
   - Display ranked options (Best Accuracy, Lowest Cost, etc.)
   - Show trade-off analysis
   - User selects preferred configuration
     ↓
7. Configuration Service
   - Generate Kubernetes YAML manifests
   - User downloads or copies configuration files
   - User deploys to cluster via kubectl apply -f
```

### Alternate Path: Telemetry-Based Optimization (Future)

```
1. Deployed Service
   - Collect actual performance metrics (latency, throughput, traffic patterns)
   - Store telemetry data
     ↓
2. Telemetry UI (Future Component)
   - Load and display deployment telemetry
   - Identify optimization opportunities
   - User triggers re-evaluation
     ↓
3. Intent Extraction Service (Alternate API)
   - Telemetry data → structured intent
   - Extract actual traffic patterns, observed latencies, real QPS
     ↓
4. Specification Service
   - Generate specification based on actual observed behavior
   - More accurate than initial estimates
     ↓
5. [Joins Primary Path]
   - Recommendation Service → UI Recommendation Viewer → Configuration Service
   - User can compare current deployment vs. optimized recommendations
```

**Rationale for Telemetry Path**:

- Initial conversational path relies on assumptions (use case, request rate,
  traffic profile)
- Real deployments may evolve over time (user behavior changes, workload
  patterns shift)
- Telemetry data provides ground truth for optimization
- Enables continuous improvement: deploy → measure → optimize → redeploy

**Design Note**: Telemetry path is an **alternate entry point** into the system.
Both paths converge at the Specification Service and share the same
recommendation and deployment workflows.

---

## Future Enhancements

### User Interface

- **React Migration**: Professional frontend with improved state management,
  WebSocket-based real-time updates
- **Advanced Visualizations**:
  - Pareto frontier charts for two-dimensional trade-offs
  - Multi-dimensional radar charts (>2 dimensions)
  - Interactive cost/latency sliders with live recommendation updates

### Telemetry Integration

- **Telemetry UI**: Interface for loading and analyzing actual deployment
  metrics
- **Observability Feedback Loop**: Store actual performance → improve future
  recommendations
- **Drift Detection**: Alert when deployed service deviates from predicted
  behavior

### Specification Service

- **Power/Cooling Budgets**: Datacenter constraint modeling (power consumption,
  thermal limits)
- **Statistical Traffic Distributions**: Full distributions (mean, variance,
  percentiles) instead of point estimates
- **Advanced SLO Types**: Streaming latency, first-chunk time, custom
  user-perceived metrics

### Recommendation Service

- **Simulator-Based Planning**: Generate recommendations without real benchmark
  data (analytical models, discrete event simulation)
- **Multi-Model Deployments**: Ensemble recommendations, A/B testing
  configurations, multi-model systems on heterogeneous hardware
- **Agentic System Support**: Recommendations for agent architectures with
  multiple coordinated models
- **Fine-Tuning Integration**: Suggest when custom fine-tuning improves quality
  vs. larger generic models

### Configuration Service

- **Additional Runtimes**: Support for llm-d and other inference engines beyond
  vLLM
- **Multi-Platform Support**: Configuration templates for direct Linux server,
  cloud-managed services, edge deployments
- **Advanced Kubernetes Features**: Templates for custom operators, advanced
  autoscaling (KEDA), multi-cluster configurations
- **External Access Patterns**: Ingress/Route generation, TLS certificate
  configuration

### Knowledge Base

- **Bring Your Own Data**: Allow users to upload custom performance and/or
  accuracy benchmarks to support new models, hardware, and traffic profiles

---

## Extensibility

NeuralNav is designed to support pluggable components, allowing customization for
different environments and requirements.

### Pluggable LLM Providers

- Support multiple LLM backends for intent extraction and other LLM tasks
- Options: Local models (Ollama), cloud APIs (OpenAI, Anthropic), user-provided
  endpoints
- Standard interface: OpenAI-compatible API for interoperability

### Configurable Data Sources

- **Model Catalog**: User-provided or curated model lists
- **Benchmarks**: Custom performance and accuracy data (see "Bring Your Own
  Data" above)
- **SLO Templates**: Custom use case definitions with organization-specific
  defaults

### Hardware and Cost Data

- GPU specifications, pricing, power consumption, and cooling requirements
- User-provided data for proprietary or on-premises hardware
- Integration with external data sources for cloud pricing updates

---

## Key Design Principles

1. **Conversational-First**: All complexity hidden behind natural language (no
   forms or wizards)
2. **Specification Editability**: Users can review and modify auto-generated
   specs before committing
3. **Multi-Criteria Optimization**: Balanced recommendations across accuracy,
   cost, latency, and complexity
4. **SLO-Driven Planning**: Recommendations guaranteed to meet user's latency
   and throughput requirements
5. **Explainability**: Clear rationale for why each recommendation was made,
   trade-off analysis
6. **Threshold-Based Filtering**: Editable accuracy and cost thresholds ensure
   all options meet minimum quality and budget requirements
7. **Ready-to-Deploy Configurations**: Generate complete configuration files,
   minimizing manual YAML editing

---

## Glossary

| Term | Definition |
|------|------------|
| **E2E** | End-to-End latency - total time from request submission to complete response |
| **HPA** | Horizontal Pod Autoscaler - Kubernetes component that scales pods based on metrics |
| **InferenceService** | KServe custom resource for deploying ML models on Kubernetes |
| **ITL** | Inter-Token Latency - time between successive tokens in streaming response |
| **KServe** | Kubernetes-native platform for serving ML models |
| **p90/p95/p99** | Percentile thresholds - the value below which 90%/95%/99% of observations fall (selectable in specification) |
| **QPS** | Queries Per Second - request throughput rate |
| **RPS** | Requests Per Second - synonymous with QPS |
| **SLO** | Service Level Objective - target performance threshold (e.g., "TTFT p95 < 200ms") |
| **Tensor Parallelism** | Splitting a model across multiple GPUs to handle larger models or increase throughput |
| **Traffic Profile** | Expected workload shape: input token count, output token count, request rate |
| **TTFT** | Time to First Token - latency from request to first token of streaming response |
| **vLLM** | High-performance LLM inference engine with PagedAttention |

---

## Notes

This architecture document describes the **production vision** for NeuralNav.
The current POC implementation has successfully validated the core workflow and
demonstrates the viability of the conversational approach to LLM deployment
planning.

The separation of Intent Extraction Service and Specification Service reflects
the **logical architecture** - these are distinct conceptual steps with
different responsibilities. Current implementation combines them for simplicity,
but the interface boundaries are preserved to enable future separation if
needed.

Detailed implementation specifics and data schemas will be similar to those
described in [ARCHITECTURE.md](./ARCHITECTURE.md), but some changes may be made
and described in future versions of this document.

### Topics for Future Versions

The following areas are not addressed in this version but may be considered for
future iterations:

- **Security** - Authentication, authorization, and data privacy considerations
- **Error Handling** - Failure modes and recovery strategies (e.g., LLM
  extraction failures, no matching configurations)
- **NeuralNav Deployment** - How NeuralNav itself is deployed and scaled
- **Multi-Tenancy** - User/organization isolation and separate knowledge bases
- **Data Versioning** - Strategy for updating model catalogs, benchmarks, and
  SLO templates over time
- **Non-Functional Requirements** - System response times, availability targets,
  and operational characteristics
