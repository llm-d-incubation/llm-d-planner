# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the architecture design for **Compass**, an open-source system that guides users from concept to production-ready LLM deployments through a conversational AI and intelligent capacity planning.

**Key Principle**: This is a **Phase 1 POC implementation** based on the architecture design. The core functionality is complete and working end-to-end.

## Repository Structure

- **docs/ARCHITECTURE.md**: Comprehensive system architecture document
  - 9 core components with technology recommendations
  - Enhanced data schemas for SLO-driven deployment planning
  - Phase 1 (3-month) vs Phase 2+ implementation strategy
  - Knowledge Base schemas with 7 data collections

- **docs/architecture-diagram.md**: Visual architecture representations
  - Mermaid component diagrams
  - Sequence diagrams showing end-to-end flows
  - State machine for workflow orchestration
  - Entity-relationship diagrams for data models

- **backend/**: Python backend implementation
  - **api/**: FastAPI REST endpoints with CORS support
  - **context_intent/**: Intent extraction, traffic profiles, Pydantic schemas
  - **recommendation/**: Multi-criteria scoring and ranking
    - `solution_scorer.py`: 4-dimension scoring (accuracy, price, latency, complexity)
    - `model_evaluator.py`: Use-case fit scoring
    - `usecase_quality_scorer.py`: Artificial Analysis benchmark integration
    - `ranking_service.py`: 5 ranked list generation
    - `capacity_planner.py`: GPU capacity planning with SLO filtering
  - **knowledge_base/**: Data access (benchmark database, JSON catalogs)
  - **orchestration/**: Workflow coordination
  - **deployment/**: Jinja2 templates for KServe/vLLM YAML generation
  - **llm/**: Ollama client for intent extraction

- **ui/**: Streamlit UI
  - Chat interface for conversational requirement gathering
  - Multi-tab recommendation display (Overview, Specifications, Performance, Cost, Monitoring)
  - Editable specifications with review mode
  - Action buttons for YAML generation and deployment
  - Monitoring dashboard with cluster status, SLO compliance, and inference testing

- **data/**: Benchmark and catalog data
  - **model_catalog.json**: 47 curated models with task/domain metadata
  - **slo_templates.json**: 9 use case templates with SLO targets
  - **benchmarks/models/**: Model benchmark data
    - `opensource_all_benchmarks.csv`: 204 open-source models from Artificial Analysis
    - `model_pricing.csv`: GPU pricing data
  - **business_context/use_case/**: Use-case specific quality scoring
    - `weighted_scores/`: 9 CSV files with pre-ranked models per use case
    - `configs/`: Use case configuration files (weights, SLOs, workloads)
    - `USE_CASE_METHODOLOGY.md`: Explains benchmark weighting strategy
  - **benchmarks_BLIS.json**: Latency/throughput benchmarks from BLIS simulator (loaded into PostgreSQL)
  - **demo_scenarios.json**: 3 test scenarios

## Architecture Key Concepts

### Problem Being Solved
Deploying LLMs in production is complex - users struggle to:
- Translate business needs into infrastructure choices (model, GPU type, SLO targets)
- Avoid trial-and-error that wastes time and money
- Understand resource requirements before committing to expensive GPU deployments

### Solution Approach
A **4-stage conversational flow**:
1. **Understand Business Context** - Extract intent via natural language
2. **Provide Tailored Recommendations** - Suggest model + GPU configurations
3. **Enable Interactive Exploration** - What-if scenario analysis
4. **One-Click Deployment** - Generate KServe/vLLM configs and deploy

### Core Innovation: SLO-Driven Capacity Planning
The system translates high-level user intent into technical specifications:
- **User says**: "I need a chatbot for 1000 users, low latency is critical"
- **System generates**:
  - Traffic profile (prompt: 512 tokens, output: 256 tokens, expected QPS: 9)
  - SLO targets (TTFT p95: 150ms, ITL p95: 25ms, E2E p95: 7000ms)
  - GPU capacity plan (e.g., "1x NVIDIA H100 GPU, independent replicas")
  - Cost estimate ($5,840/month)

### Architecture Overview

Compass is structured as a layered architecture:

**UI Layer** (Horizontal - Presentation):
- **Conversational Interface, Specification Editor, Recommendation Visualizer, Monitoring Dashboard**
- Technology: Streamlit (current) → React (future)

**Core Engines** (Vertical - Backend Services):
1. **Intent & Specification Engine** - Transform conversation into complete deployment spec
   - LLM-powered intent extraction (Ollama qwen2.5:7b)
   - Use case → traffic profile mapping (4 GuideLLM standards)
   - SLO template lookup and specification generation
2. **Recommendation Engine** - Find optimal model + GPU configurations
   - Multi-criteria scoring (accuracy, price, latency, complexity)
   - Capacity planning (GPU count, deployment topology)
   - SLO compliance filtering with near-miss tolerance
   - Ranked lists generation (5 views: best accuracy, lowest cost, etc.)
3. **Deployment Engine** - Generate and deploy Kubernetes configs
   - YAML generation (Jinja2 templates)
   - K8s deployment lifecycle management
4. **Observability Engine** - Monitor deployed services
   - Health monitoring and inference testing (current)
   - Performance tracking and feedback loop (future)

**Infrastructure** (Not numbered as core engines):
- **API Gateway** (FastAPI) - Coordinates workflow between UI and engines
- **Knowledge Base** (Data Layer) - Hybrid storage:
  - PostgreSQL: Benchmarks, deployment outcomes
  - JSON files: SLO templates, model catalog, hardware profiles

**Development Tools:**
- **vLLM Simulator** - GPU-free development and testing

### Critical Data Collections (Knowledge Base)
- **Model Benchmarks** (PostgreSQL): TTFT/ITL/E2E/throughput benchmarks for (model, GPU, tensor_parallel) combinations (source: BLIS simulator)
- **Use Case SLO Templates** (JSON): 9 use cases mapped to 4 GuideLLM traffic profiles with experience-driven SLO targets
- **Model Catalog** (JSON): 47 curated, approved models with task/domain metadata
- **Model Quality Scores** (CSV): Use-case specific scores from Artificial Analysis benchmarks (204 models)
- **Use Case Configs** (JSON): Benchmark weights, SLO targets, and workload profiles per use case
- **Deployment Outcomes** (PostgreSQL, future): Actual performance data for feedback loop

### Solution Ranking System

The recommendation engine uses **multi-criteria scoring** to rank configurations:

**4 Scoring Dimensions** (each 0-100 scale):
1. **Accuracy/Quality**: Use-case specific model capability from Artificial Analysis benchmarks
   - Source: `data/business_context/use_case/weighted_scores/*.csv`
   - Fallback: Parameter count heuristic if model not in benchmark data
2. **Price**: Cost efficiency (inverse of monthly cost, normalized)
3. **Latency**: SLO compliance and headroom from performance benchmark database
4. **Complexity**: Deployment simplicity (fewer GPUs = higher score)

**Default Weights**: 40% accuracy, 40% price, 10% latency, 10% complexity

**5 Ranked Views**:
- `best_accuracy`: Sorted by model capability
- `lowest_cost`: Sorted by price efficiency
- `lowest_latency`: Sorted by SLO headroom
- `simplest`: Sorted by deployment complexity
- `balanced`: Sorted by weighted composite score

**Key Files**:
- `backend/src/recommendation/solution_scorer.py` - Calculates 4 scores
- `backend/src/recommendation/model_evaluator.py` - Legacy accuracy scoring (use-case fit)
- `backend/src/recommendation/usecase_quality_scorer.py` - Artificial Analysis benchmark scoring
- `backend/src/recommendation/ranking_service.py` - Generates 5 ranked lists
- `backend/src/recommendation/capacity_planner.py` - Orchestrates scoring during capacity planning

## Working with This Repository

### When Modifying Architecture Documents

**docs/ARCHITECTURE.md and docs/architecture-diagram.md must stay synchronized**:
- If you change component descriptions in ARCHITECTURE.md, update architecture-diagram.md diagrams
- If you add/remove components, update both files
- Components are referenced by name (not numbered) for clarity and flexibility

### Key Architectural Decisions to Preserve

1. **Phase 1 uses Python** for all components (rapid development, stack consistency)
   - Go migration for Deployment Automation Engine is a possible future option (see Possible Future Enhancements in ARCHITECTURE.md)

2. **Phase 1 uses point estimates** for traffic (avg prompt length, avg QPS)
   - Benchmarks collected using vLLM default configuration (dynamic batching enabled)
   - Phase 2 adds full statistical distributions (mean, variance, tail) and multi-dimensional benchmarks

3. **SLO metrics use p95 percentiles** (Phase 2):
   - TTFT (Time to First Token): p95 - **pre-calculated in benchmarks**
   - ITL (Inter-Token Latency): p95 - **pre-calculated in benchmarks** (replaces TPOT terminology)
   - E2E Latency: p95 - **pre-calculated in benchmarks** from actual measurements
   - Throughput: requests/sec and tokens/sec
   - Rationale:
     - p95 is more conservative than p90, providing better UX guarantees
     - E2E latency is measured directly from benchmarks under realistic load conditions
     - Benchmarks are organized around 4 GuideLLM traffic profiles for exact matching

4. **Editable specifications**: Users must be able to review and modify auto-generated specs before deployment

5. **Feedback loop**: Actual deployment outcomes feed back into Knowledge Base to improve future recommendations

### Terminology Consistency

- Use "**Compass**" as the project name
- Use "**TTFT**" for Time to First Token (not "time-to-first-token")
- Use "**ITL**" for Inter-Token Latency (Phase 2 terminology, replaces TPOT)
- Use "**SLO**" for Service Level Objective
- Use "**E2E**" for End-to-End latency
- Use "**p95**" for 95th percentile metrics (Phase 2 standard, more conservative than p90)
- GPU configurations: "2x NVIDIA L4" or "4x A100-80GB" (not "2 L4s")

### Common Editing Patterns

**Adding a new use case template**:
1. Add corresponding entry to `data/slo_templates.json`
2. Create weighted scores CSV in `data/business_context/use_case/weighted_scores/`
3. Add use case to `UseCaseQualityScorer.USE_CASE_FILES` in `usecase_quality_scorer.py`
4. Update `USE_CASE_METHODOLOGY.md` with benchmark weighting rationale
5. Update docs/ARCHITECTURE.md if needed

**Adding a new SLO metric**:
1. Update DeploymentIntent schema in Intent & Specification Engine (docs/ARCHITECTURE.md)
2. Update MODEL_BENCHMARKS schema in Knowledge Base (docs/ARCHITECTURE.md)
3. Update PostgreSQL schema in scripts/schema.sql
4. Update data loader script if needed
4. Update Inference Observability section
5. Update dashboard example if applicable
6. Update docs/architecture-diagram.md data model ERD

**Adding a new component**:
1. Add numbered section to docs/ARCHITECTURE.md (maintain sequential numbering)
2. Update "Architecture Components" count in Overview
3. Add to docs/architecture-diagram.md component diagram
4. Create corresponding backend/src/<component>/ directory
5. Update sequence diagram if component participates in main flow
6. Update Phase 1 technology choices table if relevant

## Open Questions and Future Work

See "Open Questions for Refinement" section in docs/ARCHITECTURE.md for:
- Multi-tenancy isolation
- Security validation of generated configs
- Conversational clarification flow (future phase)
- Model catalog sync strategy

## Git Workflow

This repository uses a **pull request (PR) workflow**. See [CONTRIBUTING.md](CONTRIBUTING.md) for complete guidelines.

### Quick Summary

**Development Process**:
- Work in feature branches in your own fork
- Submit PRs to the main repository for review
- Keep PRs small and targeted (under 500 lines when possible)
- Break large features into incremental PRs that preserve functionality

**Commit Message Format** (Conventional Commits style):

```
feat: Add YAML generation module

Implement DeploymentGenerator with Jinja2 templates for KServe,
vLLM, HPA, and ServiceMonitor configurations.

Assisted-by: Claude <noreply@anthropic.com>
Signed-off-by: Your Name <your.email@example.com>
```

**Key Requirements**:
- Use conventional commit types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`
- Include DCO Signed-off-by line (use `git commit -s`)
- Include `Assisted-by: Claude <noreply@anthropic.com>` for nontrivial AI-assisted code
- **DO NOT include** the "🤖 Generated with [Claude Code]" line in commit messages

## Important Notes

- **Current Implementation Status**:
  - ✅ Project structure with synthetic data and LLM client
  - ✅ Core recommendation engine (intent extraction, traffic profiling, capacity planning)
  - ✅ Multi-criteria solution ranking with 4 scoring dimensions
  - ✅ Use-case specific quality scoring from Artificial Analysis benchmarks
  - ✅ 5 ranked recommendation views (best accuracy, lowest cost, etc.)
  - ✅ Orchestration workflow and FastAPI backend
  - ✅ Streamlit UI with chat interface, recommendation display, and editable specifications
  - ✅ YAML generation (KServe/vLLM/HPA/ServiceMonitor) and deployment automation
  - ✅ KIND cluster support with KServe installation
  - ✅ Kubernetes deployment automation and real cluster status monitoring
  - ✅ vLLM simulator for GPU-free development
  - ✅ Inference testing UI with end-to-end deployment validation
- The Knowledge Base schemas are critical - any implementation must support all collections
- SLO-driven capacity planning is the core differentiator - don't simplify this away
- Use data in data/ directory for POC; production uses PostgreSQL for latency benchmarks
- Benchmarks use vLLM default configuration with dynamic batching (no fixed batch_size)

## Simulator Mode vs Real vLLM

The system now supports two deployment modes:

### Simulator Mode (Default for Development)
- **Purpose**: GPU-free development and testing on local machines
- **Location**: `simulator/` directory contains the vLLM simulator service
- **Docker Image**: `vllm-simulator:latest` (single image for all models)
- **Configuration**: Set `DeploymentGenerator(simulator_mode=True)` in `backend/src/api/routes.py`
- **Benefits**:
  - No GPU hardware required
  - Fast deployment (~10-15 seconds to Ready)
  - Predictable behavior for demos
  - Works on KIND (Kubernetes in Docker)
  - Uses actual benchmark data for realistic latency simulation

### Real vLLM Mode (Production)
- **Purpose**: Actual model inference with GPUs
- **Configuration**: Set `DeploymentGenerator(simulator_mode=False)` in `backend/src/api/routes.py`
- **Requirements**:
  - GPU-enabled Kubernetes cluster
  - NVIDIA GPU Operator installed
  - HuggingFace token secret for model downloads
  - Sufficient GPU resources (based on recommendations)
- **Behavior**:
  - Downloads actual models from HuggingFace
  - Real GPU inference
  - Production-grade performance

### When to Use Each Mode

**Use Simulator Mode for:**
- Local development and testing
- UI/UX iteration
- Workflow validation
- Demos and presentations
- CI/CD testing (no GPU required)

**Use Real vLLM Mode for:**
- Production deployments
- Performance benchmarking
- Model quality validation
- GPU utilization testing

### Technical Details

The deployment template (`backend/src/deployment/templates/kserve-inferenceservice.yaml.j2`) uses Jinja2 conditionals:
- `{% if simulator_mode %}` - Uses `vllm-simulator:latest`, no GPU resources, fast health checks
- `{% else %}` - Uses `vllm/vllm-openai:v0.6.2`, requests GPUs, longer health checks

Single codebase supports both modes - just toggle the flag!
