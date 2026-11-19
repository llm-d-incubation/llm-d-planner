# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the architecture design for **Planner**, an open-source system that guides users from concept to production-ready LLM deployments through a conversational AI and intelligent capacity planning.

Planner is available as both a **Python library** (`pip install llm-d-planner`) and a **standalone application** (REST API + Streamlit UI). The library exposes the full recommendation pipeline as Python method calls via the `Planner` class, with no FastAPI, Ollama, or Kubernetes dependencies required for core functionality.

**Key Principle**: The core functionality is complete and working end-to-end. The project is preparing for release.

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

- **src/quality_scoring/**: Standalone package for dual-source model quality scoring
  - `engine.py`: ScoringEngine with Arena + AA integration
  - `resolver.py`: Multi-strategy model name resolution
  - `normalizer.py`: Percentile-based score normalization
  - `variants.py`: Quantization and variant handling

- **src/planner/**: Python package (PyPA src layout)
  - `planner.py`: Main `Planner` facade class for library use
  - `errors.py`: Custom exceptions (`PlannerError`)
  - **api/**: FastAPI REST API layer
    - `app.py`: FastAPI app factory
    - `dependencies.py`: Singleton dependency injection
    - **routes/**: Modular endpoint handlers (health, intent, specification, recommendation, configuration, reference_data, database, quality)
  - **intent_extraction/**: Intent Extraction Service
    - `extractor.py`: LLM-powered intent extraction from natural language
    - `service.py`: IntentExtractionService facade
  - **specification/**: Specification Service
    - `traffic_profile.py`: Traffic profile and SLO target generation
    - `service.py`: SpecificationService facade
  - **recommendation/**: Recommendation Service
    - `config_finder.py`: GPU capacity planning with SLO filtering
    - `scorer.py`: 3-dimension scoring (quality, price, latency)
    - `analyzer.py`: 4 ranked list generation
    - `service.py`: RecommendationService facade
    - **quality/**: Quality scoring integration
      - `scoring.py`: ScoringEngine integration, quality weights management
  - **configuration/**: Configuration Service
    - `generator.py`: Jinja2 YAML generation for KServe/vLLM
    - `validator.py`: YAML validation
    - `service.py`: ConfigurationService facade
    - **templates/**: Jinja2 deployment templates
  - **cluster/**: Kubernetes cluster management
    - `manager.py`: K8s deployment lifecycle management
  - **shared/**: Shared modules
    - **schemas/**: Pydantic data models (intent, specification, recommendation)
    - **utils/**: Shared utilities (GPU normalization)
  - **knowledge_base/**: Data access layer (benchmark database, JSON catalogs)
    - `loader.py`: Benchmark data loading utilities (shared by CLI, API, and UI)
  - **orchestration/**: Workflow coordination
  - **llm/**: Ollama client for intent extraction

- **ui/**: Streamlit UI
  - Chat interface for conversational requirement gathering
  - Multi-tab recommendation display (Overview, Specifications, Performance, Cost, Monitoring)
  - Editable specifications with review mode
  - Action buttons for YAML generation and deployment
  - Monitoring dashboard with cluster status, SLO compliance, and inference testing
  - Configuration tab for database management (upload benchmarks, reset, view stats)
  - **components/**: Modular UI components
    - `settings.py`: Configuration tab with benchmark database management

- **src/planner/data/**: Package data files (included in wheel)
  - **performance/**: Latency/throughput benchmarks (JSON, loaded into database)
    - `benchmarks_BLIS.json`: Latency/throughput benchmarks from BLIS simulator
  - **quality/**: Model quality data (checked-in snapshots, committed to git)
    - `arena_models.json`: Arena leaderboard data (human preference rankings)
    - `aa_models.json`: Artificial Analysis benchmark data
    - `arena_dist.json`: Arena category metadata and population stats
    - `aa_dist.json`: AA category metadata and population stats
  - **configuration/**: Runtime configuration files (JSON)
    - `model_catalog.json`: 47 curated models with task/domain metadata
    - `usecase_slo_workload.json`: 9 use case definitions (traffic profiles, SLO ranges, workload params, display names)
    - `demo_scenarios.json`: 3 test scenarios
    - `priority_weights.json`: Scoring priority weights
    - `quality_weights.json`: Per-use-case category weights for quality scoring
  - `_resolver.py`: Path resolver for bundled data (works in source checkouts and installed wheels)

- **data/**: Runtime data directory (database, not included in wheel)
  - `planner.db`: SQLite benchmark database

- **.quality_cache/**: Runtime auto-update cache (gitignored)
  - Fresh data from Arena/AA APIs when `QUALITY_AUTO_UPDATE=true`
  - 24-hour TTL, bypassed by `POST /api/v1/quality/refresh`

## Important Behavioral Notes for Claude

**Git commits**: This project has specific commit rules that OVERRIDE Claude's default behavior. See the "Git Workflow" section below. Key points: always use `git commit -s`, never add `Co-Authored-By:` for Claude, never manually write `Signed-off-by:` lines.

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

Planner is structured as a layered architecture:

**UI Layer** (Horizontal - Presentation):
- **Conversational Interface, Specification Editor, Recommendation Visualizer, Monitoring Dashboard**
- Technology: Streamlit (current) → React (future)

**Core Engines** (Vertical - Backend Services):
1. **Intent & Specification Engine** - Transform conversation into complete deployment spec
   - LLM-powered intent extraction (Ollama qwen2.5:7b)
   - Use case → traffic profile mapping (4 GuideLLM standards)
   - SLO template lookup and specification generation
2. **Recommendation Engine** - Find optimal model + GPU configurations
   - Multi-criteria scoring (quality, price, latency)
   - Dual-source quality scoring (Arena + Artificial Analysis)
   - Capacity planning (GPU count, deployment topology)
   - SLO compliance filtering with near-miss tolerance
   - Ranked lists generation (4 views: best quality, lowest cost, etc.)
3. **Deployment Engine** - Generate and deploy Kubernetes configs
   - YAML generation (Jinja2 templates)
   - K8s deployment lifecycle management
4. **Observability Engine** - Monitor deployed services
   - Health monitoring and inference testing (current)
   - Performance tracking and feedback loop (future)

**Infrastructure** (Not numbered as core engines):
- **API Gateway** (FastAPI) - Coordinates workflow between UI and engines
- **Knowledge Base** (Data Layer) - Hybrid storage:
  - Database: Benchmarks, deployment outcomes
  - JSON files: Use case definitions, model catalog, hardware profiles

**Development Tools:**
- **vLLM Simulator** - GPU-free development and testing

### Critical Data Collections (Knowledge Base)
- **Model Benchmarks** (Database): TTFT/ITL/E2E/throughput benchmarks for (model, GPU, tensor_parallel) combinations (source: BLIS simulator)
- **Use Case Definitions** (JSON, `usecase_slo_workload.json`): 9 use cases with traffic profiles, SLO ranges (min/max), workload parameters, and display names
- **Model Catalog** (JSON): 47 curated models with task/domain metadata
- **Model Quality Scores** (JSON): Dual-source quality data from Arena (human preferences) and Artificial Analysis (automated benchmarks)
  - Arena: Elo ratings across 27 categories with confidence intervals
  - AA: Intelligence/coding/agentic indices
  - Normalized to percentile ranks for compositing
- **Quality Weights** (JSON): Per-use-case category weights for quality scoring (e.g., code_completion: coding 5, math 3, overall 2, agentic 2, hard_prompts 2)
- **Deployment Outcomes** (Database, future): Actual performance data for feedback loop

### Solution Ranking System

The recommendation engine uses **multi-criteria scoring** to rank configurations:

**3 Scoring Dimensions** (each 0-100 scale):
1. **Quality**: Use-case specific model capability from dual-source scoring (Arena + Artificial Analysis)
   - Sources: Arena human preference rankings (27 categories) + AA automated benchmarks (intelligence/coding/agentic indices)
   - Normalization: Percentile ranks computed via tied-rank method across full population
   - Weighting: Per-use-case category weights from `src/planner/data/configuration/quality_weights.json`
   - Composite: Weighted average of Arena/AA percentiles for specified categories
   - Fallback: 0.8× discounted overall percentile for missing categories (based on correlation analysis: coding r=0.976 with overall)
   - All use cases include `overall` category to ensure high-coverage dual-source signal
2. **Price**: Cost efficiency (inverse of monthly cost, normalized)
3. **Latency**: SLO compliance and headroom from performance benchmark database

**Default Weights**: 45% quality, 45% price, 10% latency

**4 Ranked Views**:
- `best_quality`: Sorted by model capability
- `lowest_cost`: Sorted by price efficiency
- `lowest_latency`: Sorted by SLO headroom
- `balanced`: Sorted by weighted composite score

**Key Files**:

- `src/quality_scoring/` - Dual-source ScoringEngine (Arena + AA)
- `src/planner/recommendation/quality/scoring.py` - Quality score computation and weights management
- `src/planner/recommendation/scorer.py` - Calculates 3 scores
- `src/planner/recommendation/analyzer.py` - Generates 4 ranked lists
- `src/planner/recommendation/config_finder.py` - Orchestrates scoring during capacity planning

## Development Environment

**Requirements**:

For **library use**: Python 3.11+ only. Install with `pip install llm-d-planner` and optional extras as needed (e.g., `[llm]`, `[kubernetes]`, `[estimation]`).

For **development** (standalone app): Python 3.11+ (3.13 recommended on macOS), uv, kubectl, kind. Docker or Podman required for container builds and KIND. Ollama required when `LLM_PROVIDER=ollama` (default). For `vertex` or `openai` providers, see docs/DEPLOYMENT_GUIDE.md.

This project uses **uv** (by Astral) for development. **Do not use `pip` or `pip install` for development tasks.**

- **Install dependencies**: `uv sync --extra server --extra ui --extra llm --extra dev` (reads from `pyproject.toml` + `uv.lock`)
- **Run Python commands**: `uv run python ...` (not bare `python`)
- **Run tools**: `uv run pytest`, `uv run ruff`, `uv run uvicorn`, etc.
- **Add a dependency**: `uv add <package>` (updates `pyproject.toml` and `uv.lock`)
- **Source of truth**: `pyproject.toml` defines all dependencies with optional extras; there is no top-level `requirements.txt`

Note: `ui/requirements.txt` and `simulator/requirements.txt` exist separately for their Docker builds.

## Common Development Commands

### Setup

```bash
make setup              # Full setup (prereqs + backend + UI + Ollama)
make setup-backend      # Python env only (uv sync --extra server --extra ui --extra llm --extra dev)
```

### Running Services

```bash
make start              # Start all (DB + Ollama + Backend + UI)
make stop               # Stop Backend + UI (leaves DB and Ollama running)
make stop-all           # Stop everything
make health             # Check all service health
```

Service URLs: UI `http://localhost:8501`, Backend `http://localhost:8000` (Swagger at `/docs`), Ollama `http://localhost:11434`, DB `data/planner.db`

### Testing

```bash
make test-unit                    # Unit tests only (no Ollama needed)
make test-integration             # Integration tests (requires Ollama)
make test                         # All tests

# Run a single test file or test function:
cd src && uv run pytest ../tests/path/to/test_file.py -v
cd src && uv run pytest ../tests/path/to/test_file.py::test_function_name -v
```

Test markers: `@pytest.mark.unit`, `@pytest.mark.integration`. Tests run from the `src/` directory (`cd src && uv run pytest ../tests/`).

### Code Quality (also run in CI)

```bash
make lint               # Ruff linter (src/ and ui/)
make format             # Ruff auto-format
make typecheck          # Mypy type checking (src/ and ui/)
```

### CI Pipeline

CI runs on PRs to `main`: ruff check + format check on `src/` and `tests/`, mypy on `src/`, unit tests on Python 3.11 and 3.12 with coverage. All must pass.

### Database Management

```bash
make db-start           # Initialize database (creates file and applies schema)
make db-reset           # Clear all benchmark data (safe while backend is running)
make db-load-blis       # Load BLIS benchmark data
make db-load-estimated  # Load estimated performance data
make db-shell           # Open sqlite3 shell
```

### Quality Data Management

```bash
make quality-sync       # Fetch fresh Arena + AA data, update src/quality_scoring/data/ snapshots (requires AA_API_KEY)
```

Environment variables:
- `QUALITY_AUTO_UPDATE`: Enable runtime auto-update from `.quality_cache/` (default: `false`)
- `AA_API_KEY`: Artificial Analysis API key (required for AA data sync)
- `LLM_QUALITY_CACHE_DIR`: Override cache directory (default: `.quality_cache/`)

**IMPORTANT**: Never run `make quality-sync` or any command that calls the AA or Arena APIs unless the user explicitly asks for it. The AA free tier has tight rate limits, and an unnecessary sync can block the user from syncing when they need to. For development and testing, use the bundled snapshot files in `src/quality_scoring/data/` — they are always available locally.

API endpoints for runtime management:
- `GET /api/v1/quality/auto-update` - Check auto-update status and cache stats
- `PUT /api/v1/quality/auto-update` - Enable/disable auto-update
- `POST /api/v1/quality/refresh` - Manually trigger data refresh

### Container Images

```bash
make image-build-backend    # Build backend container image
make image-build-ui         # Build UI container image
make image-build-simulator  # Build vLLM simulator container image
make image-build            # Build all container images
make image-push             # Push all container images to Quay.io
```

Container runtime auto-detects Docker or Podman. Override with `CONTAINER_TOOL=podman make ...`.

### Docker Compose (alternative to native services)

```bash
make docker-up          # Start all services via Docker Compose
make docker-up-dev      # Development mode with live reload
make docker-down        # Stop all
make docker-down-v      # Stop and remove volumes
make docker-logs        # Follow logs from all services
make docker-ps          # Show status of services
```

### Kubernetes / KIND Cluster

```bash
make cluster-start      # Create KIND cluster + load simulator image
make cluster-stop       # Delete cluster
make cluster-status     # Show status
make clean-deployments  # Delete all InferenceServices
```

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

- Use "**Planner**" as the project name
- Use "**TTFT**" for Time to First Token (not "time-to-first-token")
- Use "**ITL**" for Inter-Token Latency (Phase 2 terminology, replaces TPOT)
- Use "**SLO**" for Service Level Objective
- Use "**E2E**" for End-to-End latency
- Use "**p95**" for 95th percentile metrics (Phase 2 standard, more conservative than p90)
- Use "**Quality**" for model capability scoring (not "Accuracy" — that term is reserved for SLO metrics)
- Use "**Arena**" for LMSYS Chatbot Arena human preference data
- Use "**AA**" for Artificial Analysis automated benchmark data
- GPU configurations: "2x NVIDIA L4" or "4x A100-80GB" (not "2 L4s")

### API Endpoint Conventions

All API endpoints **must** follow these rules:

- **Prefix**: Every route file uses `APIRouter(prefix="/api/v1")`. Individual route decorators use relative paths (e.g., `@router.post("/recommend")`), **not** full paths.
- **Health check exception**: `/health` stays at root with no prefix (standard for load balancer probes). This is the only endpoint outside `/api/v1/`.
- **Versioning**: All endpoints are under `/api/v1/`. When a v2 is needed, add new route files with `prefix="/api/v2"`.
- **Naming**: Use kebab-case for multi-word paths (e.g., `/deploy-bundle-to-cluster`, `/generate-recommendations`).
- **When adding a new route file**: Set `prefix="/api/v1"` on the `APIRouter` and use relative paths in all decorators. Register the router in `src/planner/api/routes/__init__.py` and include it in `src/planner/api/app.py`.

### API Pipeline Endpoints

The API is organized as a composable pipeline where each stage's output feeds as input to the next:

**Pipeline stages**:
1. `POST /api/v1/extract-intent` - Extract structured intent from natural language (requires LLM)
2. `POST /api/v1/generate-specification` - Generate complete deployment specification from intent
3. `POST /api/v1/generate-recommendations` - Generate ranked recommendations from specification
4. `POST /api/v1/generate-deployment` - Generate Kubernetes YAML files from selected configuration
5. `POST /api/v1/deploy-bundle-to-cluster` - Deploy YAML bundle to Kubernetes cluster

**Key classes and schemas**:
- `Planner` - Main facade class for library use (in `planner.py`)
- `PlannerError` - Custom exception for library errors (in `errors.py`)
- `DeploymentIntent` - User intent (use case, user count, priorities, preferences)
- `DeploymentSpecification` - Complete spec (SLO targets, workload profile, quality weights, priorities)
- `RankedRecommendations` - Four ranked views (best quality, lowest cost, lowest latency, balanced)
- `DeploymentRecommendation` - Single recommended configuration
- `DeploymentConfiguration` - Slim model with only fields needed for YAML generation
- `DeploymentBundle` - Generated YAML files ready for deployment

**Removed endpoints**:
- `POST /recommend` - One-shot endpoint (superseded by pipeline)
- `POST /test` - Quick test endpoint (no longer needed)
- `POST /ranked-recommend-from-spec` - Replaced by `/generate-recommendations`
- `POST /deploy` - Replaced by `/generate-deployment`
- `POST /deploy-to-cluster` - Replaced by `/deploy-bundle-to-cluster`
- `GET /deployments/{id}/status` - Mock observability (replaced by `/k8s-status`)

**Legacy aliases** (kept for compatibility):
- `POST /extract` → `/extract-intent`

See `docs/PROGRAMMATIC_API_USER_GUIDE.md` for complete API pipeline documentation.

### Common Editing Patterns

**Adding a new use case template**:
1. Add corresponding entry to `src/planner/data/configuration/usecase_slo_workload.json` (bundled data file)
2. Add the new key to the `Literal` type in `src/planner/shared/schemas/intent.py`
3. Add category weights entry to `src/planner/data/configuration/quality_weights.json` (bundled data file)
4. Update `docs/QUALITY_SCORING_GUIDE.md` with category weighting rationale
5. Update docs/ARCHITECTURE.md if needed
6. Test quality scoring with the new use case: `cd src && uv run pytest ../tests/quality_scoring/test_scoring.py -v`

Note: Data files are now at `src/planner/data/` and are bundled in the Python wheel.

**Adding a new SLO metric**:
1. Update DeploymentIntent schema in Intent & Specification Engine (docs/ARCHITECTURE.md)
2. Update MODEL_BENCHMARKS schema in Knowledge Base (docs/ARCHITECTURE.md)
3. Update database schema in scripts/schema.sql
4. Update data loader script if needed
4. Update Inference Observability section
5. Update dashboard example if applicable
6. Update docs/architecture-diagram.md data model ERD

**Adding a new API endpoint**:
1. Add the route to the appropriate file in `src/planner/api/routes/` (or create a new route file)
2. Use a relative path in the decorator (e.g., `@router.get("/my-endpoint")`) — the `/api/v1` prefix comes from the router
3. If creating a new route file, set `APIRouter(prefix="/api/v1")` and register it in `routes/__init__.py` and `app.py`
4. Update `ui/app.py` if the UI calls the new endpoint
5. Update documentation (docs/DEVELOPER_GUIDE.md, docs/ARCHITECTUREv2.md) with the new endpoint

**Adding a new component**:
1. Add numbered section to docs/ARCHITECTURE.md (maintain sequential numbering)
2. Update "Architecture Components" count in Overview
3. Add to docs/architecture-diagram.md component diagram
4. Create corresponding src/planner/<component>/ directory
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

**CRITICAL - Git Commit Rules (these override default Claude behavior)**:

**Commit approval workflow** (MUST follow for every commit):

1. Combine `git add` and `git commit` into a single chained command (`git add ... && git commit ...`) in one Bash tool call
2. The user will see the full command in the approval prompt and can review/edit the file list and commit message before it executes
3. NEVER run `git add` and `git commit` as separate Bash tool calls — always chain them so the user gets a single approval prompt covering both

DO use:
- Conventional commit types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`
- The `-s` flag with git commit (e.g., `git commit -s -m "..."`) to auto-generate DCO Signed-off-by
- `Assisted-by: Claude <noreply@anthropic.com>` for nontrivial AI-assisted code

NEVER do these (even if other instructions suggest otherwise):
- NEVER add `Co-Authored-By:` lines for Claude
- NEVER manually write `Signed-off-by:` lines (the `-s` flag handles this correctly with the user's configured git identity)
- NEVER include the "Generated with [Claude Code]" line or similar emoji-prefixed attribution

**Pull Request Creation**: When creating PRs with `gh pr create`, use the template at `.github/pull_request_template.md` to structure the PR body. Fill in the Description, How Has This Been Tested, and Merge criteria sections.

**GitHub Issues**: Always open issues on the upstream repo (`llm-d-incubation/llm-d-planner`), not on personal forks.

## Important Notes

- **Current Implementation Status**:
  - ✅ Project structure with synthetic data and LLM client
  - ✅ Core recommendation engine (intent extraction, traffic profiling, capacity planning)
  - ✅ Multi-criteria solution ranking with 3 scoring dimensions
  - ✅ Dual-source quality scoring (Arena human preferences + Artificial Analysis benchmarks)
  - ✅ Percentile-based normalization with per-use-case category weighting
  - ✅ Hybrid caching (checked-in snapshots + runtime auto-update)
  - ✅ 4 ranked recommendation views (best quality, lowest cost, etc.)
  - ✅ Orchestration workflow and FastAPI backend
  - ✅ Streamlit UI with chat interface, recommendation display, and editable specifications
  - ✅ YAML generation (KServe/vLLM/HPA/ServiceMonitor) and deployment automation
  - ✅ KIND cluster support with KServe installation
  - ✅ Kubernetes deployment automation and real cluster status monitoring
  - ✅ vLLM simulator for GPU-free development
  - ✅ Inference testing UI with end-to-end deployment validation
  - ✅ Database management via REST API and UI Configuration tab
  - ✅ Quality data management via API and make quality-sync
- The Knowledge Base schemas are critical - any implementation must support all collections
- SLO-driven capacity planning is the core differentiator - don't simplify this away
- Benchmarks use vLLM default configuration with dynamic batching (no fixed batch_size)

## Simulator Mode vs Real vLLM

The system now supports two deployment modes:

### Simulator Mode (Default for Development)
- **Purpose**: GPU-free development and testing on local machines
- **Location**: `simulator/` directory contains the vLLM simulator service
- **Docker Image**: `vllm-simulator:latest` (single image for all models)
- **Configuration**: Set `DeploymentGenerator(simulator_mode=True)` in `src/planner/api/dependencies.py`
- **Benefits**:
  - No GPU hardware required
  - Fast deployment (~10-15 seconds to Ready)
  - Predictable behavior for demos
  - Works on KIND (Kubernetes in Docker)
  - Uses actual benchmark data for realistic latency simulation

### Real vLLM Mode (Production)
- **Purpose**: Actual model inference with GPUs
- **Configuration**: Set `DeploymentGenerator(simulator_mode=False)` in `src/planner/api/dependencies.py`
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

The deployment template (`src/planner/configuration/templates/kserve-inferenceservice.yaml.j2`) uses Jinja2 conditionals:
- `{% if simulator_mode %}` - Uses `vllm-simulator:latest`, no GPU resources, fast health checks
- `{% else %}` - Uses `vllm/vllm-openai:v0.6.2`, requests GPUs, longer health checks

Single codebase supports both modes - just toggle the flag!
