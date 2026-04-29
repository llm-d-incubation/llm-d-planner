# <img src="docs/planner-logo.ico" alt="Planner" width="32" style="vertical-align: middle;"/> llm-d Planner

[![CI](https://github.com/llm-d-incubation/llm-d-planner/actions/workflows/ci.yml/badge.svg)](https://github.com/llm-d-incubation/llm-d-planner/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**From business requirements to running llm-d deployments — sizing, estimation, and deployment in one workflow.**

---

## Overview

Deploying LLMs in production means navigating a fragmented landscape of model selection, GPU sizing, performance estimation, and Kubernetes configuration. Platform teams typically piece together separate tools and manual calculations, leading to expensive trial-and-error.

llm-d Planner unifies these concerns into a single platform with three capabilities:

- **Planner** — The main workflow. Guides users from business requirements to a running LLM deployment through conversational requirements gathering, SLO-driven recommendations, and one-click Kubernetes deployment.
- **LLM Memory Analyzer** — Estimate GPU memory requirements for any HuggingFace model: model weights, KV cache, activation memory, and system overhead. Determine minimum GPU count, maximum context length, and concurrent request capacity for a given hardware configuration.
- **Inference Performance Analyzer** — Estimate inference performance (TTFT, ITL, throughput) across GPU types without running actual benchmarks. Compare latency and performance tradeoffs to find the optimal GPU for your workload.

Planner is the main workflow — it makes recommendations based on available benchmarks, and leverages the LLM Memory Analyzer and Inference Performance Analyzer to fill gaps and expand the search surface beyond what benchmarks alone can cover. The LLM Memory Analyzer and Inference Performance Analyzer are also useful on their own for deeper analysis in their respective areas, each with its own UI page, CLI command, and API endpoint.

### Key Features

- **Conversational Requirements Gathering** - Describe your AI-powered use case in natural language
- **SLO-Driven Capacity Planning** - Translate business needs into technical specifications (traffic profiles, latency targets, cost constraints)
- **GPU Memory Estimation** - Calculate model weight memory, KV cache, activation overhead, and minimum GPU requirements for any HuggingFace model
- **Performance Estimation** - Estimate TTFT, ITL, and throughput across GPU types without running benchmarks
- **SLO-Driven Recommendations** - Get optimal model + GPU configurations backed by real benchmarks, with estimated performance fallback
- **Multi-Criteria Ranking** - Score configurations on accuracy, price, latency, and complexity with 5 ranked views
- **What-If Analysis** - Explore alternatives and compare cost vs. latency tradeoffs
- **One-Click Deployment** - Generate production-ready KServe/vLLM YAML and deploy to Kubernetes
- **Performance Monitoring** - Track actual deployment status and test inference in real-time
- **CLI and API** - Use the `planner` CLI or REST API for programmatic access to all capabilities
- **GPU-Free Development** - vLLM simulator enables local testing without GPU hardware

### How It Works

**Planner flow:**

1. **Extract Intent** - LLM-powered analysis converts your description into structured requirements
2. **Map to Traffic Profile** - Map use case to appropriate [GuideLLM](https://github.com/vllm-project/guidellm) benchmark configurations
3. **Set SLO Targets** - Auto-generate TTFT, ITL, and E2E targets based on experience class
4. **Query Benchmarks** - Exact match on (model, GPU, traffic profile) from benchmark database
5. **Estimate if Missing** - For models/GPUs without benchmarks, generate estimated performance via Inference Performance Analyzer
6. **Filter by Latency** - Find configurations meeting all latency targets
7. **Plan Capacity** - Calculate required replicas based on throughput requirements
8. **Score and Rank Solutions** - Score and rank solutions for quality, cost, and latency.
9. **Present to User** - Present top options to user in each category.
10. **Generate & Deploy** - Create validated Kubernetes YAML and deploy to local or production clusters

**Capacity analysis flow:**

1. **Fetch Model Config** - Retrieve architecture details from HuggingFace (layers, heads, quantization)
2. **Calculate Memory** - Model weights, KV cache per token, activation memory, system overhead
3. **Evaluate Hardware Fit** - Determine valid tensor parallelism values and whether the model fits on a given GPU
4. **Report Capacity** - Maximum context length, concurrent requests, KV cache blocks for the configuration

**Performance analysis flow:**

1. **Select GPUs** - Choose from a catalog of GPU types (H100, A100, L40, etc.) or evaluate all
2. **Estimate Performance** - Estimate TTFT, ITL, and throughput per GPU (currently via [BentoML LLM Optimizer](https://github.com/bentoml/llm-optimizer) roofline model, with additional backends planned)
3. **Apply Constraints** - Filter by max TTFT, max ITL, max latency, and GPU count limits
4. **Rank Results** - Compare qualifying GPUs sorted by performance with cost details

## Prerequisites

**Required before running `make setup`:**

- **macOS or Linux** (Windows via WSL2)
- **Docker or Podman** (must be running)
- **Python 3.13** - `brew install python@3.13` (Linux: use your package manager or [pyenv](https://github.com/pyenv/pyenv))
- **uv** - `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Ollama** - `brew install ollama` (Linux: [ollama.com/download](https://ollama.com/download))
- **kubectl** - `brew install kubectl` (Linux: [kubernetes.io/docs/tasks/tools](https://kubernetes.io/docs/tasks/tools/))
- **KIND** - `brew install kind` (Linux: [kind.sigs.k8s.io/docs/user/quick-start](https://kind.sigs.k8s.io/docs/user/quick-start/))

## Quick Start

**Get up and running in 4 commands:**

```bash
make setup          # Install dependencies, pull Ollama model
make start          # Start all services (DB + Ollama + Backend + UI)
make db-load-blis   # Load BLIS benchmark data
make cluster-start  # Optional: Create local KIND cluster with vLLM simulator for testing deployments
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

**Note**: PostgreSQL runs as a Docker container (`planner-postgres`) with benchmark data. All `db-load-*` commands append to existing data. Use `make db-reset` first for a clean database. Benchmark data can also be uploaded and managed via the UI's **Configuration** tab or the REST API (`/api/v1/db/upload-benchmarks`).

**Stop everything:**
```bash
make stop           # Stop Backend + UI (leaves Ollama and DB running)
make stop-all       # Stop all services including Ollama and DB
make cluster-stop   # Delete cluster (optional)
```

### Using Planner

**Recommendations** (main page):

1. **Describe your use case** in the chat interface
   - Example: "I need a customer service chatbot for 5000 users with low latency"
2. **Analyze use case** - Click "Analyze Use Case" to extract intent
3. **Generate specification** - Click "Generate Specification" to create traffic profile and SLO targets
4. **Review specification** - Edit SLO targets, priorities, or constraints if needed
5. **Generate recommendations** - Click "Generate Recommendations" to find optimal configurations
6. **Select a recommendation** - Review ranked options and click "Select"
7. **Deploy** - Go to the "Deployment" tab to review, copy, or download generated deployment files

**LLM Memory Analyzer** (sidebar page): Enter a HuggingFace model ID and GPU configuration to see memory breakdown (model weights, KV cache, activation, overhead), maximum context length, and concurrent request capacity.

**Inference Performance Analyzer** (sidebar page): Enter a model ID and workload parameters (input/output token lengths) to get estimated inference performance across GPU types, ranked by cost.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed system design.

## Implemented Features

- ✅ **Planner**: Intent extraction, traffic profiling, multi-criteria model + GPU recommendation, SLO-driven capacity planning
- ✅ **LLM Memory Analyzer**: GPU memory estimation (model weights, KV cache for MHA/GQA/MQA/MLA, activation, overhead), model-fits-GPU analysis, maximum context length and concurrent request calculations
- ✅ **Inference Performance Analyzer**: Performance estimation (TTFT, ITL, throughput) across GPU types, cost comparison, latency constraint filtering
- ✅ **Estimated Performance Fallback**: When benchmarks are unavailable, roofline estimates fill the gap in the recommendation pipeline (visually differentiated as "Estimated" vs "Benchmarked")
- ✅ **Traffic Profile Framework**: 4 GuideLLM standard configurations with experience-driven SLOs (9 use cases, 5 experience classes, p95 targets)
- ✅ **FastAPI Backend**: REST endpoints for recommendations, capacity planning, GPU estimation, and database management (`/api/v1/model-info`, `/api/v1/calculate`, `/api/v1/estimate`)
- ✅ **Streamlit UI**: Chat interface, recommendation display, specification editor, LLM Memory Analyzer page, Inference Performance Analyzer page
- ✅ **CLI**: `planner plan` for capacity planning, `planner estimate` for GPU performance estimation
- ✅ **Deployment Automation**: YAML generation (KServe/vLLM/HPA/ServiceMonitor), Kubernetes deployment
- ✅ **Local Kubernetes**: KIND cluster support, KServe installation, cluster management
- ✅ **vLLM Simulator**: GPU-free development mode with realistic latency simulation
- ✅ **Monitoring & Testing**: Real-time deployment status, inference testing UI, cluster observability
- ✅ **Database Management**: PostgreSQL benchmark storage, upload/reset via REST API or UI Configuration tab

## Key Technologies

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI, Pydantic |
| Frontend | Streamlit |
| LLM | Ollama (qwen2.5:7b) |
| Data | PostgreSQL |
| Performance Estimation | BentoML llm-optimizer (roofline model) |
| Model Configs | HuggingFace Hub API, transformers |
| YAML Generation | Jinja2 templates |
| Kubernetes | KIND (local), KServe v0.14.0 |
| Deployment | kubectl |


## CLI

The `planner` CLI provides direct access to the LLM Memory Analyzer and Inference Performance Analyzer without running the web services.

```bash
# Capacity planning — memory breakdown for a model
planner plan --model Qwen/Qwen3-32B

# With GPU memory specified — shows allocatable KV cache, max concurrent requests
planner plan --model Qwen/Qwen3-32B --gpu-memory 80 --tp 4

# Auto-calculate maximum context length that fits
planner plan --model Qwen/Qwen3-32B --gpu-memory 80 --max-model-len -1

# GPU performance estimation — estimate TTFT/ITL/throughput across GPUs
planner estimate --model Qwen/Qwen3-32B --input-len 512 --output-len 128

# Estimate with specific GPUs and latency constraints
planner estimate --model Qwen/Qwen3-32B --input-len 512 --output-len 128 \
  --gpu-list H100,A100,L40 --max-ttft 100 --max-itl 10

# Human-readable output sorted by cost
planner estimate --model Qwen/Qwen3-32B --input-len 512 --output-len 128 --pretty
```

Run `planner --help`, `planner plan --help`, or `planner estimate --help` for all options.

## Development Commands

```bash
make help                    # Show all available commands
make start                   # Start all services (DB + Ollama + Backend + UI)
make stop                    # Stop Backend + UI (leaves Ollama and DB running)
make stop-all                # Stop everything including Ollama and DB
make restart                 # Restart all services
make logs-backend            # Show backend logs
make logs-ui                 # Show UI logs

# Database (PostgreSQL)
make db-start                # Start PostgreSQL (initializes schema on first run)
make db-load-blis            # Load BLIS benchmark data (appends)
make db-load-estimated       # Load estimated performance data (appends)
make db-load-interpolated    # Load interpolated benchmark data (appends)
make db-load-guidellm        # Load benchmark data created with GuideLLM (not included in repo yet) (appends)
make db-reset                # Reset database (remove all data and reinitialize)
make db-shell                # Open PostgreSQL shell
make db-query-models         # Query available models in database
make db-query-traffic        # Query traffic patterns in database

# Kubernetes
make cluster-status          # Check Kubernetes cluster status
make clean-deployments       # Delete all InferenceServices

# Testing
make test                    # Run all tests (requires DB and Ollama)
make test-unit               # Run unit tests (no external dependencies)
make test-db                 # Run database tests (requires PostgreSQL)
make test-integration        # Run integration tests (requires Ollama and DB)

make clean                   # Remove generated files
```

## vLLM Simulator Mode

Planner includes a **GPU-free simulator** for local development:

- **No GPU required** - Run deployments on any laptop
- **OpenAI-compatible API** - `/v1/completions` and `/v1/chat/completions`
- **Realistic latency** - Uses benchmark data to simulate TTFT/ITL
- **Fast deployment** - Pods become Ready in ~10-15 seconds

The deployment mode defaults to **production** (real vLLM with GPUs). Switch between production and simulator modes at runtime using the **Configuration** tab in the UI, or via the REST API:

- `GET /api/v1/deployment-mode` - Check current mode
- `PUT /api/v1/deployment-mode` - Set mode (`{"mode": "simulator"}` or `{"mode": "production"}`)

See [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md#vllm-simulator-details) for details.

## Documentation

- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - Development workflows, testing, debugging
- **[Architecture](docs/ARCHITECTURE.md)** - Detailed system design and component specifications
- **[Traffic and SLOs](docs/traffic_and_slos.md)** - Traffic profile framework and experience-driven SLOs (Phase 2)
- **[Logging Guide](docs/LOGGING.md)** - Logging system and debugging
- **[Claude Code Guidance](CLAUDE.md)** - AI assistant instructions for contributors

## Future Enhancements

1. **Prefill/Decode Disaggregation** - Support P/D disaggregation as a deployment topology option
2. **llm-d Stack Configuration** - Expose stack-level configuration including routing, P/D disaggregation, and fine-grained vLLM parameters
3. **vLLM Parameter Search** - Expand configuration search to include serving-stack knobs (batch size, scheduling policy, memory utilization)
4. **Security Hardening** - YAML validation, RBAC, network policies

Some of these enhancements may involve collaboration with other llm-d SIGs.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

See [CLAUDE.md](CLAUDE.md) for AI assistant guidance when making changes.

## License

This project is licensed under Apache License 2.0. See the [LICENSE file](LICENSE) for details.
