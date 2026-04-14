# <img src="docs/planner-logo.ico" alt="Planner" width="32" style="vertical-align: middle;"/> llm-d Planner

[![CI](https://github.com/llm-d-incubation/llm-d-planner/actions/workflows/ci.yml/badge.svg)](https://github.com/llm-d-incubation/llm-d-planner/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Confidently navigate LLM deployments from concept to production.**

---

## Overview

The system addresses a critical challenge: **how do you translate business requirements into the right model and infrastructure choices without expensive trial-and-error?**

llm-d Planner combines three capabilities into a unified platform:

- **Conversational Recommendation Engine** — Describe your use case in natural language and get SLO-driven model + GPU recommendations backed by real benchmarks, with multi-criteria ranking and one-click Kubernetes deployment.
- **Capacity Planner** — Estimate GPU memory requirements for any HuggingFace model: model weights, KV cache, activation memory, and CUDA overhead. Determine minimum GPU count, maximum context length, and concurrent request capacity for a given hardware configuration.
- **GPU Recommender** — Predict inference performance (TTFT, ITL, throughput) across GPU types using BentoML's roofline model, without running actual benchmarks. Compare cost and latency tradeoffs to find the optimal GPU for your workload.

When benchmark data is unavailable for a model/GPU combination, the GPU Recommender automatically generates estimated performance, filling gaps in the recommendation pipeline so users always get actionable results.

### Key Features

- **Conversational Requirements Gathering** - Describe your use case in natural language
- **SLO-Driven Capacity Planning** - Translate business needs into technical specifications (traffic profiles, latency targets, cost constraints)
- **GPU Memory Estimation** - Calculate model weight memory, KV cache, activation overhead, and minimum GPU requirements for any HuggingFace model
- **Roofline Performance Prediction** - Estimate TTFT, ITL, and throughput across GPU types without running benchmarks
- **Intelligent Recommendations** - Get optimal model + GPU configurations backed by real benchmarks, with estimated performance fallback
- **Multi-Criteria Ranking** - Score configurations on accuracy, price, latency, and complexity with 5 ranked views
- **What-If Analysis** - Explore alternatives and compare cost vs. latency tradeoffs
- **One-Click Deployment** - Generate production-ready KServe/vLLM YAML and deploy to Kubernetes
- **Performance Monitoring** - Track actual deployment status and test inference in real-time
- **CLI and API** - Use the `planner` CLI or REST API for programmatic access to all capabilities
- **GPU-Free Development** - vLLM simulator enables local testing without GPU hardware

### How It Works

**Recommendation flow:**

1. **Extract Intent** - LLM-powered analysis converts your description into structured requirements
2. **Map to Traffic Profile** - Match use case to one of 4 GuideLLM benchmark configurations
3. **Set SLO Targets** - Auto-generate TTFT (p95), ITL (p95), and E2E (p95) targets based on experience class
4. **Query Benchmarks** - Exact match on (model, GPU, traffic profile) from PostgreSQL database
5. **Estimate if Missing** - For models/GPUs without benchmarks, generate estimated performance via roofline model
6. **Filter by SLOs** - Find configurations meeting all p95 latency targets
7. **Plan Capacity** - Calculate required replicas based on throughput requirements
8. **Generate & Deploy** - Create validated Kubernetes YAML and deploy to local or production clusters
9. **Monitor & Validate** - Track deployment status and test inference endpoints

**Capacity Planner flow:**

1. **Fetch Model Config** - Retrieve architecture details from HuggingFace (layers, heads, quantization)
2. **Calculate Memory** - Model weights, KV cache per token, activation memory, CUDA/system overhead
3. **Evaluate Hardware Fit** - Determine valid tensor parallelism values and whether the model fits on a given GPU
4. **Report Capacity** - Maximum context length, concurrent requests, KV cache blocks for the configuration

**GPU Recommender flow:**

1. **Select GPUs** - Choose from a catalog of GPU types (H100, A100, L40, etc.) or evaluate all
2. **Run Roofline Model** - BentoML's `llm-optimizer` estimates TTFT, ITL, and throughput per GPU
3. **Apply Constraints** - Filter by max TTFT, max ITL, max latency, and GPU count limits
4. **Rank by Cost** - Compare qualifying GPUs sorted by cost with performance details

## Prerequisites

**Required before running `make setup`:**

- **macOS or Linux** (Windows via WSL2)
- **Docker Desktop** (must be running)
- **Python 3.13** - `brew install python@3.13`
- **uv** - `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Ollama** - `brew install ollama`
- **kubectl** - `brew install kubectl`
- **KIND** - `brew install kind`

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

**Conversational Recommendations** (main page):

1. **Describe your use case** in the chat interface
   - Example: "I need a customer service chatbot for 5000 users with low latency"
2. **Analyze use case** - Click "Analyze Use Case" to extract intent
3. **Generate specification** - Click "Generate Specification" to create traffic profile and SLO targets
4. **Review specification** - Edit SLO targets, priorities, or constraints if needed
5. **Generate recommendations** - Click "Generate Recommendations" to find optimal configurations
6. **Select a recommendation** - Review ranked options and click "Select"
7. **Deploy** - Go to the "Deployment" tab to review, copy, or download generated deployment files

**Capacity Planner** (sidebar page): Enter a HuggingFace model ID and GPU configuration to see memory breakdown (model weights, KV cache, activation, overhead), maximum context length, and concurrent request capacity.

**GPU Recommender** (sidebar page): Enter a model ID and workload parameters (input/output token lengths) to get estimated inference performance across GPU types, ranked by cost.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed system design.

## Implemented Features

- ✅ **Core Recommendation Engine**: Intent extraction, traffic profiling, multi-criteria model + GPU recommendation, SLO-driven capacity planning
- ✅ **Capacity Planner**: GPU memory estimation (model weights, KV cache, activation, overhead), model-fits-GPU analysis, maximum context length and concurrent request calculations
- ✅ **GPU Recommender**: Roofline model performance estimation (TTFT, ITL, throughput) across GPU types via BentoML `llm-optimizer`, cost comparison, latency constraint filtering
- ✅ **Estimated Performance Fallback**: When benchmarks are unavailable, roofline estimates fill the gap in the recommendation pipeline (visually differentiated as "Estimated" vs "Benchmarked")
- ✅ **FastAPI Backend**: REST endpoints for recommendations, capacity planning, GPU estimation, and database management
- ✅ **Streamlit UI**: Chat interface, recommendation display, specification editor, Capacity Planner page, GPU Recommender page
- ✅ **CLI**: `planner plan` for capacity planning, `planner estimate` for GPU performance estimation
- ✅ **Deployment Automation**: YAML generation (KServe/vLLM/HPA/ServiceMonitor), Kubernetes deployment
- ✅ **Local Kubernetes**: KIND cluster support, KServe installation, cluster management
- ✅ **vLLM Simulator**: GPU-free development mode with realistic latency simulation
- ✅ **Monitoring & Testing**: Real-time deployment status, inference testing UI, cluster observability
- ✅ **Database Management**: Upload/reset benchmark data via REST API or UI Configuration tab

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

The `planner` CLI provides direct access to capacity planning and GPU estimation without running the web services.

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

## Completed Milestone Features

- ✅ **PostgreSQL Database** - Production-grade benchmark storage with psycopg2
- ✅ **Traffic Profile Framework** - 4 GuideLLM standard configurations: (512→256), (1024→1024), (4096→512), (10240→1536)
- ✅ **Experience-Driven SLOs** - 9 use cases mapped to 5 experience classes (instant, conversational, interactive, deferred, batch)
- ✅ **p95 Percentiles** - Conservative SLO guarantees using p95 targets
- ✅ **GPU Memory Estimation** - Model weights, KV cache (MHA/GQA/MQA/MLA), activation memory with empirically validated constants, CUDA/system overhead
- ✅ **Hardware Fit Analysis** - Valid tensor parallelism values, auto-calculated max context length, concurrent request capacity
- ✅ **Roofline Performance Estimation** - BentoML `llm-optimizer` predicts TTFT, ITL, throughput per GPU type without running benchmarks
- ✅ **Cost Comparison** - GPU cost ranking with custom cost overrides, latency constraint filtering
- ✅ **Estimated Performance Fallback** - When benchmark data is missing, roofline estimates fill the gap (visually differentiated as "Estimated" vs "Benchmarked")
- ✅ **CLI** - `planner plan` and `planner estimate` subcommands for programmatic access
- ✅ **REST API** - `/api/v1/model-info`, `/api/v1/calculate`, `/api/v1/estimate` endpoints

## Future Enhancements

1. **Production-Grade Ingress** - External access with TLS, authentication, rate limiting
2. **Production GPU Validation** - End-to-end testing with real GPU clusters
3. **Feedback Loop** - Actual metrics → benchmark updates
4. **Statistical Traffic Models** - Full distributions (not just point estimates)
5. **Multi-Dimensional Benchmarks** - Concurrency, batching, KV cache effects
6. **Security Hardening** - YAML validation, RBAC, network policies
7. **Multi-Tenancy** - Namespaces, resource quotas, isolation
8. **Advanced Simulation** - SimPy, Monte Carlo for what-if analysis

## Contributing

We are early in development of this project, but contributions are welcome. 

See [CLAUDE.md](CLAUDE.md) for AI assistant guidance when making changes.

## License

This project is licensed under Apache License 2.0. See the [LICENSE file](LICENSE) for details.
