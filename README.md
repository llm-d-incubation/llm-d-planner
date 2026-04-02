# <img src="docs/planner-logo.ico" alt="Planner" width="32" style="vertical-align: middle;"/> llm-d Planner

[![CI](https://github.com/llm-d-incubation/llm-d-planner/actions/workflows/ci.yml/badge.svg)](https://github.com/llm-d-incubation/llm-d-planner/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Confidently navigate LLM deployments from concept to production.**

---

## Overview

The system addresses a critical challenge: **how do you translate business requirements into the right model and infrastructure choices without expensive trial-and-error?**

llm-d Planner guides you from concept to production LLM deployments through SLO-driven capacity planning. Conversationally define your requirements — Planner translates them into traffic profiles, performance targets, and cost constraints. Get intelligent model and GPU recommendations based on real benchmarks. Explore alternatives, compare tradeoffs, deploy with one click, and monitor actual performance—staying on course as your needs evolve.

The code in this repository implements the **Planner Phase 2 MVP** with production-grade data management. Phase 1 (POC) demonstrated the end-to-end workflow with synthetic data. Phase 2 adds PostgreSQL for benchmark storage, a traffic profile framework aligned with GuideLLM standards, experience-driven SLO mapping, and p95 percentile targets for conservative guarantees.

### Key Features

- **🗣️ Conversational Requirements Gathering** - Describe your use case in natural language
- **📊 SLO-Driven Capacity Planning** - Translate business needs into technical specifications (traffic profiles, latency targets, cost constraints)
- **🎯 Intelligent Recommendations** - Get optimal model + GPU configurations backed by real benchmark data
- **🔍 What-If Analysis** - Explore alternatives and compare cost vs. latency tradeoffs
- **⚡ One-Click Deployment** - Generate production-ready KServe/vLLM YAML and deploy to Kubernetes
- **📈 Performance Monitoring** - Track actual deployment status and test inference in real-time
- **💻 GPU-Free Development** - vLLM simulator enables local testing without GPU hardware

### How It Works

1. **Extract Intent** - LLM-powered analysis converts your description into structured requirements
2. **Map to Traffic Profile** - Match use case to one of 4 GuideLLM benchmark configurations
3. **Set SLO Targets** - Auto-generate TTFT (p95), ITL (p95), and E2E (p95) targets based on experience class
4. **Query Benchmarks** - Exact match on (model, GPU, traffic profile) from PostgreSQL database
5. **Filter by SLOs** - Find configurations meeting all p95 latency targets
6. **Plan Capacity** - Calculate required replicas based on throughput requirements
7. **Generate & Deploy** - Create validated Kubernetes YAML and deploy to local or production clusters
8. **Monitor & Validate** - Track deployment status and test inference endpoints

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

1. **Describe your use case** in the chat interface
   - Example: "I need a customer service chatbot for 5000 users with low latency"
2. **Analyze use case** - Click "Analyze Use Case" to extract intent
3. **Generate specification** - Click "Generate Specification" to create traffic profile and SLO targets
4. **Review specification** - Edit SLO targets, priorities, or constraints if needed
5. **Generate recommendations** - Click "Generate Recommendations" to find optimal configurations
6. **Select a recommendation** - Review ranked options and click "Select"
7. **Deploy** - Go to the "Deployment" tab to review, copy, or download generated deployment files

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed system design.

## Implemented Features

- ✅ **Foundation**: Project structure, synthetic data, LLM client (Ollama)
- ✅ **Core Recommendation Engine**: Intent extraction, traffic profiling, model recommendation, capacity planning
- ✅ **FastAPI Backend**: REST endpoints, orchestration workflow, knowledge base access
- ✅ **Streamlit UI**: Chat interface, recommendation display, specification editor
- ✅ **Deployment Automation**: YAML generation (KServe/vLLM/HPA/ServiceMonitor), Kubernetes deployment
- ✅ **Local Kubernetes**: KIND cluster support, KServe installation, cluster management
- ✅ **vLLM Simulator**: GPU-free development mode with realistic latency simulation
- ✅ **Monitoring & Testing**: Real-time deployment status, inference testing UI, cluster observability
- ✅ **Database Management**: Upload/reset benchmark data via REST API or UI Configuration tab (no shell access needed)

## Key Technologies

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI, Pydantic |
| Frontend | Streamlit |
| LLM | Ollama (qwen2.5:7b) |
| Data | PostgreSQL |
| YAML Generation | Jinja2 templates |
| Kubernetes | KIND (local), KServe v0.14.0 |
| Deployment | kubectl |


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

## Phase 2 Completed Features

Phase 2 MVP improvements (now complete):

- ✅ **PostgreSQL Database** - Production-grade benchmark storage with psycopg2
- ✅ **Traffic Profile Framework** - 4 GuideLLM standard configurations: (512→256), (1024→1024), (4096→512), (10240→1536)
- ✅ **Experience-Driven SLOs** - 9 use cases mapped to 5 experience classes (instant, conversational, interactive, deferred, batch)
- ✅ **p95 Percentiles** - More conservative SLO guarantees (changed from p90)
- ✅ **ITL Terminology** - Inter-Token Latency instead of TPOT (Time Per Output Token)
- ✅ **Exact Traffic Matching** - No fuzzy matching, exact (prompt_tokens, output_tokens) queries
- ✅ **Pre-calculated E2E** - E2E latency stored in benchmarks for accuracy
- ✅ **Enhanced SLO Filtering** - Find configurations meeting all p95 targets

## Future Enhancements (Phase 3+)

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
