# NeuralNav - Technical Overview

---

## Slide 1: Project Goals

**Problem**: LLM production deployments are complex and error-prone
- Developers struggle to translate business needs into infrastructure choices
- Trial-and-error wastes time and money on expensive GPU resources
- No guidance on model selection, GPU sizing, or SLO target setting

**Solution**: Conversational assistant that automates the path from concept to production
- 4-stage flow: Understand context → Generate recommendations → Enable exploration → One-click deployment
- SLO-driven capacity planning with benchmark-backed predictions
- GPU-free development via vLLM simulator

**Current Status**: Phase 1 POC complete - full end-to-end workflow operational

---

## Slide 2: Core Innovation - SLO-Driven Capacity Planning

**User input** (natural language):
> "I need a chatbot for 1000 users, low latency is critical"

**System generates** (structured specs):
- Traffic profile: avg prompt 150 tokens, generation 200 tokens, peak QPS 100
- SLO targets: TTFT p90 < 200ms, TPOT p90 < 50ms, E2E p90 < 10150ms
- GPU recommendation: 2x NVIDIA L4, independent replicas
- Cost estimate: $800/month

**Key differentiator**: Translate high-level intent into production-ready infrastructure specifications

---

## Slide 3: Architecture Overview (10 Components)

1. **Conversational Interface** - Streamlit UI with chat and multi-tab recommendations
2. **Context & Intent Engine** - Extract structured specs from conversation, map to SLO templates
3. **Recommendation Engine** - 3 sub-components:
   - Traffic Profile Generator
   - Model Recommendation (filter by task, rank by priority)
   - Capacity Planning (GPU sizing, SLO compliance prediction)
4. **Simulation & Exploration** - What-if analysis, editable specifications
5. **Deployment Automation** - Generate KServe/vLLM YAML, deploy to K8s
6. **Knowledge Base** - Benchmarks, SLO templates, model catalog, deployment outcomes
7. **LLM Backend** - Ollama (llama3.1:8b) for conversational AI
8. **Orchestration** - FastAPI workflow coordination
9. **Inference Observability** - Monitor TTFT/TPOT/throughput, validate SLO compliance
10. **vLLM Simulator** - GPU-free development and testing

---

## Slide 4: Technology Stack (Phase 1)

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **UI** | Streamlit | Rapid prototyping, built-in session management |
| **Backend** | FastAPI + Python | Stack consistency, mature ecosystem |
| **Intent Extraction** | Ollama + Pydantic | Structured output with validation |
| **Recommendations** | Rule-based + LLM | Explainable, no training required |
| **Deployment** | Jinja2 + K8s Python client | Direct control, native templating |
| **Knowledge Base** | JSON files (POC) → PostgreSQL (Production) | Simple for POC, scalable for production |
| **Inference Platform** | KServe + vLLM | Industry standard for LLM serving |
| **Simulator** | FastAPI + Docker | Benchmark-driven latency simulation |

**Design principle**: Full Python stack for rapid Phase 1 development (may migrate deployment engine to Go in Phase 2+ for advanced K8s integration)

---

## Slide 5: Knowledge Base - 7 Critical Data Collections

1. **Model Benchmarks** - TTFT/TPOT/throughput for (model, GPU, tensor_parallel) tuples
   - 24 model+GPU combinations in POC
   - Collected using vLLM default config (dynamic batching enabled)

2. **Use Case SLO Templates** - Default targets for chatbot, summarization, code-gen, etc.
   - 7 templates with traffic defaults and SLO targets

3. **Model Catalog** - Curated, approved models with task/domain metadata
   - 10 models in POC (Llama-3, Mistral, Qwen, etc.)

4. **Hardware Profiles** - GPU specs, availability, pricing

5. **Cost Data** - Per-GPU hourly/monthly rates

6. **Deployment Templates** - Pre-configured patterns for common scenarios

7. **Deployment Outcomes** - Actual performance data for feedback loop (Phase 2)

**Current POC**: Synthetic data in JSON files; production requires database with vector search

---

## Slide 6: Simulator Mode vs Real vLLM

**Two deployment modes with single codebase**:

**Simulator Mode** (default for POC):
- Docker image: `vllm-simulator:latest`
- No GPU required, runs on KIND (Kubernetes in Docker)
- Fast deployment (~10-15 seconds to Ready)
- Benchmark-driven latency simulation using actual performance data
- Use case: Local development, testing, demos, CI/CD

**Real vLLM Mode** (production):
- Docker image: `vllm/vllm-openai:v0.6.2`
- Requires GPU-enabled K8s cluster + NVIDIA GPU Operator
- Downloads models from HuggingFace, real GPU inference
- Use case: Production deployments, performance benchmarking

**Toggle**: Single flag in `backend/src/api/routes.py` - `DeploymentGenerator(simulator_mode=True/False)`

---

## Slide 7: Implementation Status & Demo Flow

**What's complete** (Phase 1 POC):
- ✅ Conversational UI with chat interface
- ✅ Intent extraction and traffic profile generation
- ✅ Model + GPU recommendations with SLO compliance prediction
- ✅ Editable specifications (user can modify auto-generated specs)
- ✅ YAML generation (KServe InferenceService, HPA, ServiceMonitor)
- ✅ Kubernetes deployment automation (KIND cluster tested)
- ✅ Real cluster monitoring (pod status, resource usage)
- ✅ Inference testing UI (end-to-end validation via OpenAI API)
- ✅ vLLM simulator for GPU-free development

**Demo walkthrough**:
1. Conversational requirement gathering
2. Auto-generated recommendations with multiple options
3. Editable specifications review
4. One-click deployment to KIND cluster
5. Monitoring dashboard with inference testing