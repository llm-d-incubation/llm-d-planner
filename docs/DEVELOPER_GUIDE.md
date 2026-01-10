# Developer Guide

This guide provides step-by-step instructions for developing and testing NeuralNav.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Component Startup Sequence](#component-startup-sequence)
- [Development Workflows](#development-workflows)
- [Testing](#testing)
- [Debugging](#debugging)
- [Making Changes](#making-changes)
- [Simulator Development](#simulator-development)
- [Clean Up](#clean-up)
- [Code Quality](#code-quality)
- [Useful Commands](#useful-commands)
- [Alternative Setup Methods](#alternative-setup-methods)
- [Running Services Manually](#running-services-manually)
- [Troubleshooting](#troubleshooting)
- [Manual Kubernetes Cluster Setup](#manual-kubernetes-cluster-setup)
- [YAML Deployment Generation](#yaml-deployment-generation)
- [vLLM Simulator Details](#vllm-simulator-details)
- [Testing Details](#testing-details)

## Development Environment Setup

### Prerequisites

Ensure you have all required tools installed:

```bash
make check-prereqs
```

This checks for:
- Docker or Podman (running)
- Python 3.11+
- Ollama
- kubectl
- KIND

### Container Runtime Support

NeuralNav supports both **Docker** and **Podman** as container runtimes.

#### Compatibility Matrix

| Component | Docker | Podman | Notes |
|-----------|--------|--------|-------|
| PostgreSQL (`db-*` targets) | ✅ | ✅ | Works with either |
| Simulator build/push/pull | ✅ | ✅ | Works with either |
| KIND cluster (`cluster-*` targets) | ✅ | ❌ | KIND requires Docker |
| Docker Compose (`docker-*` targets) | ✅ | ⚠️ | Requires `podman-compose` |

#### Auto-Detection Behavior

The Makefile automatically detects which container runtime is available **and running**:
- **If Docker daemon is running:** Docker is used (for KIND compatibility)
- **If only Podman daemon is running:** Podman is used automatically
- **If neither daemon is running:** Commands will fail with a helpful error

This means if you quit Docker Desktop, the Makefile will automatically use Podman (if its machine is running), and vice versa.

#### Setting the Container Tool

**Option 1: Per-command override**
```bash
CONTAINER_TOOL=podman make db-start
CONTAINER_TOOL=podman make db-init
CONTAINER_TOOL=podman make db-load-guidellm
```

**Option 2: Export for your shell session**
```bash
export CONTAINER_TOOL=podman
make db-start
make db-init
make db-load-guidellm
# All commands in this session will use Podman
```

**Option 3: Create a `.env` file (persistent)**
```bash
echo "CONTAINER_TOOL=podman" >> .env
```
The Makefile automatically loads `.env`, so all subsequent `make` commands will use Podman. The `.env` file is in `.gitignore` so it won't affect other developers.

#### Using Podman on macOS

Podman on macOS requires a Linux VM to run containers:

```bash
# First time setup - initialize the Podman machine
podman machine init

# Start the Podman machine (required before each use, or after reboot)
podman machine start
```

**Important:** When starting the Podman machine, you may see this message:
```
Another process was listening on the default Docker API socket address.
You can still connect Docker API clients by setting DOCKER_HOST using the
following command in your terminal session:

    export DOCKER_HOST='unix:///var/folders/.../podman-machine-default-api.sock'
```

**Do NOT set `DOCKER_HOST`** - this redirects the Docker CLI to Podman, which causes confusion. Instead, use `CONTAINER_TOOL=podman` as described above.

#### Port Conflicts

If you switch between Docker and Podman, you may encounter port conflicts:

```
Error: listen tcp :5432: bind: address already in use
```

**To resolve:**
1. Stop and remove the container in the other runtime:
   ```bash
   # If switching TO Podman, clean up Docker first:
   docker stop neuralnav-postgres && docker rm neuralnav-postgres

   # If switching TO Docker, clean up Podman first:
   podman stop neuralnav-postgres && podman rm neuralnav-postgres
   ```

2. If the port is still in use, check what's holding it:
   ```bash
   lsof -i :5432
   ```

3. You may need to restart Docker Desktop if it has a stale port binding.

#### Mixed Docker/Podman Setup

You can use Podman for simple containers while keeping Docker for KIND:

1. Keep Docker Desktop installed (required for KIND clusters)
2. Use `CONTAINER_TOOL=podman` or `.env` for database operations
3. KIND cluster commands (`make cluster-*`) will use Docker automatically via `scripts/kind-cluster.sh`

Example workflow:
```bash
# Use Podman for database
export CONTAINER_TOOL=podman
make db-init
make db-load-guidellm

# KIND still uses Docker (no change needed)
make cluster-start
```

### Initial Setup

Create virtual environments and install dependencies:

```bash
make setup
```

This creates a single shared virtual environment in `venv/` (at project root) used by both the backend and UI.

## Component Startup Sequence

The system consists of 4 main components that must start in order:

### 1. Ollama Service

**Purpose:** LLM inference for intent extraction

**Start:**
```bash
make start-ollama
```

**Manual start:**
```bash
ollama serve
```

**Verify:**
```bash
curl http://localhost:11434/api/tags
ollama list  # Should show llama3.1:8b
```

### 2. FastAPI Backend

**Purpose:** Recommendation engine, workflow orchestration, API endpoints

**Start:**
```bash
make start-backend
```

**Manual start:**
```bash
cd backend
source venv/bin/activate
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Verify:**
```bash
curl http://localhost:8000/health
# Should return: {"status":"healthy"}
```

**API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 3. Streamlit UI

**Purpose:** Conversational interface, recommendation display

**Start:**
```bash
make start-ui
```

**Manual start:**
```bash
source venv/bin/activate
streamlit run ui/app.py
```

**Access:**
- http://localhost:8501

**Note:** UI runs from project root to access `docs/` assets

### 4. KIND Cluster (Optional)

**Purpose:** Local Kubernetes for deployment testing

**Start:**
```bash
make cluster-start
```

**Manual start:**
```bash
scripts/kind-cluster.sh start
```

**Verify:**
```bash
kubectl cluster-info
kubectl get pods -A
make cluster-status
```

## Development Workflows

### Quick Development Cycle

**Start all services:**
```bash
make dev
```

**Make code changes, then:**
- **Backend changes:** Auto-reloads (uvicorn `--reload` flag)
- **UI changes:** Refresh browser (Streamlit auto-detects changes)
- **Data changes:** Restart backend to reload JSON files

**Stop services:**
```bash
make stop
```

### Working on Specific Components

**Backend only:**
```bash
make start-backend
make logs-backend  # Tail logs
```

**UI only (requires backend running):**
```bash
make start-ui
make logs-ui
```

**Test API endpoints:**
```bash
# Get recommendation
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"message": "I need a chatbot for 1000 users"}'
```

### Cluster Development

**Create cluster:**
```bash
make cluster-start
# Builds simulator, creates cluster, loads image
```

**Deploy from UI:**
1. Get recommendation
2. Generate YAML
3. Click "Deploy to Kubernetes"
4. Monitor in "Deployment Management" tab

**Manual deployment:**
```bash
# After generating YAML via UI
kubectl apply -f generated_configs/kserve-inferenceservice.yaml
kubectl get inferenceservices
kubectl get pods
```

**Clean up deployments:**
```bash
make clean-deployments  # Delete all InferenceServices
```

**Restart cluster:**
```bash
make cluster-restart  # Fresh cluster
```

## Testing

### Unit Tests

Test individual components without external dependencies:

```bash
make test-unit
```

**Run specific test:**
```bash
cd backend
source venv/bin/activate
pytest tests/test_model_evaluator.py -v
```

### Integration Tests

Test components with Ollama integration:

```bash
make test-integration
```

Requires Ollama running with `llama3.1:8b` model.

### End-to-End Tests

Test full workflow including Kubernetes deployment:

```bash
make test-e2e
```

Requires:
- Ollama running
- KIND cluster running
- Backend and UI services

### Workflow Test

Test the complete recommendation workflow:

```bash
make test-workflow
```

This runs `backend/test_workflow.py` which tests all 3 demo scenarios.

### Watch Mode

Run tests continuously on file changes:

```bash
make test-watch
```

## Debugging

### Logging

NeuralNav implements comprehensive logging to help you debug and monitor the system. For complete logging documentation, see [docs/LOGGING.md](LOGGING.md).

**Quick Start:**

Enable debug logging to see full LLM prompts and responses:
```bash
# Enable debug mode
export NEURALNAV_DEBUG=true
make start-backend

# Or inline:
NEURALNAV_DEBUG=true make start-backend
```

**Log Levels:**
- **INFO (default)**: User requests, workflow steps, LLM metadata, results
- **DEBUG**: Full LLM prompts, complete responses, detailed timing

**Log Locations:**
- Console output (stdout/stderr)
- `logs/backend.log` - Main application logs
- `logs/neuralnav.log` - Structured detailed logs

**Common Log Searches:**
```bash
# View all user requests
grep "\[USER MESSAGE\]" logs/backend.log

# View LLM prompts (DEBUG mode only)
grep "\[LLM PROMPT\]" logs/backend.log

# View extracted intents
grep "\[EXTRACTED INTENT\]" logs/backend.log

# Follow a complete request flow
grep -A 50 "USER REQUEST" logs/backend.log
```

**Log Tags:**
- `[USER REQUEST]` - User request start
- `[USER MESSAGE]` - User's actual message
- `[LLM REQUEST]` - Request to LLM (metadata)
- `[LLM PROMPT]` - Full prompt text (DEBUG only)
- `[LLM RESPONSE]` - Response from LLM (metadata)
- `[LLM RESPONSE CONTENT]` - Full response text (DEBUG only)
- `[EXTRACTED INTENT]` - Parsed intent from LLM
- `Step 1`, `Step 2`, etc. - Workflow progress

**Privacy Note:** DEBUG mode logs contain full user messages and LLM interactions. Only use in development/testing.

### View Logs

**Backend logs:**
```bash
make logs-backend
# Or manually:
tail -f .pids/backend.pid.log
# Or for detailed logs:
tail -f logs/backend.log
```

**UI logs:**
```bash
make logs-ui
# Or manually:
tail -f .pids/ui.pid.log
```

**Kubernetes pod logs:**
```bash
kubectl logs -f <pod-name>
kubectl describe pod <pod-name>
```

### Check Service Health

```bash
make health
```

Checks:
- Backend: http://localhost:8000/health
- UI: http://localhost:8501
- Ollama: http://localhost:11434/api/tags

### Debug Intent Extraction

Test LLM client directly:

```bash
cd backend
source venv/bin/activate
python -c "
from src.llm.ollama_client import OllamaClient
from src.context_intent.extractor import IntentExtractor

client = OllamaClient()
extractor = IntentExtractor(client)

message = 'I need a chatbot for 5000 users with low latency'
intent = extractor.extract_intent(message)
print(intent)
"
```

### Debug Recommendations

Test recommendation engine:

```bash
cd backend
source venv/bin/activate
python -c "
from src.orchestration.workflow import RecommendationWorkflow

workflow = RecommendationWorkflow()
rec = workflow.generate_recommendation('I need a chatbot for 1000 users')
print(rec)
"
```

### Debug Cluster Deployments

**Check InferenceService status:**
```bash
kubectl get inferenceservices
kubectl describe inferenceservice <deployment-id>
```

**Check pod status:**
```bash
kubectl get pods
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

**Port-forward to service:**
```bash
kubectl port-forward svc/<deployment-id>-predictor 8080:80
curl http://localhost:8080/health
```

## Making Changes

### Adding a New Model

1. Add model to `data/model_catalog.json`:
```json
{
  "model_id": "new-model-id",
  "name": "New Model Name",
  "size_parameters": "7B",
  "context_length": 8192,
  "supported_tasks": ["chat", "instruction_following"],
  "recommended_for": ["chatbot"],
  "domain_specialization": ["general"]
}
```

2. Add benchmarks to `data/benchmarks.json`
3. Restart backend: `make restart`

### Adding a New Use Case Template

1. Add template to `data/slo_templates.json`:
```json
{
  "use_case": "new_use_case",
  "description": "Description",
  "prompt_tokens_mean": 200,
  "generation_tokens_mean": 150,
  "ttft_p90_target_ms": 250,
  "tpot_p90_target_ms": 60,
  "e2e_p90_target_ms": 3000
}
```

2. Update `backend/src/context_intent/extractor.py` USE_CASE_MAP
3. Restart backend

### Modifying the UI

UI code is in `ui/app.py`. Changes auto-reload in the browser.

**Key sections:**
- `render_chat_interface()` - Chat input/history
- `render_recommendation()` - Recommendation tabs
- `render_deployment_management_tab()` - Cluster management

### Modifying the Recommendation Algorithm

**Model scoring:** `backend/src/recommendation/model_evaluator.py`
- `score_model()` - Adjust scoring weights

**Capacity planning:** `backend/src/recommendation/capacity_planner.py`
- `plan_capacity()` - GPU sizing logic
- `_calculate_required_replicas()` - Scaling calculations

**Traffic profiling:** `backend/src/recommendation/traffic_profile.py`
- `generate_profile()` - Traffic estimation
- `generate_slo_targets()` - SLO target generation

### Code Quality

**Lint code:**
```bash
make lint
```

**Format code:**
```bash
make format
```

**Both use the shared project venv at root.**

## Simulator Development

### Building the Simulator

```bash
make build-simulator
```

Creates `vllm-simulator:latest` Docker image.

### Testing the Simulator Locally

```bash
# Can use podman instead of docker
docker run -p 8080:8080 \
  -e MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 \
  -e GPU_TYPE=NVIDIA-L4 \
  -e TENSOR_PARALLEL_SIZE=1 \
  vllm-simulator:latest

# Test
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10}'
```

### Pushing to Quay.io

```bash
make push-simulator
```

Auto-prompts for login if not authenticated.

## Clean Up

**Remove generated files:**
```bash
make clean
```

**Remove everything (including venvs):**
```bash
make clean-all
```

**Remove cluster:**
```bash
make cluster-stop
```

## Code Quality

### Linting and Formatting

NeuralNav uses [Ruff](https://docs.astral.sh/ruff/) for linting and code formatting.

**Run linter:**
```bash
make lint
```

Or manually:
```bash
source venv/bin/activate
ruff check backend/ ui/
```

**Auto-fix issues:**
```bash
source venv/bin/activate
ruff check backend/ ui/ --fix
```

**Format code:**
```bash
source venv/bin/activate
ruff format backend/ ui/
```

**Configuration:**
Ruff is configured in `pyproject.toml` with:
- Line length: 100 characters
- Python 3.11+ syntax
- Import sorting (isort)
- Modern Python upgrades
- Common bug detection

**Before committing:**
Always run `make lint` to catch issues early. Most issues can be auto-fixed with `ruff check --fix`.

## Useful Commands

**See all available make targets:**
```bash
make help
```

**Show configuration:**
```bash
make info
```

**Open UI in browser:**
```bash
make open-ui
```

**Open API docs:**
```bash
make open-backend
```

## Alternative Setup Methods

### Manual Backend Installation

**Terminal 1:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Verify correct venv
which python  # Should show: .../venv/bin/python

pip install -r requirements.txt
```

### Manual Frontend Installation

**Terminal 2 (or deactivate first):**
```bash
# If in same terminal: deactivate first
deactivate

cd frontend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Verify correct venv
which python  # Should show: .../frontend/venv/bin/python

pip install -r requirements.txt
```

### Manual Ollama Model Pull

The POC uses `llama3.1:8b` for intent extraction:

```bash
ollama pull llama3.1:8b
```

**Alternative models** (if needed):
- `llama3.2:3b` - Smaller/faster, less accurate
- `mistral:7b` - Good balance of speed and quality

### Verify Ollama Setup

```bash
# Test Ollama is working
ollama list  # Should show llama3.1:8b
```

## Running Services Manually

### Option 1: Run Full Stack with UI (Recommended)

The easiest way to use NeuralNav:

```bash
# Terminal 1 - Start Ollama (if not already running)
ollama serve

# Terminal 2 - Start FastAPI Backend
scripts/run_api.sh

# Terminal 3 - Start Streamlit UI
scripts/run_ui.sh
```

Then open http://localhost:8501 in your browser.

### Option 2: Test End-to-End Workflow

Test the complete recommendation workflow with demo scenarios:

```bash
cd backend
source venv/bin/activate
python test_workflow.py
```

This tests all 3 demo scenarios end-to-end.

### Option 3: Run FastAPI Backend Only

Start the API server:

```bash
./run_api.sh
```

Or manually:

```bash
scripts/run_api.sh
```

Test the API:

```bash
# Health check
curl http://localhost:8000/health

# Full recommendation
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"message": "I need a chatbot for 5000 users with low latency"}'
```

### Option 4: Test Individual Components

Test the LLM client:

```bash
cd backend
source venv/bin/activate
python -c "
from src.llm.ollama_client import OllamaClient
client = OllamaClient(model='llama3.2:3b')
print('Ollama available:', client.is_available())
print('Pulling model...')
client.ensure_model_pulled()
print('Model ready!')
"
```

## Troubleshooting

### Ollama Connection Issues

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# If not running
ollama serve
```

### Model Not Found

```bash
ollama pull llama3.2:3b
```

### Import Errors

```bash
# Make sure you're in the right venv
which python  # Should show path to venv

# Reinstall dependencies
pip install -r requirements.txt
```

## Manual Kubernetes Cluster Setup

### KIND Cluster Installation

**Install KIND (if not already installed):**
```bash
brew install kind
```

**Create cluster with KServe:**
```bash
# Ensure Docker Desktop is running

# Create cluster
kind create cluster --config config/kind-cluster.yaml

# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.14.4/cert-manager.yaml

# Wait for cert-manager
kubectl wait --for=condition=available --timeout=300s -n cert-manager deployment/cert-manager

# Install KServe
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.13.0/kserve.yaml
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.13.0/kserve-cluster-resources.yaml

# Wait for KServe
kubectl wait --for=condition=available --timeout=300s -n kserve deployment/kserve-controller-manager

# Configure KServe for RawDeployment mode
kubectl patch configmap/inferenceservice-config -n kserve --type=strategic -p '{"data": {"deploy": "{\"defaultDeploymentMode\": \"RawDeployment\"}"}}'
```

### Deploy Models Through UI

1. Get a deployment recommendation from the chat interface
2. Click **"Generate Deployment YAML"** in the Actions section
3. If cluster is accessible, click **"Deploy to Kubernetes"**
4. Go to **Monitoring** tab to see:
   - Real Kubernetes deployment status
   - InferenceService conditions
   - Pod information
   - Performance metrics

### Manual Deployment Commands

**Deploy generated YAML:**
```bash
# After generating YAML via UI
kubectl apply -f generated_configs/kserve-inferenceservice.yaml
kubectl get inferenceservices
kubectl get pods
```

**View all resources:**
```bash
kubectl get pods -A
```

**View deployments:**
```bash
kubectl get inferenceservices
kubectl get pods
```

**Delete a specific deployment:**
```bash
kubectl delete inferenceservice <deployment-id>
```

**Check cluster info:**
```bash
kubectl cluster-info
```

## YAML Deployment Generation

The system automatically generates production-ready Kubernetes configurations:

- ✅ KServe InferenceService YAML with vLLM configuration
- ✅ HorizontalPodAutoscaler (HPA) for autoscaling
- ✅ Prometheus ServiceMonitor for metrics collection
- ✅ Grafana Dashboard ConfigMap
- ✅ Full YAML validation before generation
- ✅ Files written to `generated_configs/` directory

**How to use:**
1. Get a deployment recommendation from the chat interface
2. Go to the **Cost** tab and click **"Generate Deployment YAML"**
3. View generated YAML file paths
4. Check `generated_configs/` directory for all YAML files

## vLLM Simulator Details

### Deploy a Model in Simulator Mode (default)

Simulator mode is enabled by default for all deployments:

```bash
# Start the UI
scripts/run_ui.sh

# In the UI:
# 1. Get a deployment recommendation
# 2. Click "Generate Deployment YAML"
# 3. Click "Deploy to Kubernetes"
# 4. Go to Monitoring tab
# 5. Pod should become Ready in ~10-15 seconds
```

### Test Inference

Once deployed:
1. Go to **Monitoring** tab
2. See "🧪 Inference Testing" section
3. Enter a test prompt
4. Click "🚀 Send Test Request"
5. View the simulated response and metrics

### Switch to Real vLLM

To use real vLLM with actual GPUs (requires GPU-enabled cluster):

```python
# In backend/src/api/routes.py
deployment_generator = DeploymentGenerator(simulator_mode=False)
```

Then deploy to a GPU-enabled cluster with:
- NVIDIA GPU Operator installed
- GPU nodes with appropriate labels
- Sufficient GPU resources

### Simulator vs Real vLLM

| Feature | Simulator Mode | Real vLLM Mode |
|---------|---------------|----------------|
| GPU Required | ❌ No | ✅ Yes |
| Model Download | ❌ No | ✅ Yes (from HuggingFace) |
| Inference | Canned responses | Real generation |
| Latency | Simulated (from benchmarks) | Actual GPU performance |
| Use Case | Development, testing, demos | Production deployment |
| Cluster | Works on KIND (local) | Requires GPU-enabled cluster |

## Testing Details

### Quick Tests

```bash
# Test end-to-end workflow
cd backend && source venv/bin/activate
cd ..
python tests/test_workflow.py

# Test FastAPI endpoints
scripts/run_api.sh  # Start server in terminal 1
# In terminal 2:
curl -X POST http://localhost:8000/api/v1/test
```

For comprehensive testing instructions, see [backend/TESTING.md](../backend/TESTING.md).
