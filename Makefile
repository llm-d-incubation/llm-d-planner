# Planner - Makefile
#
# This Makefile provides common development tasks for Planner.
# Supports macOS and Linux.

.PHONY: help
.DEFAULT_GOAL := help

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
    OPEN_CMD := open
    PYTHON := python3
else ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
    OPEN_CMD := xdg-open
    PYTHON := python3
else
    $(error Unsupported platform: $(UNAME_S). Please use macOS or Linux (or WSL2 on Windows))
endif

# Container runtime detection
# - CONTAINER_TOOL: Used for simulator builds and general container operations
# - KIND always requires Docker (it creates containers to simulate K8s nodes)
#
# Auto-detection prefers docker if running (for KIND compatibility), falls back to podman if running
# Only selects a runtime if its daemon is actually running
CONTAINER_TOOL ?= $(shell \
	if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then \
		echo docker; \
	elif command -v podman >/dev/null 2>&1 && podman info >/dev/null 2>&1; then \
		echo podman; \
	else \
		echo ""; \
	fi)

# Configuration
REGISTRY ?= quay.io
REGISTRY_ORG ?= llm-d-planner

BACKEND_IMAGE ?= llm-d-planner-backend
BACKEND_TAG ?= latest
BACKEND_FULL_IMAGE := $(REGISTRY)/$(REGISTRY_ORG)/$(BACKEND_IMAGE):$(BACKEND_TAG)

UI_IMAGE ?= llm-d-planner-ui
UI_TAG ?= latest
UI_FULL_IMAGE := $(REGISTRY)/$(REGISTRY_ORG)/$(UI_IMAGE):$(UI_TAG)

SIMULATOR_IMAGE ?= vllm-simulator
SIMULATOR_TAG ?= latest
SIMULATOR_FULL_IMAGE := $(REGISTRY)/$(REGISTRY_ORG)/$(SIMULATOR_IMAGE):$(SIMULATOR_TAG)

OLLAMA_MODEL ?= qwen2.5:7b
KIND_CLUSTER_NAME ?= planner


SRC_DIR := src
UI_DIR := ui
TEST_DIR := tests
SIMULATOR_DIR := simulator

VENV := .venv
# Shared venv at project root for both backend and UI (managed by uv)

# PID files for background processes
PID_DIR := .pids
OLLAMA_PID := $(PID_DIR)/ollama.pid
BACKEND_PID := $(PID_DIR)/backend.pid
UI_PID := $(PID_DIR)/ui.pid

# Log directory
LOG_DIR := logs

# Allow local overrides via .env file
-include .env
export

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

##@ Help

help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\n$(BLUE)Usage:$(NC)\n  make $(GREEN)<target>$(NC)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup & Installation

check-prereqs: ## Check if required tools are installed
	@printf "$(BLUE)Checking prerequisites...$(NC)\n"
	@[ -n "$(CONTAINER_TOOL)" ] || (printf "$(RED)✗ Docker or Podman not found$(NC).\n" && exit 1)
	@printf "$(GREEN)✓ $(CONTAINER_TOOL) found (container runtime)$(NC)\n"
	@if [ "$(CONTAINER_TOOL)" = "podman" ]; then \
		if command -v docker >/dev/null 2>&1; then \
			printf "$(GREEN)✓ docker also found (required for KIND cluster)$(NC)\n"; \
		else \
			printf "$(YELLOW)⚠ Docker not found - KIND cluster commands will not work$(NC)\n"; \
		fi; \
	fi
	@command -v kubectl >/dev/null 2>&1 || (printf "$(RED)✗ kubectl not found$(NC). Run: brew install kubectl\n" && exit 1)
	@printf "$(GREEN)✓ kubectl found$(NC)\n"
	@command -v kind >/dev/null 2>&1 || (printf "$(RED)✗ kind not found$(NC). Run: brew install kind\n" && exit 1)
	@printf "$(GREEN)✓ kind found$(NC)\n"
	@command -v ollama >/dev/null 2>&1 || (printf "$(RED)✗ ollama not found$(NC). Run: brew install ollama\n" && exit 1)
	@printf "$(GREEN)✓ ollama found$(NC)\n"
	@command -v $(PYTHON) >/dev/null 2>&1 || (printf "$(RED)✗ $(PYTHON) not found$(NC). Run: brew install python@3.13\n" && exit 1)
	@PY_MINOR=$$($(PYTHON) -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo "0"); \
	if [ "$$PY_MINOR" -lt "13" ]; then \
		printf "$(RED)✗ Python >= 3.13 required, found 3.$$PY_MINOR$(NC). Run: brew install python@3.13\n" && exit 1; \
	fi
	@printf "$(GREEN)✓ $(PYTHON) found ($$($(PYTHON) --version 2>&1))$(NC)\n"
	@command -v uv >/dev/null 2>&1 || (printf "$(RED)✗ uv not found$(NC). Run: curl -LsSf https://astral.sh/uv/install.sh | sh\n" && exit 1)
	@printf "$(GREEN)✓ uv found$(NC)\n"
	@$(CONTAINER_TOOL) info >/dev/null 2>&1 || (printf "$(RED)✗ Docker or Podman daemon not running$(NC).\n" && exit 1)
	@printf "$(GREEN)✓ $(CONTAINER_TOOL) daemon running$(NC)\n"
	@command -v docker-compose >/dev/null 2>&1 || docker compose version >/dev/null 2>&1 || (printf "$(RED)✗ docker compose not found$(NC). Install Docker Compose or update Docker Desktop\n" && exit 1)
	@printf "$(GREEN)✓ docker compose found$(NC)\n"
	@printf "$(GREEN)All prerequisites satisfied!$(NC)\n"

setup-backend: ## Set up Python environment (all dependencies including optional providers)
	@printf "$(BLUE)Setting up Python environment...$(NC)\n"
	uv sync --extra server --extra ui --extra llm --extra openai --extra vertex --extra dev --extra estimation --extra quality-sync --extra kubernetes
	uv pip install "llm-optimizer @ git+https://github.com/bentoml/llm-optimizer.git"
	@printf "$(GREEN)✓ Python environment ready (includes all dependencies)$(NC)\n"

setup-vertex: ## Install Vertex AI dependencies (only needed for LLM_PROVIDER=vertex)
	@printf "$(BLUE)Installing Vertex AI dependencies...$(NC)\n"
	uv sync --extra vertex
	@printf "$(GREEN)✓ Vertex AI dependencies installed$(NC)\n"

setup-ui: setup-backend ## Set up UI (uses shared venv)
	@printf "$(GREEN)✓ UI ready (shares project venv)$(NC)\n"

setup-ollama: ## Pull Ollama model (skipped when LLM_PROVIDER != ollama)
	@if [ "$(LLM_PROVIDER)" != "" ] && [ "$(LLM_PROVIDER)" != "ollama" ]; then \
		printf "$(YELLOW)Skipping Ollama setup (LLM_PROVIDER=$(LLM_PROVIDER))$(NC)\n"; \
	else \
		printf "$(BLUE)Checking if Ollama model $(OLLAMA_MODEL) is available...$(NC)\n"; \
		if ! pgrep -x "ollama" > /dev/null; then \
			printf "$(YELLOW)Starting Ollama service...$(NC)\n"; \
			ollama serve > /dev/null 2>&1 & \
			sleep 2; \
		fi; \
		ollama list | grep -q $(OLLAMA_MODEL) || (printf "$(YELLOW)Pulling model $(OLLAMA_MODEL)...$(NC)\n" && ollama pull $(OLLAMA_MODEL)); \
		printf "$(GREEN)✓ Ollama model $(OLLAMA_MODEL) ready$(NC)\n"; \
	fi

setup: check-prereqs setup-backend setup-ui setup-ollama ## Run all setup tasks
	@printf "$(GREEN)✓ Setup complete!$(NC)\n"
	@printf "\n"
	@printf "$(BLUE)Next steps:$(NC)\n"
	@printf "  make cluster-start # Create Kubernetes cluster\n"
	@printf "  make start         # Start all services\n"

##@ Development

start: db-start setup-ollama ## Start all services (DB + Ollama + Backend + UI)
	@printf "$(BLUE)Starting all services...$(NC)\n"
	@mkdir -p $(PID_DIR)
	@if [ "$(LLM_PROVIDER)" = "" ] || [ "$(LLM_PROVIDER)" = "ollama" ]; then \
		$(MAKE) start-ollama; \
	else \
		printf "$(YELLOW)Skipping Ollama (LLM_PROVIDER=$(LLM_PROVIDER))$(NC)\n"; \
	fi
	@sleep 3
	@$(MAKE) start-backend
	@sleep 3
	@$(MAKE) start-ui
	@printf "\n"
	@printf "$(GREEN)✓ All services started!$(NC)\n"
	@printf "\n"
	@printf "$(BLUE)Service URLs:$(NC)\n"
	@printf "  UI:      http://localhost:8501\n"
	@printf "  Backend: http://localhost:8000\n"
	@if [ "$(LLM_PROVIDER)" = "" ] || [ "$(LLM_PROVIDER)" = "ollama" ]; then \
		printf "  Ollama:  http://localhost:11434\n"; \
	fi
	@printf "  DB:      $(DB_PATH)\n"
	@printf "\n"
	@printf "$(BLUE)Logs:$(NC)\n"
	@printf "  make logs-backend\n"
	@printf "  make logs-ui\n"
	@printf "\n"
	@printf "$(BLUE)Stop:$(NC)\n"
	@printf "  make stop          # Stop Backend + UI (leaves Ollama and DB running)\n"
	@printf "  make stop-all      # Stop everything including Ollama and DB\n"

start-ollama: ## Start Ollama service
	@printf "$(BLUE)Starting Ollama...$(NC)\n"
	@if pgrep -x "ollama" > /dev/null; then \
		printf "$(YELLOW)Ollama already running$(NC)\n"; \
	else \
		ollama serve > /dev/null 2>&1 & echo $$! > $(OLLAMA_PID); \
		printf "$(GREEN)✓ Ollama started (PID: $$(cat $(OLLAMA_PID)))$(NC)\n"; \
	fi

start-backend: ## Start FastAPI backend
	@printf "$(BLUE)Starting backend...$(NC)\n"
	@mkdir -p $(PID_DIR) $(LOG_DIR)
	@if [ -f $(BACKEND_PID) ] && [ -s $(BACKEND_PID) ] && kill -0 $$(cat $(BACKEND_PID) 2>/dev/null) 2>/dev/null; then \
		printf "$(YELLOW)Backend already running (PID: $$(cat $(BACKEND_PID)))$(NC)\n"; \
	else \
		( PYTHONPATH=src uv run uvicorn planner.api.app:app --reload --host 0.0.0.0 --port 8000 > $(LOG_DIR)/backend.log 2>&1 & echo $$! > $(BACKEND_PID) ); \
		sleep 2; \
		printf "$(GREEN)✓ Backend started (PID: $$(cat $(BACKEND_PID)))$(NC)\n"; \
	fi

start-ui: ## Start Streamlit UI
	@printf "$(BLUE)Starting UI...$(NC)\n"
	@mkdir -p $(PID_DIR) $(LOG_DIR)
	@if [ -f $(UI_PID) ] && [ -s $(UI_PID) ] && kill -0 $$(cat $(UI_PID) 2>/dev/null) 2>/dev/null; then \
		printf "$(YELLOW)UI already running (PID: $$(cat $(UI_PID)))$(NC)\n"; \
	else \
		uv run streamlit run $(UI_DIR)/main.py --server.headless true > $(LOG_DIR)/ui.log 2>&1 & echo $$! > $(UI_PID); \
		sleep 2; \
		printf "$(GREEN)✓ UI started (PID: $$(cat $(UI_PID)))$(NC)\n"; \
	fi

stop: ## Stop Backend + UI (leaves Ollama and DB running)
	@printf "$(BLUE)Stopping services...$(NC)\n"
	@# Stop by PID files first
	@if [ -f $(UI_PID) ]; then \
		kill $$(cat $(UI_PID)) 2>/dev/null || true; \
		rm -f $(UI_PID); \
	fi
	@if [ -f $(BACKEND_PID) ]; then \
		kill $$(cat $(BACKEND_PID)) 2>/dev/null || true; \
		rm -f $(BACKEND_PID); \
	fi
	@# Kill any remaining Planner processes by pattern matching
	@pkill -f "streamlit run ui/main.py" 2>/dev/null || true
	@pkill -f "uvicorn planner.api.app:app" 2>/dev/null || true
	@# Give processes time to exit gracefully
	@sleep 1
	@# Force kill if still running
	@pkill -9 -f "streamlit run ui/main.py" 2>/dev/null || true
	@pkill -9 -f "uvicorn planner.api.app:app" 2>/dev/null || true
	@printf "$(GREEN)✓ All Planner services stopped$(NC)\n"
	@# Don't stop Ollama as it might be used by other apps/tools
	@if [ "$(MAKECMDGOALS)" != "stop-all" ]; then \
		printf "$(YELLOW)Note: Ollama left running (use 'make stop-all' to stop everything)$(NC)\n"; \
	fi

restart: stop start ## Restart all services

stop-all: stop ## Stop everything (Backend + UI + Ollama)
	@pkill -x ollama 2>/dev/null || true
	@printf "$(GREEN)✓ Ollama stopped$(NC)\n"
	@printf "$(GREEN)✓ All services and infrastructure stopped$(NC)\n"

logs-backend: ## Show backend logs (dump current log)
	@cat $(LOG_DIR)/backend.log

logs-backend-f: ## Follow backend logs (tail -f)
	@tail -f $(LOG_DIR)/backend.log

logs-ui: ## Show UI logs (dump current log)
	@cat $(LOG_DIR)/ui.log

logs-ui-f: ## Follow UI logs (tail -f)
	@tail -f $(LOG_DIR)/ui.log

health: ## Check if all services are running
	@printf "$(BLUE)Checking service health...$(NC)\n"
	@curl -s http://localhost:8000/health > /dev/null && printf "$(GREEN)✓ Backend healthy$(NC)\n" || printf "$(RED)✗ Backend not responding$(NC)\n"
	@curl -s http://localhost:8501 > /dev/null && printf "$(GREEN)✓ UI healthy$(NC)\n" || printf "$(RED)✗ UI not responding$(NC)\n"
	@curl -s http://localhost:11434/api/tags > /dev/null && printf "$(GREEN)✓ Ollama healthy$(NC)\n" || printf "$(RED)✗ Ollama not responding$(NC)\n"

open-ui: ## Open UI in browser
	@$(OPEN_CMD) http://localhost:8501

open-backend: ## Open backend API docs in browser
	@$(OPEN_CMD) http://localhost:8000/docs

##@ Container Images

image-build-backend: ## Build backend container image
	@printf "$(BLUE)Building backend image...$(NC)\n"
	@PKG_VERSION=$$(uv run python -c "from importlib.metadata import version; print(version('llm-d-planner'))" 2>/dev/null) && \
	  [ -n "$$PKG_VERSION" ] || { printf "$(RED)Error: Could not determine llm-d-planner version. Run 'make setup-backend' first.$(NC)\n"; exit 1; } && \
	  $(CONTAINER_TOOL) build --platform linux/amd64 --build-arg SETUPTOOLS_SCM_PRETEND_VERSION=$$PKG_VERSION -f Dockerfile -t $(BACKEND_IMAGE):$(BACKEND_TAG) -t $(BACKEND_FULL_IMAGE) .
	@printf "$(GREEN)✓ Backend image built:$(NC)\n"
	@printf "  - $(BACKEND_IMAGE):$(BACKEND_TAG)\n"
	@printf "  - $(BACKEND_FULL_IMAGE)\n"

image-build-ui: ## Build UI container image
	@printf "$(BLUE)Building UI image...$(NC)\n"
	$(CONTAINER_TOOL) build --platform linux/amd64 -f ui/Dockerfile -t $(UI_IMAGE):$(UI_TAG) -t $(UI_FULL_IMAGE) .
	@printf "$(GREEN)✓ UI image built:$(NC)\n"
	@printf "  - $(UI_IMAGE):$(UI_TAG)\n"
	@printf "  - $(UI_FULL_IMAGE)\n"

image-build-simulator: ## Build vLLM simulator container image
	@printf "$(BLUE)Building simulator image...$(NC)\n"
	$(CONTAINER_TOOL) build --platform linux/amd64 -f $(SIMULATOR_DIR)/Dockerfile -t vllm-simulator:latest -t $(SIMULATOR_FULL_IMAGE) .
	@printf "$(GREEN)✓ Simulator image built:$(NC)\n"
	@printf "  - vllm-simulator:latest\n"
	@printf "  - $(SIMULATOR_FULL_IMAGE)\n"

image-build: image-build-backend image-build-ui image-build-simulator ## Build all container images

image-push-backend: image-build-backend ## Push backend image to Quay.io
	@printf "$(BLUE)Pushing backend image to $(BACKEND_FULL_IMAGE)...$(NC)\n"
	@if ! $(CONTAINER_TOOL) login quay.io --get-login > /dev/null 2>&1; then \
		printf "$(YELLOW)Not logged in to Quay.io. Please login:$(NC)\n"; \
		$(CONTAINER_TOOL) login quay.io || (printf "$(RED)✗ Login failed$(NC)\n" && exit 1); \
	else \
		printf "$(GREEN)✓ Already logged in to Quay.io$(NC)\n"; \
	fi
	@printf "$(BLUE)Pushing image...$(NC)\n"
	$(CONTAINER_TOOL) push $(BACKEND_FULL_IMAGE)
	@printf "$(GREEN)✓ Image pushed to $(BACKEND_FULL_IMAGE)$(NC)\n"

image-push-ui: image-build-ui ## Push UI image to Quay.io
	@printf "$(BLUE)Pushing UI image to $(UI_FULL_IMAGE)...$(NC)\n"
	@if ! $(CONTAINER_TOOL) login quay.io --get-login > /dev/null 2>&1; then \
		printf "$(YELLOW)Not logged in to Quay.io. Please login:$(NC)\n"; \
		$(CONTAINER_TOOL) login quay.io || (printf "$(RED)✗ Login failed$(NC)\n" && exit 1); \
	else \
		printf "$(GREEN)✓ Already logged in to Quay.io$(NC)\n"; \
	fi
	@printf "$(BLUE)Pushing image...$(NC)\n"
	$(CONTAINER_TOOL) push $(UI_FULL_IMAGE)
	@printf "$(GREEN)✓ Image pushed to $(UI_FULL_IMAGE)$(NC)\n"

image-push-simulator: image-build-simulator ## Push simulator image to Quay.io
	@printf "$(BLUE)Pushing simulator image to $(SIMULATOR_FULL_IMAGE)...$(NC)\n"
	@if ! $(CONTAINER_TOOL) login quay.io --get-login > /dev/null 2>&1; then \
		printf "$(YELLOW)Not logged in to Quay.io. Please login:$(NC)\n"; \
		$(CONTAINER_TOOL) login quay.io || (printf "$(RED)✗ Login failed$(NC)\n" && exit 1); \
	else \
		printf "$(GREEN)✓ Already logged in to Quay.io$(NC)\n"; \
	fi
	@printf "$(BLUE)Pushing image...$(NC)\n"
	$(CONTAINER_TOOL) push $(SIMULATOR_FULL_IMAGE)
	@printf "$(GREEN)✓ Image pushed to $(SIMULATOR_FULL_IMAGE)$(NC)\n"

image-push: image-push-backend image-push-ui image-push-simulator ## Push all container images to Quay.io

##@ Docker Compose

docker-up: ## Start all services with Docker Compose
	@printf "$(BLUE)Starting all services with Docker Compose...$(NC)\n"
	docker-compose up -d
	@printf "$(GREEN)✓ All services started!$(NC)\n"
	@printf "\n"
	@printf "$(BLUE)Service URLs:$(NC)\n"
	@printf "  UI:        http://localhost:8501\n"
	@printf "  Backend:   http://localhost:8000\n"
	@printf "  API Docs:  http://localhost:8000/docs\n"
	@printf "  Ollama:    http://localhost:11434\n"
	@printf "\n"
	@printf "$(BLUE)Logs:$(NC)\n"
	@printf "  make docker-logs\n"
	@printf "\n"
	@printf "$(BLUE)Stop:$(NC)\n"
	@printf "  make docker-down\n"

docker-up-dev: ## Start all services with Docker Compose (development mode)
	@printf "$(BLUE)Starting services in development mode...$(NC)\n"
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --force-recreate
	@printf "$(GREEN)✓ Development environment started$(NC)\n"

docker-down: ## Stop all Docker Compose services
	@printf "$(BLUE)Stopping Docker Compose services...$(NC)\n"
	docker-compose down
	@printf "$(GREEN)✓ All services stopped$(NC)\n"

docker-down-v: ## Stop and remove volumes
	@printf "$(BLUE)Stopping services and removing volumes...$(NC)\n"
	@printf "$(YELLOW)WARNING: This will delete all database data!$(NC)\n"
	docker-compose down -v
	@printf "$(GREEN)✓ Services stopped and volumes removed$(NC)\n"

docker-logs: ## View logs from all Docker Compose services
	@docker-compose logs -f

docker-ps: ## Show status of Docker Compose services
	@docker-compose ps

##@ Kubernetes Cluster

cluster-start: check-prereqs image-build-simulator ## Create KIND cluster and load simulator image
	@printf "$(BLUE)Creating KIND cluster...$(NC)\n"
	./scripts/kind-cluster.sh start
	@printf "$(GREEN)✓ Cluster ready!$(NC)\n"

cluster-stop: ## Delete KIND cluster
	@printf "$(BLUE)Stopping KIND cluster...$(NC)\n"
	./scripts/kind-cluster.sh stop
	@printf "$(GREEN)✓ Cluster deleted$(NC)\n"

cluster-restart: ## Restart KIND cluster
	@printf "$(BLUE)Restarting KIND cluster...$(NC)\n"
	./scripts/kind-cluster.sh restart
	@printf "$(GREEN)✓ Cluster restarted$(NC)\n"

cluster-status: ## Show cluster status
	./scripts/kind-cluster.sh status

cluster-load-simulator: image-build-simulator ## Load simulator image into KIND cluster
	@printf "$(BLUE)Loading simulator image into KIND cluster...$(NC)\n"
	kind load docker-image vllm-simulator:latest --name "$${KIND_CLUSTER_NAME:?KIND_CLUSTER_NAME must not be empty}"
	@printf "$(GREEN)✓ Simulator image loaded$(NC)\n"

clean-deployments: ## Delete all InferenceServices from cluster
	@printf "$(BLUE)Deleting all InferenceServices...$(NC)\n"
	kubectl delete inferenceservices --all
	@printf "$(GREEN)✓ All deployments deleted$(NC)\n"

##@ Data

DB_PATH ?= data/planner.db

db-start: ## Initialize database (creates file and applies schema if needed)
	@printf "$(BLUE)Initializing database...$(NC)\n"
	@mkdir -p $$(dirname $(DB_PATH))
	@if [ -f $(DB_PATH) ]; then \
		printf "$(YELLOW)Database already exists at $(DB_PATH)$(NC)\n"; \
	else \
		uv run python -c "from planner.knowledge_base.db import create_connection; import os; os.environ['PLANNER_DB_PATH'] = '$(DB_PATH)'; conn = create_connection(); conn.close(); print('Schema initialized')"; \
		printf "$(GREEN)✓ Database created at $(DB_PATH)$(NC)\n"; \
	fi
	@BENCH_COUNT=$$(sqlite3 $(DB_PATH) "SELECT COUNT(*) FROM exported_summaries;" 2>/dev/null || echo "0"); \
	if [ "$$BENCH_COUNT" = "0" ]; then \
		printf "$(YELLOW)Note: Database is empty. Load benchmark data with one of:$(NC)\n"; \
		printf "  make db-load-blis          # BLIS benchmark data\n"; \
		printf "  make db-load-estimated     # Estimated performance data\n"; \
		printf "  make db-load-interpolated  # Interpolated benchmark data\n"; \
	fi

db-remove: ## Remove database file
	@printf "$(BLUE)Removing database...$(NC)\n"
	@rm -f $(DB_PATH) $(DB_PATH)-wal $(DB_PATH)-shm
	@printf "$(GREEN)✓ Database removed$(NC)\n"

db-load-blis: db-start ## Load BLIS benchmark data (appends)
	@printf "$(BLUE)Loading BLIS benchmark data...$(NC)\n"
	@PLANNER_DB_PATH=$(DB_PATH) uv run python scripts/load_benchmarks.py src/planner/data/performance/benchmarks_BLIS.json
	@printf "$(GREEN)✓ BLIS data loaded$(NC)\n"

db-load-estimated: db-start ## Load estimated performance benchmarks (appends)
	@printf "$(BLUE)Loading estimated performance data...$(NC)\n"
	@PLANNER_DB_PATH=$(DB_PATH) uv run python scripts/load_benchmarks.py src/planner/data/performance/benchmarks_estimated_performance.json
	@printf "$(GREEN)✓ Estimated data loaded$(NC)\n"

db-load-interpolated: db-start ## Load interpolated benchmark data (appends)
	@printf "$(BLUE)Loading interpolated benchmark data...$(NC)\n"
	@PLANNER_DB_PATH=$(DB_PATH) uv run python scripts/load_benchmarks.py src/planner/data/performance/benchmarks_interpolated_v2.json
	@printf "$(GREEN)✓ Interpolated data loaded$(NC)\n"

db-load-guidellm: db-start ## Load GuideLLM benchmark data (appends)
	@printf "$(BLUE)Loading GuideLLM benchmark data...$(NC)\n"
	@if [ ! -f src/planner/data/performance/benchmarks_GuideLLM.json ]; then \
		printf "$(RED)✗ src/planner/data/performance/benchmarks_GuideLLM.json not found$(NC)\n"; \
		exit 1; \
	fi
	@PLANNER_DB_PATH=$(DB_PATH) uv run python scripts/load_benchmarks.py src/planner/data/performance/benchmarks_GuideLLM.json
	@printf "$(GREEN)✓ GuideLLM data loaded$(NC)\n"

db-shell: ## Open database shell
	@sqlite3 $(DB_PATH)

db-query-traffic: ## Query unique traffic patterns from database
	@printf "$(BLUE)Querying unique traffic patterns...$(NC)\n"
	@sqlite3 -header -column $(DB_PATH) \
		"SELECT prompt_tokens, output_tokens, COUNT(*) as num_benchmarks \
		FROM exported_summaries \
		GROUP BY prompt_tokens, output_tokens \
		ORDER BY prompt_tokens, output_tokens;"

db-query-models: ## Query available models in database
	@printf "$(BLUE)Querying available models...$(NC)\n"
	@sqlite3 -header -column $(DB_PATH) \
		"SELECT model_hf_repo, hardware, hardware_count, COUNT(*) as num_benchmarks \
		FROM exported_summaries \
		GROUP BY model_hf_repo, hardware, hardware_count \
		ORDER BY model_hf_repo, hardware, hardware_count;"

db-reset: db-start ## Reset database (clear all benchmark data, safe while backend is running)
	@printf "$(BLUE)Clearing benchmark data...$(NC)\n"
	@sqlite3 $(DB_PATH) "DELETE FROM exported_summaries;"
	@printf "$(GREEN)✓ Database reset complete$(NC)\n"

quality-sync: ## Refresh checked-in quality benchmark data (Arena + AA)
	@printf "$(BLUE)Syncing Arena leaderboard (no API key needed)...$(NC)\n"
	@LLM_QUALITY_CACHE_DIR=src/quality_scoring/data uv run python -c "from quality_scoring.arena_client import sync; count, path = sync(); print(f'Arena: {count} rows')"
	@printf "$(BLUE)Syncing AA models (requires AA_API_KEY)...$(NC)\n"
	@if [ -n "$$AA_API_KEY" ]; then \
		LLM_QUALITY_CACHE_DIR=src/quality_scoring/data uv run python -c "from quality_scoring.aa_client import sync; count, path = sync(api_key='$$AA_API_KEY'); print(f'AA: {count} models')"; \
	else \
		printf "$(YELLOW)⚠ AA_API_KEY not set — skipping AA sync$(NC)\n"; \
	fi
	@printf "$(BLUE)Formatting and sorting JSON files for readable diffs...$(NC)\n"
	@uv run python scripts/format_quality_data.py
	@printf "$(GREEN)✓ Quality data synced to src/quality_scoring/data/$(NC)\n"

##@ Testing

test: test-unit test-hf test-integration ## Run all tests (requires LLM)
	@printf "$(GREEN)✓ All tests passed$(NC)\n"

test-unit: ## Run unit tests (no external dependencies)
	@printf "$(BLUE)Running unit tests...$(NC)\n"
	cd $(SRC_DIR) && uv run pytest ../tests/ -v -m unit

test-hf: ## Run HuggingFace tests (requires network, retries flaky failures)
	@printf "$(BLUE)Running HuggingFace tests...$(NC)\n"
	cd $(SRC_DIR) && uv run pytest ../tests/ -v -m hf_network --reruns 2

test-integration: setup-ollama ## Run integration tests (requires LLM)
	@printf "$(BLUE)Running integration tests...$(NC)\n"
	cd $(SRC_DIR) && uv run pytest ../tests/ -v -m integration

test-intent: setup-ollama ## Run intent extraction tests (requires LLM)
	@printf "$(BLUE)Running intent extraction tests...$(NC)\n"
	cd $(SRC_DIR) && uv run pytest ../tests/ -v -m intent_extraction

##@ Code Quality

lint: ## Run linters
	@printf "$(BLUE)Running linters...$(NC)\n"
	@if uv run ruff --version >/dev/null 2>&1; then uv run ruff check $(SRC_DIR)/ $(TEST_DIR)/ $(UI_DIR)/; else printf "$(YELLOW)ruff not installed, skipping$(NC)\n"; fi
	@printf "$(GREEN)✓ Linting complete$(NC)\n"

lint-fix: ## Run linters with auto-fix
	@printf "$(BLUE)Running linters with auto-fix...$(NC)\n"
	@if uv run ruff --version >/dev/null 2>&1; then uv run ruff check --fix $(SRC_DIR)/ $(TEST_DIR)/ $(UI_DIR)/; else printf "$(YELLOW)ruff not installed, skipping$(NC)\n"; fi
	@printf "$(GREEN)✓ Lint fix complete$(NC)\n"

format: ## Auto-format code
	@printf "$(BLUE)Formatting code...$(NC)\n"
	@if uv run ruff --version >/dev/null 2>&1; then uv run ruff format $(SRC_DIR)/ $(TEST_DIR)/ $(UI_DIR)/; else printf "$(YELLOW)ruff not installed, skipping$(NC)\n"; fi
	@printf "$(GREEN)✓ Formatting complete$(NC)\n"

typecheck:  ## Run typecheck
	@printf "$(BLUE)Running typecheck...$(NC)\n"
	@if uv run mypy --version >/dev/null 2>&1; then uv run mypy $(SRC_DIR)/ $(UI_DIR)/ $(TEST_DIR)/; else printf "$(YELLOW)mypy not installed, skipping$(NC)\n"; fi
	@printf "$(GREEN)✓ Typecheck complete$(NC)\n"

##@ Cleanup

clean: ## Clean generated files and caches
	@printf "$(BLUE)Cleaning generated files...$(NC)\n"
	rm -rf $(PID_DIR)
	rm -f $(BACKEND_PID).log $(UI_PID).log
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf generated_configs/*.yaml 2>/dev/null || true
	rm -rf logs/prompts/*.txt 2>/dev/null || true
	@printf "$(GREEN)✓ Cleanup complete$(NC)\n"

clean-all: clean ## Clean everything including virtual environments
	@printf "$(BLUE)Cleaning virtual environments (uv-managed)...$(NC)\n"
	rm -rf $(VENV)
	@printf "$(GREEN)✓ Deep cleanup complete$(NC)\n"

##@ Utilities

info: ## Show configuration and platform info
	@printf "$(BLUE)Platform Information:$(NC)\n"
	@printf "  Platform: $(PLATFORM)\n"
	@printf "  OS: $(UNAME_S)\n"
	@printf "  Arch: $(UNAME_M)\n"
	@printf "  Python: $(PYTHON) ($$($(PYTHON) --version 2>&1))\n"
	@printf "\n"
	@printf "$(BLUE)Configuration:$(NC)\n"
	@printf "  Registry: $(REGISTRY)\n"
	@printf "  Org: $(REGISTRY_ORG)\n"
	@printf "  Backend Image: $(BACKEND_FULL_IMAGE)\n"
	@printf "  Simulator Image: $(SIMULATOR_FULL_IMAGE)\n"
	@printf "  Ollama Model: $(OLLAMA_MODEL)\n"
	@printf "  KIND Cluster: $(KIND_CLUSTER_NAME)\n"
	@printf "\n"
	@printf "$(BLUE)Paths:$(NC)\n"
	@printf "  Source: $(SRC_DIR)\n"
	@printf "  UI: $(UI_DIR)\n"
	@printf "  Simulator: $(SIMULATOR_DIR)\n"
