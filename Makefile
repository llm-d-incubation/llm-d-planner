# NeuralNav - Makefile
#
# This Makefile provides common development tasks for NeuralNav.
# Supports macOS and Linux.

.PHONY: help
.DEFAULT_GOAL := help

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
    OPEN_CMD := open
    PYTHON := python3.13
else ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
    OPEN_CMD := xdg-open
    PYTHON := python3
else
    $(error Unsupported platform: $(UNAME_S). Please use macOS or Linux (or WSL2 on Windows))
endif

# Container runtime detection
# - CONTAINER_TOOL: Used for PostgreSQL, simulator builds, and general container operations
#   Can be overridden: CONTAINER_TOOL=podman make db-start
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
REGISTRY_ORG ?= vllm-assistant
SIMULATOR_IMAGE ?= vllm-simulator
SIMULATOR_TAG ?= latest
SIMULATOR_FULL_IMAGE := $(REGISTRY)/$(REGISTRY_ORG)/$(SIMULATOR_IMAGE):$(SIMULATOR_TAG)

OLLAMA_MODEL ?= qwen2.5:7b
KIND_CLUSTER_NAME ?= compass-poc

PGDUMP_INPUT ?= data/integ-oct-29.sql
PGDUMP_OUTPUT ?= data/benchmarks_GuideLLM.json

BACKEND_DIR := backend
UI_DIR := ui
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
	@printf "$(GREEN)✓ $(PYTHON) found$(NC)\n"
	@command -v uv >/dev/null 2>&1 || (printf "$(RED)✗ uv not found$(NC). Run: curl -LsSf https://astral.sh/uv/install.sh | sh\n" && exit 1)
	@printf "$(GREEN)✓ uv found$(NC)\n"
	@# Check Python version and warn if 3.14+
	@PY_VERSION=$$($(PYTHON) -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0"); \
	if [ "$$(echo "$$PY_VERSION" | cut -d. -f1)" = "3" ] && [ "$$(echo "$$PY_VERSION" | cut -d. -f2)" -ge "14" ]; then \
		printf "$(YELLOW)⚠ Warning: Python $$PY_VERSION detected. Some dependencies (psycopg2-binary) may not have wheels yet.$(NC)\n"; \
		printf "$(YELLOW)  Recommend using Python 3.13 for best compatibility.$(NC)\n"; \
	fi
	@$(CONTAINER_TOOL) info >/dev/null 2>&1 || (printf "$(RED)✗ Docker or Podman daemon not running$(NC).\n" && exit 1)
	@printf "$(GREEN)✓ $(CONTAINER_TOOL) daemon running$(NC)\n"
	@command -v docker-compose >/dev/null 2>&1 || docker compose version >/dev/null 2>&1 || (printf "$(RED)✗ docker compose not found$(NC). Install Docker Compose or update Docker Desktop\n" && exit 1)
	@printf "$(GREEN)✓ docker compose found$(NC)\n"
	@printf "$(GREEN)All prerequisites satisfied!$(NC)\n"

setup-backend: ## Set up Python environment (includes backend and UI dependencies)
	@printf "$(BLUE)Setting up Python environment...$(NC)\n"
	uv sync
	@printf "$(GREEN)✓ Python environment ready (includes backend and UI dependencies)$(NC)\n"

setup-ui: setup-backend ## Set up UI (uses shared venv)
	@printf "$(GREEN)✓ UI ready (shares project venv)$(NC)\n"

setup-ollama: ## Pull Ollama model
	@printf "$(BLUE)Checking if Ollama model $(OLLAMA_MODEL) is available...$(NC)\n"
	@# Start ollama if not running
	@if ! pgrep -x "ollama" > /dev/null; then \
		printf "$(YELLOW)Starting Ollama service...$(NC)\n"; \
		ollama serve > /dev/null 2>&1 & \
		sleep 2; \
	fi
	@# Check if model exists, pull if not
	@ollama list | grep -q $(OLLAMA_MODEL) || (printf "$(YELLOW)Pulling model $(OLLAMA_MODEL)...$(NC)\n" && ollama pull $(OLLAMA_MODEL))
	@printf "$(GREEN)✓ Ollama model $(OLLAMA_MODEL) ready$(NC)\n"

setup: check-prereqs setup-backend setup-ui setup-ollama ## Run all setup tasks
	@printf "$(GREEN)✓ Setup complete!$(NC)\n"
	@printf "\n"
	@printf "$(BLUE)Next steps:$(NC)\n"
	@printf "  make cluster-start # Create Kubernetes cluster\n"
	@printf "  make dev           # Start all services\n"

##@ Development

dev: setup-ollama ## Start all services (Ollama + Backend + UI)
	@printf "$(BLUE)Starting all services...$(NC)\n"
	@mkdir -p $(PID_DIR)
	@$(MAKE) start-ollama
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
	@printf "  Ollama:  http://localhost:11434\n"
	@printf "\n"
	@printf "$(BLUE)Logs:$(NC)\n"
	@printf "  make logs-backend\n"
	@printf "  make logs-ui\n"
	@printf "\n"
	@printf "$(BLUE)Stop:$(NC)\n"
	@printf "  make stop\n"

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
		cd $(BACKEND_DIR) && \
		( uv run uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000 > ../$(LOG_DIR)/backend.log 2>&1 & echo $$! > ../$(BACKEND_PID) ); \
		sleep 2; \
		printf "$(GREEN)✓ Backend started (PID: $$(cat $(BACKEND_PID)))$(NC)\n"; \
	fi

start-ui: ## Start Streamlit UI
	@printf "$(BLUE)Starting UI...$(NC)\n"
	@mkdir -p $(PID_DIR) $(LOG_DIR)
	@if [ -f $(UI_PID) ] && [ -s $(UI_PID) ] && kill -0 $$(cat $(UI_PID) 2>/dev/null) 2>/dev/null; then \
		printf "$(YELLOW)UI already running (PID: $$(cat $(UI_PID)))$(NC)\n"; \
	else \
		uv run streamlit run $(UI_DIR)/app.py --server.headless true > $(LOG_DIR)/ui.log 2>&1 & echo $$! > $(UI_PID); \
		sleep 2; \
		printf "$(GREEN)✓ UI started (PID: $$(cat $(UI_PID)))$(NC)\n"; \
	fi

stop: ## Stop all services
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
	@# Kill any remaining NeuralNav processes by pattern matching
	@pkill -f "streamlit run ui/app.py" 2>/dev/null || true
	@pkill -f "uvicorn src.api.routes:app" 2>/dev/null || true
	@# Give processes time to exit gracefully
	@sleep 1
	@# Force kill if still running
	@pkill -9 -f "streamlit run ui/app.py" 2>/dev/null || true
	@pkill -9 -f "uvicorn src.api.routes:app" 2>/dev/null || true
	@printf "$(GREEN)✓ All NeuralNav services stopped$(NC)\n"
	@# Don't stop Ollama as it might be used by other apps
	@printf "$(YELLOW)Note: Ollama left running (use 'pkill ollama' to stop manually)$(NC)\n"

restart: stop dev ## Restart all services

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

##@ Docker & Simulator

build-simulator: ## Build vLLM simulator Docker image
	@printf "$(BLUE)Building simulator image...$(NC)\n"
	cd $(SIMULATOR_DIR) && $(CONTAINER_TOOL) build -t vllm-simulator:latest -t $(SIMULATOR_FULL_IMAGE) .
	@printf "$(GREEN)✓ Simulator image built:$(NC)\n"
	@printf "  - vllm-simulator:latest\n"
	@printf "  - $(SIMULATOR_FULL_IMAGE)\n"

push-simulator: build-simulator ## Push simulator image to Quay.io
	@printf "$(BLUE)Pushing simulator image to $(SIMULATOR_FULL_IMAGE)...$(NC)\n"
	@# Check if logged in to Quay.io
	@if ! $(CONTAINER_TOOL) login quay.io --get-login > /dev/null 2>&1; then \
		printf "$(YELLOW)Not logged in to Quay.io. Please login:$(NC)\n"; \
		$(CONTAINER_TOOL) login quay.io || (printf "$(RED)✗ Login failed$(NC)\n" && exit 1); \
	else \
		printf "$(GREEN)✓ Already logged in to Quay.io$(NC)\n"; \
	fi
	@printf "$(BLUE)Pushing image...$(NC)\n"
	$(CONTAINER_TOOL) push $(SIMULATOR_FULL_IMAGE)
	@printf "$(GREEN)✓ Image pushed to $(SIMULATOR_FULL_IMAGE)$(NC)\n"

pull-simulator: ## Pull simulator image from Quay.io
	@printf "$(BLUE)Pulling simulator image from $(SIMULATOR_FULL_IMAGE)...$(NC)\n"
	$(CONTAINER_TOOL) pull $(SIMULATOR_FULL_IMAGE)
	$(CONTAINER_TOOL) tag $(SIMULATOR_FULL_IMAGE) vllm-simulator:latest
	@printf "$(GREEN)✓ Image pulled and tagged as vllm-simulator:latest$(NC)\n"

docker-build: ## Build all Docker images
	@printf "$(BLUE)Building all Docker images...$(NC)\n"
	DOCKER_BUILDKIT=1 docker-compose build --parallel
	@printf "$(GREEN)✓ All images built$(NC)\n"

docker-up: ## Start all services with Docker Compose
	@printf "$(BLUE)Starting all services with Docker Compose...$(NC)\n"
	docker-compose up -d
	@printf "$(GREEN)✓ All services started!$(NC)\n"
	@printf "\n"
	@printf "$(BLUE)Service URLs:$(NC)\n"
	@printf "  UI:        http://localhost:8501\n"
	@printf "  Backend:   http://localhost:8000\n"
	@printf "  API Docs:  http://localhost:8000/docs\n"
	@printf "  PostgreSQL: localhost:5432\n"
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

docker-logs: ## View logs from all Docker services
	@docker-compose logs -f

docker-logs-backend: ## View backend Docker logs
	@docker-compose logs -f backend

docker-logs-ui: ## View UI Docker logs
	@docker-compose logs -f ui

docker-ps: ## Show status of Docker Compose services
	@docker-compose ps

docker-restart: ## Restart Docker Compose services
	@printf "$(BLUE)Restarting Docker Compose services...$(NC)\n"
	docker-compose restart
	@printf "$(GREEN)✓ Services restarted$(NC)\n"

docker-clean: ## Remove all Docker resources
	@printf "$(BLUE)Cleaning Docker resources...$(NC)\n"
	docker-compose down -v --remove-orphans
	@printf "$(GREEN)✓ Docker resources cleaned$(NC)\n"

docker-shell-backend: ## Open shell in backend container
	@docker-compose exec backend /bin/bash

docker-shell-ui: ## Open shell in UI container
	@docker-compose exec ui /bin/bash

docker-shell-postgres: ## Open PostgreSQL shell in container
	@docker-compose exec postgres psql -U compass -d compass

##@ Kubernetes Cluster

cluster-start: check-prereqs build-simulator ## Create KIND cluster and load simulator image
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

cluster-load-simulator: build-simulator ## Load simulator image into KIND cluster
	@printf "$(BLUE)Loading simulator image into KIND cluster...$(NC)\n"
	kind load docker-image vllm-simulator:latest --name $(KIND_CLUSTER_NAME)
	@printf "$(GREEN)✓ Simulator image loaded$(NC)\n"

clean-deployments: ## Delete all InferenceServices from cluster
	@printf "$(BLUE)Deleting all InferenceServices...$(NC)\n"
	kubectl delete inferenceservices --all
	@printf "$(GREEN)✓ All deployments deleted$(NC)\n"

##@ PostgreSQL Database

db-start: ## Start PostgreSQL container for benchmark data
	@printf "$(BLUE)Starting PostgreSQL...$(NC)\n"
	@if $(CONTAINER_TOOL) ps -a --format '{{.Names}}' | grep -q '^compass-postgres$$'; then \
		if $(CONTAINER_TOOL) ps --format '{{.Names}}' | grep -q '^compass-postgres$$'; then \
			printf "$(YELLOW)PostgreSQL already running$(NC)\n"; \
		else \
			$(CONTAINER_TOOL) start compass-postgres; \
			printf "$(GREEN)✓ PostgreSQL started$(NC)\n"; \
		fi \
	else \
		$(CONTAINER_TOOL) run --name compass-postgres -d \
			-e POSTGRES_PASSWORD=compass \
			-e POSTGRES_DB=compass \
			-p 5432:5432 \
			postgres:16; \
		sleep 3; \
		printf "$(GREEN)✓ PostgreSQL started on port 5432$(NC)\n"; \
	fi
	@printf "$(BLUE)Database URL:$(NC) postgresql://postgres:compass@localhost:5432/compass\n"

db-stop: ## Stop PostgreSQL container
	@printf "$(BLUE)Stopping PostgreSQL...$(NC)\n"
	@$(CONTAINER_TOOL) stop compass-postgres 2>/dev/null || true
	@printf "$(GREEN)✓ PostgreSQL stopped$(NC)\n"

db-remove: db-stop ## Stop and remove PostgreSQL container
	@printf "$(BLUE)Removing PostgreSQL container...$(NC)\n"
	@$(CONTAINER_TOOL) rm compass-postgres 2>/dev/null || true
	@printf "$(GREEN)✓ PostgreSQL container removed$(NC)\n"

db-init: db-start ## Initialize PostgreSQL schema
	@printf "$(BLUE)Initializing PostgreSQL schema...$(NC)\n"
	@sleep 2
	@$(CONTAINER_TOOL) exec -i compass-postgres psql -U postgres -d compass < scripts/schema.sql
	@printf "$(GREEN)✓ Schema initialized$(NC)\n"

db-load-synthetic: db-start ## Load synthetic benchmark data (appends)
	@printf "$(BLUE)Loading synthetic benchmark data...$(NC)\n"
	@uv run python scripts/load_benchmarks.py data/benchmarks_synthetic.json
	@printf "$(GREEN)✓ Synthetic data loaded$(NC)\n"

db-load-blis: db-start ## Load BLIS benchmark data (appends)
	@printf "$(BLUE)Loading BLIS benchmark data...$(NC)\n"
	@uv run python scripts/load_benchmarks.py data/benchmarks_BLIS.json
	@printf "$(GREEN)✓ BLIS data loaded$(NC)\n"

db-load-estimated: db-start ## Load estimated performance benchmarks (appends)
	@printf "$(BLUE)Loading estimated performance data...$(NC)\n"
	@uv run python scripts/load_benchmarks.py data/benchmarks_estimated_performance.json
	@printf "$(GREEN)✓ Estimated data loaded$(NC)\n"

db-load-interpolated: db-start ## Load interpolated benchmark data (appends)
	@printf "$(BLUE)Loading interpolated benchmark data...$(NC)\n"
	@uv run python scripts/load_benchmarks.py data/benchmarks_interpolated_v2.json
	@printf "$(GREEN)✓ Interpolated data loaded$(NC)\n"

db-load-guidellm: db-start ## Load GuideLLM benchmark data (appends)
	@printf "$(BLUE)Loading GuideLLM benchmark data...$(NC)\n"
	@if [ ! -f data/benchmarks_GuideLLM.json ]; then \
		printf "$(RED)✗ data/benchmarks_GuideLLM.json not found$(NC)\n"; \
		printf "$(YELLOW)Run 'make db-convert-pgdump' first to create it from a pg_dump file$(NC)\n"; \
		exit 1; \
	fi
	@uv run python scripts/load_benchmarks.py data/benchmarks_GuideLLM.json
	@printf "$(GREEN)✓ GuideLLM data loaded$(NC)\n"

db-convert-pgdump: db-start ## Convert PostgreSQL dump to JSON format
	@printf "$(BLUE)Converting PostgreSQL dump to JSON...$(NC)\n"
	@if [ ! -f $(PGDUMP_INPUT) ]; then \
		printf "$(RED)✗ $(PGDUMP_INPUT) not found$(NC)\n"; \
		exit 1; \
	fi
	@uv run python scripts/convert_pgdump_to_json.py $(PGDUMP_INPUT) -o $(PGDUMP_OUTPUT)
	@printf "$(GREEN)✓ Created $(PGDUMP_OUTPUT)$(NC)\n"

db-shell: ## Open PostgreSQL shell
	@$(CONTAINER_TOOL) exec -it compass-postgres psql -U postgres -d compass

db-query-traffic: ## Query unique traffic patterns from database
	@printf "$(BLUE)Querying unique traffic patterns...$(NC)\n"
	@$(CONTAINER_TOOL) exec -i compass-postgres psql -U postgres -d compass -c \
		"SELECT DISTINCT mean_input_tokens, mean_output_tokens, COUNT(*) as num_benchmarks \
		FROM exported_summaries \
		GROUP BY prompt_tokens, output_tokens \
		ORDER BY prompt_tokens, output_tokens;"

db-query-models: ## Query available models in database
	@printf "$(BLUE)Querying available models...$(NC)\n"
	@$(CONTAINER_TOOL) exec -i compass-postgres psql -U postgres -d compass -c \
		"SELECT DISTINCT model_hf_repo, hardware, hardware_count, COUNT(*) as num_benchmarks \
		FROM exported_summaries \
		GROUP BY model_hf_repo, hardware, hardware_count \
		ORDER BY model_hf_repo, hardware, hardware_count;"

db-reset: db-remove db-init ## Reset PostgreSQL (remove and reinitialize)
	@printf "$(GREEN)✓ PostgreSQL reset complete$(NC)\n"

##@ Testing

test: test-unit ## Run all tests
	@printf "$(GREEN)✓ All tests passed$(NC)\n"

test-unit: ## Run unit tests
	@printf "$(BLUE)Running unit tests...$(NC)\n"
	cd $(BACKEND_DIR) && uv run pytest ../tests/ -v -m "not integration and not e2e"

test-integration: setup-ollama ## Run integration tests (requires Ollama)
	@printf "$(BLUE)Running integration tests...$(NC)\n"
	cd $(BACKEND_DIR) && uv run pytest ../tests/ -v -m integration

test-e2e: ## Run end-to-end tests (requires cluster)
	@printf "$(BLUE)Running end-to-end tests...$(NC)\n"
	@kubectl cluster-info > /dev/null 2>&1 || (printf "$(RED)✗ Kubernetes cluster not accessible$(NC). Run: make cluster-start\n" && exit 1)
	cd $(BACKEND_DIR) && uv run pytest ../tests/ -v -m e2e

test-workflow: setup-ollama ## Run workflow integration test
	@printf "$(BLUE)Running workflow test...$(NC)\n"
	cd $(BACKEND_DIR) && uv run $(PYTHON) test_workflow.py

test-watch: ## Run tests in watch mode
	@printf "$(BLUE)Running tests in watch mode...$(NC)\n"
	cd $(BACKEND_DIR) && uv run pytest-watch

##@ Code Quality

lint: ## Run linters
	@printf "$(BLUE)Running linters...$(NC)\n"
	@uv run ruff check $(BACKEND_DIR)/src/ $(UI_DIR)/*.py || printf "$(YELLOW)ruff not installed, skipping$(NC)\n"
	@printf "$(GREEN)✓ Linting complete$(NC)\n"

format: ## Auto-format code
	@printf "$(BLUE)Formatting code...$(NC)\n"
	@uv run ruff format $(BACKEND_DIR)/ $(UI_DIR)/ || printf "$(YELLOW)ruff not installed, skipping$(NC)\n"
	@printf "$(GREEN)✓ Formatting complete$(NC)\n"

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
	@printf "  Simulator Image: $(SIMULATOR_FULL_IMAGE)\n"
	@printf "  Ollama Model: $(OLLAMA_MODEL)\n"
	@printf "  KIND Cluster: $(KIND_CLUSTER_NAME)\n"
	@printf "\n"
	@printf "$(BLUE)Paths:$(NC)\n"
	@printf "  Backend: $(BACKEND_DIR)\n"
	@printf "  UI: $(UI_DIR)\n"
	@printf "  Simulator: $(SIMULATOR_DIR)\n"
