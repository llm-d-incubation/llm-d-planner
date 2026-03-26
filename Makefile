# Compass - Makefile
#
# This Makefile provides common development tasks for Compass.
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

# Configuration
REGISTRY ?= quay.io
REGISTRY_ORG ?= vllm-assistant
SIMULATOR_IMAGE ?= vllm-simulator
SIMULATOR_TAG ?= latest
SIMULATOR_FULL_IMAGE := $(REGISTRY)/$(REGISTRY_ORG)/$(SIMULATOR_IMAGE):$(SIMULATOR_TAG)

OLLAMA_MODEL ?= qwen2.5:7b
KIND_CLUSTER_NAME ?= compass-poc

BACKEND_DIR := backend
UI_DIR := ui
SIMULATOR_DIR := simulator

VENV := venv
# Shared venv at project root for both backend and UI

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
	@command -v docker >/dev/null 2>&1 || (ecprintfho "$(RED)✗ docker not found$(NC). Install from https://www.docker.com/products/docker-desktop\n" && exit 1)
	@printf "$(GREEN)✓ docker found$(NC)\n"
	@command -v kubectl >/dev/null 2>&1 || (printf "$(RED)✗ kubectl not found$(NC). Run: brew install kubectl\n" && exit 1)
	@printf "$(GREEN)✓ kubectl found$(NC)\n"
	@command -v kind >/dev/null 2>&1 || (printf "$(RED)✗ kind not found$(NC). Run: brew install kind\n" && exit 1)
	@printf "$(GREEN)✓ kind found$(NC)\n"
	@command -v ollama >/dev/null 2>&1 || (printf "$(RED)✗ ollama not found$(NC). Run: brew install ollama\n" && exit 1)
	@printf "$(GREEN)✓ ollama found$(NC)\n"
	@command -v $(PYTHON) >/dev/null 2>&1 || (printf "$(RED)✗ $(PYTHON) not found$(NC). Run: brew install python@3.13\n" && exit 1)
	@printf "$(GREEN)✓ $(PYTHON) found$(NC)\n"
	@# Check Python version and warn if 3.14+
	@PY_VERSION=$$($(PYTHON) -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0"); \
	if [ "$$(echo "$$PY_VERSION" | cut -d. -f1)" = "3" ] && [ "$$(echo "$$PY_VERSION" | cut -d. -f2)" -ge "14" ]; then \
		printf "$(YELLOW)⚠ Warning: Python $$PY_VERSION detected. Some dependencies (psycopg2-binary) may not have wheels yet.$(NC)\n"; \
		printf "$(YELLOW)  Recommend using Python 3.13 for best compatibility.$(NC)\n"; \
	fi
	@docker info >/dev/null 2>&1 || (printf "$(RED)✗ Docker daemon not running$(NC). Start Docker Desktop\n" && exit 1)
	@printf "$(GREEN)✓ Docker daemon running$(NC)\n"
	@printf "$(GREEN)All prerequisites satisfied!$(NC)\n"

setup-backend: ## Set up Python environment (includes backend and UI dependencies)
	@printf "$(BLUE)Setting up Python environment...$(NC)\n"
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip
	. $(VENV)/bin/activate && pip install -r requirements.txt
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
		. $(VENV)/bin/activate && cd $(BACKEND_DIR) && \
		( uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000 > ../$(LOG_DIR)/backend.log 2>&1 & echo $$! > ../$(BACKEND_PID) ); \
		sleep 2; \
		printf "$(GREEN)✓ Backend started (PID: $$(cat $(BACKEND_PID)))$(NC)\n"; \
	fi

start-ui: ## Start Streamlit UI
	@printf "$(BLUE)Starting UI...$(NC)\n"
	@mkdir -p $(PID_DIR) $(LOG_DIR)
	@if [ -f $(UI_PID) ] && [ -s $(UI_PID) ] && kill -0 $$(cat $(UI_PID) 2>/dev/null) 2>/dev/null; then \
		printf "$(YELLOW)UI already running (PID: $$(cat $(UI_PID)))$(NC)\n"; \
	else \
		. $(VENV)/bin/activate && streamlit run $(UI_DIR)/app.py --server.headless true > $(LOG_DIR)/ui.log 2>&1 & echo $$! > $(UI_PID); \
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
	@# Kill any remaining Compass processes by pattern matching
	@pkill -f "streamlit run ui/app.py" 2>/dev/null || true
	@pkill -f "uvicorn src.api.routes:app" 2>/dev/null || true
	@# Give processes time to exit gracefully
	@sleep 1
	@# Force kill if still running
	@pkill -9 -f "streamlit run ui/app.py" 2>/dev/null || true
	@pkill -9 -f "uvicorn src.api.routes:app" 2>/dev/null || true
	@printf "$(GREEN)✓ All Compass services stopped$(NC)\n"
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
	cd $(SIMULATOR_DIR) && docker build -t vllm-simulator:latest -t $(SIMULATOR_FULL_IMAGE) .
	@printf "$(GREEN)✓ Simulator image built:$(NC)\n"
	@printf "  - vllm-simulator:latest\n"
	@printf "  - $(SIMULATOR_FULL_IMAGE)\n"

push-simulator: build-simulator ## Push simulator image to Quay.io
	@printf "$(BLUE)Pushing simulator image to $(SIMULATOR_FULL_IMAGE)...$(NC)\n"
	@# Check if logged in to Quay.io
	@if ! docker login quay.io --get-login > /dev/null 2>&1; then \
		printf "$(YELLOW)Not logged in to Quay.io. Please login:$(NC)\n"; \
		docker login quay.io || (printf "$(RED)✗ Login failed$(NC)\n" && exit 1); \
	else \
		printf "$(GREEN)✓ Already logged in to Quay.io$(NC)\n"; \
	fi
	@printf "$(BLUE)Pushing image...$(NC)\n"
	docker push $(SIMULATOR_FULL_IMAGE)
	@printf "$(GREEN)✓ Image pushed to $(SIMULATOR_FULL_IMAGE)$(NC)\n"

pull-simulator: ## Pull simulator image from Quay.io
	@printf "$(BLUE)Pulling simulator image from $(SIMULATOR_FULL_IMAGE)...$(NC)\n"
	docker pull $(SIMULATOR_FULL_IMAGE)
	docker tag $(SIMULATOR_FULL_IMAGE) vllm-simulator:latest
	@printf "$(GREEN)✓ Image pulled and tagged as vllm-simulator:latest$(NC)\n"

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

postgres-start: ## Start PostgreSQL container for benchmark data
	@printf "$(BLUE)Starting PostgreSQL...$(NC)\n"
	@if docker ps -a --format '{{.Names}}' | grep -q '^compass-postgres$$'; then \
		if docker ps --format '{{.Names}}' | grep -q '^compass-postgres$$'; then \
			printf "$(YELLOW)PostgreSQL already running$(NC)\n"; \
		else \
			docker start compass-postgres; \
			printf "$(GREEN)✓ PostgreSQL started$(NC)\n"; \
		fi \
	else \
		docker run --name compass-postgres -d \
			-e POSTGRES_PASSWORD=compass \
			-e POSTGRES_DB=compass \
			-p 5432:5432 \
			postgres:16; \
		sleep 3; \
		printf "$(GREEN)✓ PostgreSQL started on port 5432$(NC)\n"; \
	fi
	@printf "$(BLUE)Database URL:$(NC) postgresql://postgres:compass@localhost:5432/compass\n"

postgres-stop: ## Stop PostgreSQL container
	@printf "$(BLUE)Stopping PostgreSQL...$(NC)\n"
	@docker stop compass-postgres 2>/dev/null || true
	@printf "$(GREEN)✓ PostgreSQL stopped$(NC)\n"

postgres-remove: postgres-stop ## Stop and remove PostgreSQL container
	@printf "$(BLUE)Removing PostgreSQL container...$(NC)\n"
	@docker rm compass-postgres 2>/dev/null || true
	@printf "$(GREEN)✓ PostgreSQL container removed$(NC)\n"

postgres-init: postgres-start ## Initialize PostgreSQL schema
	@printf "$(BLUE)Initializing PostgreSQL schema...$(NC)\n"
	@sleep 2
	@docker exec -i compass-postgres psql -U postgres -d compass < scripts/schema.sql
	@printf "$(GREEN)✓ Schema initialized$(NC)\n"

postgres-load-synthetic: postgres-init ## Load synthetic benchmark data from JSON
	@printf "$(BLUE)Loading synthetic benchmark data...$(NC)\n"
	@. $(VENV)/bin/activate && $(PYTHON) scripts/load_benchmarks.py data/benchmarks_synthetic.json
	@printf "$(GREEN)✓ Synthetic data loaded$(NC)\n"

postgres-load-blis: postgres-init ## Load BLIS benchmark data from JSON
	@printf "$(BLUE)Loading BLIS benchmark data...$(NC)\n"
	@. $(VENV)/bin/activate && $(PYTHON) scripts/load_benchmarks.py data/benchmarks_BLIS.json
	@printf "$(GREEN)✓ BLIS data loaded$(NC)\n"

postgres-load-real: postgres-init ## Load real benchmark data from SQL dump
	@printf "$(BLUE)Loading real benchmark data from integ-oct-29.sql...$(NC)\n"
	@if [ ! -f data/integ-oct-29.sql ]; then \
		printf "$(RED)✗ data/integ-oct-29.sql not found$(NC)\n"; \
		printf "$(YELLOW)This file is not in version control due to NDA restrictions$(NC)\n"; \
		exit 1; \
	fi
	@# Copy dump file into container temporarily
	@docker cp data/integ-oct-29.sql compass-postgres:/tmp/integ-oct-29.sql
	@# Restore data only (schema already created by postgres-init)
	@docker exec compass-postgres pg_restore -U postgres -d compass --data-only /tmp/integ-oct-29.sql 2>&1 | grep -v "ERROR.*cloudsqlsuperuser" || true
	@# Clean up
	@docker exec compass-postgres rm /tmp/integ-oct-29.sql
	@# Show statistics
	@printf "\n"
	@printf "$(BLUE)📊 Database Statistics:$(NC)\n"
	@docker exec -i compass-postgres psql -U postgres -d compass -c \
		"SELECT COUNT(*) as total_benchmarks FROM exported_summaries;" | grep -v "^-" | grep -v "row"
	@docker exec -i compass-postgres psql -U postgres -d compass -c \
		"SELECT COUNT(DISTINCT model_hf_repo) as num_models FROM exported_summaries;" | grep -v "^-" | grep -v "row"
	@docker exec -i compass-postgres psql -U postgres -d compass -c \
		"SELECT COUNT(DISTINCT hardware) as num_hardware_types FROM exported_summaries;" | grep -v "^-" | grep -v "row"
	@docker exec -i compass-postgres psql -U postgres -d compass -c \
		"SELECT COUNT(DISTINCT (prompt_tokens, output_tokens)) as num_traffic_profiles FROM exported_summaries WHERE prompt_tokens IS NOT NULL;" | grep -v "^-" | grep -v "row"
	@printf "$(GREEN)✓ Real benchmark data loaded$(NC)\n"

postgres-shell: ## Open PostgreSQL shell
	@docker exec -it compass-postgres psql -U postgres -d compass

postgres-query-traffic: ## Query unique traffic patterns from database
	@printf "$(BLUE)Querying unique traffic patterns...$(NC)\n"
	@docker exec -i compass-postgres psql -U postgres -d compass -c \
		"SELECT DISTINCT prompt_tokens, output_tokens, COUNT(*) as num_benchmarks \
		FROM exported_summaries \
		GROUP BY prompt_tokens, output_tokens \
		ORDER BY prompt_tokens, output_tokens;"

postgres-query-models: ## Query available models in database
	@printf "$(BLUE)Querying available models...$(NC)\n"
	@docker exec -i compass-postgres psql -U postgres -d compass -c \
		"SELECT DISTINCT model_hf_repo, hardware, hardware_count, COUNT(*) as num_benchmarks \
		FROM exported_summaries \
		GROUP BY model_hf_repo, hardware, hardware_count \
		ORDER BY model_hf_repo, hardware, hardware_count;"

postgres-reset: postgres-remove postgres-init ## Reset PostgreSQL (remove and reinitialize)
	@printf "$(GREEN)✓ PostgreSQL reset complete$(NC)\n"

##@ Testing

test: test-unit ## Run all tests
	@printf "$(GREEN)✓ All tests passed$(NC)\n"

test-unit: ## Run unit tests
	@printf "$(BLUE)Running unit tests...$(NC)\n"
	. $(VENV)/bin/activate && cd $(BACKEND_DIR) && pytest ../tests/ -v -m "not integration and not e2e"

test-integration: setup-ollama ## Run integration tests (requires Ollama)
	@printf "$(BLUE)Running integration tests...$(NC)\n"
	. $(VENV)/bin/activate && cd $(BACKEND_DIR) && pytest ../tests/ -v -m integration

test-e2e: ## Run end-to-end tests (requires cluster)
	@printf "$(BLUE)Running end-to-end tests...$(NC)\n"
	@kubectl cluster-info > /dev/null 2>&1 || (printf "$(RED)✗ Kubernetes cluster not accessible$(NC). Run: make cluster-start\n" && exit 1)
	. $(VENV)/bin/activate && cd $(BACKEND_DIR) && pytest ../tests/ -v -m e2e

test-workflow: setup-ollama ## Run workflow integration test
	@printf "$(BLUE)Running workflow test...$(NC)\n"
	. $(VENV)/bin/activate && cd $(BACKEND_DIR) && $(PYTHON) test_workflow.py

test-watch: ## Run tests in watch mode
	@printf "$(BLUE)Running tests in watch mode...$(NC)\n"
	. $(VENV)/bin/activate && cd $(BACKEND_DIR) && pytest-watch

##@ Code Quality

lint: ## Run linters
	@printf "$(BLUE)Running linters...$(NC)\n"
	@if [ -d $(VENV) ]; then \
		. $(VENV)/bin/activate && \
		(command -v ruff >/dev/null 2>&1 && ruff check $(BACKEND_DIR)/src/ $(UI_DIR)/*.py || printf "$(YELLOW)ruff not installed, skipping$(NC)\n"); \
	fi
	@printf "$(GREEN)✓ Linting complete$(NC)\n"

format: ## Auto-format code
	@printf "$(BLUE)Formatting code...$(NC)\n"
	@if [ -d $(VENV) ]; then \
		. $(VENV)/bin/activate && \
		(command -v ruff >/dev/null 2>&1 && ruff format $(BACKEND_DIR)/ $(UI_DIR)/ || printf "$(YELLOW)ruff not installed, skipping$(NC)\n"); \
	fi
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
	@printf "$(BLUE)Cleaning virtual environments...$(NC)\n"
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
