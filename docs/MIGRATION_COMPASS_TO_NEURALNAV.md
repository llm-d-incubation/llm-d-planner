# Migration Guide: Compass to NeuralNav

This guide covers the steps required after pulling changes from the `rename-to-neuralnav` branch (or after it's merged to main).

## Overview

The project has been renamed from "Compass" to "NeuralNav". This affects:
- Docker container names (`compass-*` → `neuralnav-*`)
- PostgreSQL database name, user, and password
- Docker network name
- KIND cluster name
- Environment variables
- Container registry organization on quay.io

## Migration Steps

### 1. Rebase To The Latest Code

```bash
git checkout main
git fetch upstream
git git rebase upstream/main
git push
```

### 2. Clean Up Old Docker Resources

Stop and remove old containers with the `compass-*` naming:

```bash
# If using docker-compose
docker-compose down -v

# Remove any standalone compass-* containers
docker stop compass-postgres 2>/dev/null; docker rm compass-postgres 2>/dev/null
docker stop compass-ollama 2>/dev/null; docker rm compass-ollama 2>/dev/null
docker stop compass-backend 2>/dev/null; docker rm compass-backend 2>/dev/null
docker stop compass-ui 2>/dev/null; docker rm compass-ui 2>/dev/null
docker stop compass-simulator 2>/dev/null; docker rm compass-simulator 2>/dev/null
```

### 3. Clean Up Old KIND Cluster (if using Kubernetes)

```bash
kind delete cluster --name compass-poc
```

### 4. Rebuild Python Environment

The package name changed in `pyproject.toml`, so recreate the virtual environment:

```bash
rm -rf .venv venv
make setup
# or manually: uv venv && uv pip install -e .
```

### 5. Start Fresh with New Names

Choose the option that matches your development workflow:

**Option A: Using docker-compose (full stack)**

```bash
make docker-build
docker-compose up --build -d
```

**Option B: Using standalone database**

```bash
make db-init
make db-load-blis  # Load benchmark data
make dev            # Start all services (Ollama + Backend + UI)
```

### 6. Set Up KIND Cluster (if using Kubernetes)

```bash
make pull-simulator  # Pulls quay.io/neuralnav/vllm-simulator:latest
make cluster-start        # Create KIND cluster and load simulator image
```

### 7. Update Environment Variables

If you have any of these set in your shell profile (`.bashrc`, `.zshrc`) or `.env` files, update them:

| Old | New |
|-----|-----|
| `COMPASS_DEBUG=true` | `NEURALNAV_DEBUG=true` |

## Verification

After completing the migration, verify everything is working:

### Check Docker Containers

```bash
# Should show neuralnav-* containers
docker ps --format "{{.Names}}" | grep neuralnav
```

Expected output (varies based on your setup):
```
neuralnav-postgres
neuralnav-backend
neuralnav-ui
neuralnav-ollama
```

### Test Backend Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status":"healthy","service":"neuralnav"}
```

### Test Database Connection

```bash
make docker-shell-postgres
# or for standalone: make db-shell
```

You should connect to the `neuralnav` database.

### Check KIND Cluster (if applicable)

```bash
kind get clusters
```

Expected output:
```
neuralnav
```

## Troubleshooting

### Port 5432 Already in Use

If you see an error about port 5432 being in use, an old container may still be running:

```bash
# Find what's using the port
lsof -i :5432

# Stop any remaining compass containers
docker stop compass-postgres 2>/dev/null
docker rm compass-postgres 2>/dev/null
```

### Old Database Data

The migration creates a fresh database. If you need data from the old database:

1. Export data from old container before removing it
2. Or regenerate synthetic data: `make db-load-synthetic`

### Container Build Failures

If docker-compose build fails, try cleaning Docker cache:

```bash
docker-compose build --no-cache
```

## What Changed

For reference, here are the key naming changes:

| Component | Old Name | New Name |
|-----------|----------|----------|
| PostgreSQL container | `compass-postgres` | `neuralnav-postgres` |
| Backend container | `compass-backend` | `neuralnav-backend` |
| UI container | `compass-ui` | `neuralnav-ui` |
| Ollama container | `compass-ollama` | `neuralnav-ollama` |
| Simulator container | `compass-simulator` | `neuralnav-simulator` |
| Docker network | `compass-network` | `neuralnav-network` |
| Database name | `compass` | `neuralnav` |
| Database user | `compass` | `neuralnav` |
| KIND cluster | `compass-poc` | `neuralnav` |
| Debug env var | `COMPASS_DEBUG` | `NEURALNAV_DEBUG` |
| Quay.io org | `vllm-assistant` | `neuralnav` |
| KServe annotation | `compass/simulator-mode` | `neuralnav/simulator-mode` |
