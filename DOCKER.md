# Docker Deployment Guide for Compass

This guide explains how to run Compass using Docker and Docker Compose.

## Overview

Compass is containerized into the following services:

- **postgres**: PostgreSQL database for benchmarks and deployment outcomes
- **ollama**: Ollama LLM service for intent extraction (qwen2.5:7b model)
- **backend**: FastAPI REST API server
- **ui**: Streamlit web interface
- **simulator**: vLLM simulator for GPU-free development (optional)

## Prerequisites

- Docker Engine 20.10+ or Docker Desktop
- Docker Compose v2.0+
- At least 8GB of RAM available for Docker
- 10GB of free disk space (for Ollama models and PostgreSQL data)

## Quick Start

**Note:** This guide shows raw `docker-compose` commands. For convenience, you can use `make` commands instead (see [Makefile Commands](#makefile-commands-convenience-shortcuts) section below). Run `make help` to see all available shortcuts.

### Production Mode

Run the full stack (backend, UI, PostgreSQL, Ollama):

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service health
docker-compose ps
```

Access the application:
- **UI**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Makefile Commands (Convenience Shortcuts)

This repository includes a **Makefile** that provides convenient shortcuts for all Docker operations. You can use either the raw `docker-compose` commands shown throughout this guide OR the equivalent `make` commands.

**Why use Makefile commands?**
- Shorter, easier to remember commands
- Colored output for better visibility
- Built-in health checks and status messages
- Consistent cross-platform behavior (macOS/Linux)

### Quick Command Reference

| Task | Docker Compose Command | Makefile Shortcut |
|------|------------------------|-------------------|
| **Build images** | `docker-compose build` | `make docker-build` |
| **Start services** | `docker-compose up -d` | `make docker-up` |
| **Start dev mode** | `docker-compose -f docker-compose.yml -f docker-compose.dev.yml up` | `make docker-up-dev` |
| **Stop services** | `docker-compose down` | `make docker-down` |
| **Stop + remove volumes** | `docker-compose down -v` | `make docker-down-v` |
| **View logs (all)** | `docker-compose logs -f` | `make docker-logs` |
| **View backend logs** | `docker-compose logs -f backend` | `make docker-logs-backend` |
| **View UI logs** | `docker-compose logs -f ui` | `make docker-logs-ui` |
| **Check status** | `docker-compose ps` | `make docker-ps` |
| **Restart services** | `docker-compose restart` | `make docker-restart` |
| **Clean everything** | `docker-compose down -v --remove-orphans` | `make docker-clean` |
| **Backend shell** | `docker-compose exec backend /bin/bash` | `make docker-shell-backend` |
| **UI shell** | `docker-compose exec ui /bin/bash` | `make docker-shell-ui` |
| **PostgreSQL shell** | `docker-compose exec postgres psql -U compass -d compass` | `make docker-shell-postgres` |

### Examples

**Using docker-compose:**
```bash
docker-compose build
docker-compose up -d
docker-compose logs -f backend
docker-compose down
```

**Using Makefile (equivalent):**
```bash
make docker-build
make docker-up
make docker-logs-backend
make docker-down
```

**Complete list of available commands:**
```bash
make help  # Show all available Makefile targets
```

### Development Mode

Run with hot reload enabled for code changes:

```bash
# Using docker-compose
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Using Makefile (recommended)
make docker-up-dev
```

### With Simulator

Run the vLLM simulator for GPU-free testing:

```bash
# Start with simulator profile
docker-compose --profile simulator up -d

# Simulator will be available at http://localhost:8080
```

## Common Operations

All operations below show both `docker-compose` commands and their `make` equivalents. Use whichever you prefer!

### Building Images

```bash
# Build all images
docker-compose build
make docker-build

# Build specific service (docker-compose only)
docker-compose build backend

# Force rebuild without cache (docker-compose only)
docker-compose build --no-cache
```

### Managing Services

```bash
# Start services
docker-compose up -d
make docker-up

# Stop services
docker-compose down
make docker-down

# Stop and remove volumes (WARNING: deletes database data)
docker-compose down -v
make docker-down-v

# Restart services
docker-compose restart
make docker-restart

# Restart specific service (docker-compose only)
docker-compose restart backend

# View logs (all services)
docker-compose logs -f
make docker-logs

# View backend logs
docker-compose logs -f backend
make docker-logs-backend

# View UI logs
docker-compose logs -f ui
make docker-logs-ui

# Check service status
docker-compose ps
make docker-ps
```

### Database Operations

```bash
# Access PostgreSQL CLI
docker-compose exec postgres psql -U compass -d compass
make docker-shell-postgres

# Run database migrations/scripts
docker-compose exec postgres psql -U compass -d compass -f /scripts/schema.sql

# Load benchmark data
docker-compose exec backend python scripts/load_benchmarks.py

# Backup database
docker-compose exec postgres pg_dump -U compass compass > backup.sql

# Restore database
docker-compose exec -T postgres psql -U compass compass < backup.sql
```

### Ollama Operations

```bash
# Pull a different model
docker-compose exec ollama ollama pull llama3.2

# List installed models
docker-compose exec ollama ollama list

# Run model interactively
docker-compose exec ollama ollama run qwen2.5:7b
```

## Configuration

### Environment Variables

Create a `.env` file in the project root to customize configuration:

```env
# Database Configuration
POSTGRES_DB=compass
POSTGRES_USER=compass
POSTGRES_PASSWORD=your_secure_password

# Ollama Configuration
OLLAMA_MODEL=qwen2.5:7b

# Backend Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# UI Configuration
STREAMLIT_SERVER_PORT=8501
```

### Port Mapping

Default port mappings (can be customized in docker-compose.yml):

| Service   | Container Port | Host Port |
|-----------|----------------|-----------|
| UI        | 8501           | 8501      |
| Backend   | 8000           | 8000      |
| PostgreSQL| 5432           | 5432      |
| Ollama    | 11434          | 11434     |
| Simulator | 8080           | 8080      |

To change host ports, edit `docker-compose.yml`:

```yaml
services:
  ui:
    ports:
      - "8080:8501"  # Access UI on http://localhost:8080
```

## Development Workflow

### Hot Reload

Both backend and UI support hot reload in development mode:

1. **Backend**: Uvicorn's `--reload` flag watches for Python file changes
2. **UI**: Streamlit automatically reloads when files change

Start in development mode:

```bash
# Using docker-compose
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Using Makefile (recommended)
make docker-up-dev
```

Edit files in `backend/src/` or `ui/` - changes will be reflected immediately.

### Debugging

#### View Container Logs

```bash
# Follow logs for all services
docker-compose logs -f
make docker-logs

# View specific service logs
docker-compose logs -f backend
make docker-logs-backend

docker-compose logs -f ui
make docker-logs-ui
```

#### Access Container Shell

```bash
# Backend container
docker-compose exec backend /bin/bash
make docker-shell-backend

# UI container
docker-compose exec ui /bin/bash
make docker-shell-ui

# PostgreSQL container
docker-compose exec postgres /bin/bash
make docker-shell-postgres
```

#### Check Service Health

```bash
# View service status
docker-compose ps
make docker-ps

# Test health endpoints
curl http://localhost:8000/health  # Backend
curl http://localhost:8501/_stcore/health  # UI
curl http://localhost:11434/  # Ollama
```

## Architecture

### Network Topology

All services communicate via a bridge network (`compass-network`):

```
┌─────────────────────────────────────────────┐
│           compass-network (bridge)          │
│                                             │
│  ┌─────────┐    ┌─────────┐    ┌────────┐   │
│  │   UI    │───▶│ Backend │───▶│Postgres│   │
│  │ :8501   │    │  :8000  │    │ :5432  │   │
│  └─────────┘    └────┬────┘    └────────┘   │
│                      │                      │
│                      ▼                      │
│                 ┌─────────┐                 │
│                 │ Ollama  │                 │
│                 │ :11434  │                 │
│                 └─────────┘                 │
└─────────────────────────────────────────────┘
         │         │         │         │
         ▼         ▼         ▼         ▼
    localhost  localhost localhost localhost
      :8501     :8000     :5432    :11434
```

### Volume Mounts

**Development Volumes** (source code - hot reload):
- `./backend/src` → `/app/backend/src`
- `./ui` → `/app/ui`
- `./data` → `/app/data`

**Persistent Volumes** (data):
- `postgres_data`: PostgreSQL database files
- `ollama_data`: Ollama models and cache

## Troubleshooting

### Services Won't Start

**Check Docker resources:**
```bash
docker system df
docker system prune  # Free up space
```

**Check service logs:**
```bash
docker-compose logs backend
make docker-logs-backend

docker-compose logs postgres  # Use docker-compose for PostgreSQL
```

**Verify dependencies:**
```bash
# Ensure Ollama model is downloaded
docker-compose exec ollama ollama list

# Verify database schema
docker-compose exec postgres psql -U compass -c "\dt"
```

### Database Connection Errors

**Ensure PostgreSQL is healthy:**
```bash
docker-compose ps postgres
make docker-ps  # Show all services

docker-compose logs postgres
```

**Test database connection:**
```bash
docker-compose exec postgres pg_isready -U compass
```

**Reset database (WARNING: deletes all data):**
```bash
# Using docker-compose
docker-compose down -v
docker-compose up -d postgres
# Wait for PostgreSQL to initialize
docker-compose up -d

# Using Makefile (easier)
make docker-down-v
make docker-up
```

### Ollama Model Not Loading

**Check if model is downloaded:**
```bash
docker-compose exec ollama ollama list
```

**Manually pull model:**
```bash
docker-compose exec ollama ollama pull qwen2.5:7b
```

**Check Ollama logs:**
```bash
docker-compose logs ollama
```

### Port Conflicts

**Error**: "port is already allocated"

**Solution**: Change host port in `docker-compose.yml`:
```yaml
services:
  ui:
    ports:
      - "8502:8501"  # Use different host port
```

### Slow Performance

**Increase Docker resources:**
- Docker Desktop → Preferences → Resources
- Allocate more CPU cores and RAM (recommended: 4 CPUs, 8GB RAM)

**Check resource usage:**
```bash
docker stats
```

## Production Deployment

### Security Checklist

Before deploying to production:

- [ ] Change default PostgreSQL password
- [ ] Use secrets management (Docker Swarm secrets or Kubernetes secrets)
- [ ] Enable TLS/HTTPS with reverse proxy (nginx, traefik)
- [ ] Configure firewall rules (restrict PostgreSQL port 5432)
- [ ] Set `DEBUG=false` in environment variables
- [ ] Review CORS settings in backend
- [ ] Enable authentication/authorization

### Using Reverse Proxy

Example nginx configuration:

```nginx
server {
    listen 80;
    server_name compass.example.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
    }
}
```

### Resource Requirements

**Minimum (Development)**:
- 2 CPU cores
- 8GB RAM
- 10GB disk space

**Recommended (Production)**:
- 4+ CPU cores
- 16GB RAM
- 50GB disk space (for Ollama models + PostgreSQL data)

## Next Steps

- See [README.md](README.md) for overall project documentation
- See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
- See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system architecture
- See [backend/TESTING.md](backend/TESTING.md) for testing procedures

## Support

For issues and questions:
- Check [GitHub Issues](https://github.com/yourusername/compass/issues)
- Review Docker logs: `docker-compose logs -f`
- Verify service health: `docker-compose ps`
