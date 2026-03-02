# Backend Dockerfile for NeuralNav
FROM --platform=linux/amd64 python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files and README (hatchling requires README.md for package metadata)
COPY pyproject.toml uv.lock README.md ./

# Install Python dependencies (frozen = use lockfile exactly, no-dev = skip dev deps)
RUN uv sync --frozen --no-dev

# Copy backend source code
COPY src/neuralnav ./src/neuralnav

# Copy data files (Knowledge Base)
COPY data ./data

# Copy scripts (schema init, benchmark loading — used by db-init Job)
COPY scripts ./scripts

# Create directories for generated files
RUN mkdir -p /app/generated_configs /app/logs/prompts && \
    chmod -R 770 /app/generated_configs /app/logs

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
# Use the venv created by uv sync (avoids uv run writing to .venv at runtime)
ENV PATH="/app/.venv/bin:$PATH"

ARG MODEL_CATALOG_URL

# Expose backend API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()" || exit 1

# Run the backend API server
CMD ["uvicorn", "neuralnav.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
