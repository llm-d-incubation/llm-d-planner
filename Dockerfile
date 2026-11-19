# Backend Dockerfile for Planner
FROM python:3.15-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files and README (hatchling requires README.md for package metadata)
COPY pyproject.toml uv.lock README.md ./

# Provide version for hatch-vcs (backed by setuptools-scm) since .git is excluded
# by .dockerignore. ARG is available to RUN instructions; no ENV needed at runtime.
ARG SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

# Create src/planner/ so uv sync can invoke the hatch-vcs hook to write _version.py
RUN mkdir -p src/planner

# Install Python dependencies (frozen = use lockfile exactly, no-dev = skip dev deps)
RUN uv sync --frozen --no-dev --extra server --extra llm --extra openai --extra vertex --extra kubernetes --extra estimation --extra quality-sync
RUN uv pip install "llm-optimizer @ git+https://github.com/bentoml/llm-optimizer.git"

# Copy backend source code and data files (Knowledge Base is now in src/planner/data/)
COPY src/planner ./src/planner
COPY src/quality_scoring ./src/quality_scoring

# Copy scripts (schema init, benchmark loading)
COPY scripts ./scripts

# Create non-root user and directories for generated files
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid 0 --no-create-home appuser && \
    mkdir -p /app/generated_configs /app/logs/prompts /app/.cache /app/db && \
    chown -R appuser:0 /app && \
    chmod -R g=u /app/generated_configs /app/logs /app/.cache /app/db

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache
# Use the venv created by uv sync (avoids uv run writing to .venv at runtime)
ENV PATH="/app/.venv/bin:$PATH"

ARG MODEL_CATALOG_URL

# Switch to non-root user
USER appuser

# Expose backend API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()" || exit 1

# Run the backend API server
CMD ["uvicorn", "planner.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
