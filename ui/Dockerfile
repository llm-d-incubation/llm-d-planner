# UI Dockerfile for Planner Streamlit Frontend
FROM python:3.15-slim

# Set working directory
WORKDIR /app

# Copy minimal UI requirements
COPY ui/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy UI source code
COPY ui ./ui

# Copy planner data module (used by slo.py and Capacity_Planner for config file paths).
# Create a minimal planner package — the full __init__.py would pull in all deps.
RUN mkdir -p src/planner/data/configuration && touch src/planner/__init__.py
COPY src/planner/data/__init__.py ./src/planner/data/__init__.py
COPY src/planner/data/_resolver.py ./src/planner/data/_resolver.py
COPY src/planner/data/configuration/ ./src/planner/data/configuration/

# Copy Streamlit configuration
COPY .streamlit ./.streamlit

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose Streamlit port
EXPOSE 8501

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health')" || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "ui/main.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
