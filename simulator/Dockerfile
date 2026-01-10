FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY simulator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy simulator service
COPY simulator/simulator_service.py .

# Create data directory for benchmarks
RUN mkdir -p /app/data

# Copy benchmark data from main data directory
COPY data/benchmarks_BLIS.json /app/data/benchmarks.json

# Expose vLLM-compatible port
EXPOSE 8080

# Set default environment variables
ENV MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
ENV TENSOR_PARALLEL_SIZE="1"
ENV GPU_TYPE="H100"
ENV PORT="8080"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Run the simulator
CMD ["python", "simulator_service.py"]
