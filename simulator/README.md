# vLLM Simulator

A mock vLLM-compatible API server that simulates realistic LLM inference behavior without requiring GPUs.

## Purpose

The simulator allows:
- **Development without GPUs** - Test deployment workflows on any laptop
- **Predictable demos** - Canned responses ensure consistent behavior
- **Fast iteration** - No model downloads, instant startup
- **Realistic metrics** - Uses actual benchmark data for latency simulation
- **Cost savings** - No expensive GPU resources needed for development

## Features

### OpenAI-Compatible API
Implements the same endpoints as vLLM:
- `POST /v1/completions` - Text completion
- `POST /v1/chat/completions` - Chat completion
- `GET /v1/models` - List models
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

### Realistic Performance Simulation
- Loads benchmark data for specific model + GPU combinations
- Simulates TTFT (Time to First Token) based on benchmarks
- Simulates TPOT (Time Per Output Token) based on benchmarks
- Returns metrics matching vLLM's Prometheus format

### Pattern-Based Responses
Returns appropriate canned responses based on prompt patterns:
- **Code requests** - Returns code snippets
- **Summarization** - Returns summaries
- **Q&A** - Returns informative answers
- **Chat** - Returns conversational responses
- **Creative** - Returns creative text

## Building

```bash
# Build the Docker image (replace with podman as needed)
docker build -t vllm-simulator:latest .

# Test locally
docker run -p 8080:8080 \
  -e MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3" \
  -e GPU_TYPE="NVIDIA-L4" \
  -e TENSOR_PARALLEL_SIZE="1" \
  vllm-simulator:latest
```

## Usage

### Environment Variables

- `MODEL_NAME` - Model identifier (e.g., "mistralai/Mistral-7B-Instruct-v0.3")
- `GPU_TYPE` - GPU type for benchmark lookup (e.g., "NVIDIA-L4", "NVIDIA-A100-80GB")
- `TENSOR_PARALLEL_SIZE` - Tensor parallelism degree (default: 1)
- `PORT` - Port to listen on (default: 8080)

### Testing the API

```bash
# Health check
curl http://localhost:8080/health

# List models
curl http://localhost:8080/v1/models

# Text completion
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a hello world function in Python",
    "max_tokens": 100
  }'

# Chat completion
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain machine learning"}
    ],
    "max_tokens": 150
  }'

# Metrics
curl http://localhost:8080/metrics
```

## Deployment to Kubernetes

The simulator is used automatically when `simulator_mode: true` is set in the deployment generator. The InferenceService YAML will use `vllm-simulator:latest` instead of `vllm/vllm-openai`.

### Load into KIND cluster

```bash
# Build image (use podman as needed)
docker build -t vllm-simulator:latest .

# Load into KIND
kind load docker-image vllm-simulator:latest --name neuralnav
```

## Architecture

```
┌─────────────────────────────────────────────┐
│ FastAPI Application                          │
│                                              │
│ ┌────────────────────────────────────────┐ │
│ │ BenchmarkLoader                         │ │
│ │ - Loads benchmarks.json                │ │
│ │ - Finds matching model+GPU config      │ │
│ │ - Provides TTFT/TPOT values            │ │
│ └────────────────────────────────────────┘ │
│                                              │
│ ┌────────────────────────────────────────┐ │
│ │ CannedResponses                         │ │
│ │ - Pattern matching on prompts          │ │
│ │ - Returns appropriate response type    │ │
│ └────────────────────────────────────────┘ │
│                                              │
│ ┌────────────────────────────────────────┐ │
│ │ OpenAI-Compatible Endpoints             │ │
│ │ - /v1/completions                      │ │
│ │ - /v1/chat/completions                 │ │
│ │ - /v1/models                           │ │
│ └────────────────────────────────────────┘ │
│                                              │
│ ┌────────────────────────────────────────┐ │
│ │ Monitoring Endpoints                    │ │
│ │ - /health (Kubernetes probes)          │ │
│ │ - /metrics (Prometheus format)         │ │
│ └────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

## Limitations

This is a **simulator** for development and testing. It does NOT:
- Actually load or run LLM models
- Provide real AI-generated responses
- Require or use GPU resources
- Support streaming responses (yet)
- Handle complex multi-turn conversations

For production deployments, use real vLLM with actual GPUs.

## Future Enhancements

- Streaming response support
- Error injection for testing error handling
- Configurable response patterns via API
- Multi-model support in single instance
- Request/response logging for debugging
