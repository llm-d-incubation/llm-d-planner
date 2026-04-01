# Configuration Explorer

The configuration explorer provides capacity planning and GPU recommendation tools for LLM inference. It is integrated into the NeuralNav UI as dedicated pages, and also available as a CLI.

Features include:

- **Capacity planning**:
  - Get per-GPU memory requirements to load and serve a model, and compare parallelism strategies.
  - Determine KV cache memory requirements based on workload characteristics.
  - Estimate peak activation memory, CUDA graph overhead, and non-torch memory for accurate capacity planning (see empirical results [here](./empirical-vllm-memory-results.md))
- **GPU recommendation**:
  - Recommend GPU configurations using BentoML's llm-optimizer roofline algorithm.
  - Analyze throughput, latency (TTFT, ITL, E2E), and concurrency trade-offs across different GPU types.
  - Export recommendations in JSON format for integration with other tools.

## CLI

After installing NeuralNav (`uv sync --extra dev`), the `planner` command is available:

```bash
# Run capacity planning
planner plan --model Qwen/Qwen2.5-3B --gpu-memory 80 --max-model-len 16000

# Run GPU recommendation and performance estimation (BentoML's roofline model)
planner estimate --model Qwen/Qwen2.5-3B --input-len 512 --output-len 128 --max-gpus 8

# Human-readable output
planner estimate --model Qwen/Qwen2.5-3B --input-len 512 --output-len 128 --pretty

# Override GPU costs with custom pricing
planner estimate --model Qwen/Qwen2.5-3B \
  --input-len 512 --output-len 128 \
  --custom-gpu-cost H100:30.50 \
  --custom-gpu-cost A100:22 \
  --custom-gpu-cost L40:25.00 \
  --pretty

# Get help
planner --help
```

**Note**: You'll need a HuggingFace token set as the `HF_TOKEN` environment variable to access gated models.

## Web Application

The configuration explorer is available as two pages in the NeuralNav Streamlit UI (`make start`):

### Pages

1. **Capacity Planner** (`ui/pages/1_Capacity_Planner.py`) - Analyze GPU memory requirements and capacity planning for LLM models
2. **GPU Recommender** (`ui/pages/2_GPU_Recommender.py`) - Get optimal GPU recommendations based on model and workload requirements

### Using the Capacity Planner

The Capacity Planner lets you explore memory requirements and parallelism strategies for a given model:

1. **Enter a model**: Provide a HuggingFace model ID
2. **Configure hardware**: Select GPU type and memory
3. **Set workload**: Specify context length, batch size, and parallelism (TP/PP/DP)
4. **Review breakdown**: Inspect model weight memory, KV cache, activation, and overhead components

### Using the GPU Recommender

The GPU Recommender helps you find the optimal GPU for running LLM inference:

1. **Configure Model**: Enter a HuggingFace model ID (e.g., `meta-llama/Llama-2-7b-hf`)
2. **Set Workload Parameters**:
   - Input sequence length (tokens)
   - Output sequence length (tokens)
   - Maximum number of GPUs
3. **Define Constraints (Optional)**:
   - Maximum Time to First Token (TTFT) in milliseconds
   - Maximum Inter-Token Latency (ITL) in milliseconds
   - Maximum End-to-End Latency in seconds
4. **Run Analysis**: Click the "Run Analysis" button to evaluate all available GPUs
5. **Review Results**:
   - Compare GPUs through interactive visualizations
   - Examine throughput, latency metrics, and optimal concurrency
   - View detailed analysis for each GPU
6. **Export**: Download results as JSON or CSV for further analysis

The GPU Recommender uses BentoML's llm-optimizer roofline algorithm to provide synthetic performance estimates across different GPU types.

### Cost Information

The GPU Recommender displays cost information to help you find cost-effective GPU configurations:

- **Default GPU Costs**: Built-in reference costs for common GPUs (H200, H100, A100, L40, etc.)
- **Custom Cost Override**: Specify your own GPU costs using any numbers you prefer (e.g., your actual $/hour or $/token pricing)
- **Cost-Based Sorting**: Sort results by cost to find the most economical option

**⚠️ IMPORTANT**: Default costs are **reference values for relative comparison only**. They do **NOT** represent actual pricing from any provider. Use custom costs that reflect your actual infrastructure pricing.
