"""Pure utility functions for the Planner UI.

No Streamlit dependency — safe to import anywhere.
"""

# Provider mapping for model name normalization
PROVIDER_MAPPING = {
    "gpt-oss": "OpenAI",
    "gptoss": "OpenAI",
    "kimi": "Moonshot",
    "deepseek": "DeepSeek",
    "qwen": "Qwen",
    "llama": "Meta",
    "mistral": "Mistral",
    "gemma": "Google",
    "minimax": "MiniMax",
    "phi": "Microsoft",
    "falcon": "TII",
    "yi": "01.AI",
    "internlm": "Shanghai AI Lab",
    "baichuan": "Baichuan",
    "chatglm": "Zhipu",
    "glm": "Zhipu",
    "starcoder": "BigCode",
    "codellama": "Meta",
    "aya": "Cohere",
}


def normalize_model_name(raw_name: str) -> str:
    """Normalize model name to consistent Provider/Model-Name format.

    Examples:
        "GPT-OSS 120B" → "OpenAI/GPT-OSS-120B"
        "gpt-oss-120b" → "OpenAI/GPT-OSS-120B"
        "Moonshot/Kimi-K2-Thinking" → "Moonshot/Kimi-K2-Thinking"
        "kimi-k2-thinking" → "Moonshot/Kimi-K2-Thinking"
        "DeepSeek/DeepSeek-V3.1-Reasoning" → "DeepSeek/DeepSeek-V3.1-Reasoning"
    """
    if not raw_name:
        return "Unknown"

    name = raw_name.strip()

    # Already has provider prefix (e.g., "Moonshot/Kimi-K2")
    if "/" in name:
        parts = name.split("/", 1)
        provider = parts[0].strip()
        model = parts[1].strip()
        # Normalize model part: title case with hyphens
        model_normalized = "-".join(
            word.title() if not word.isupper() else word
            for word in model.replace("_", "-").split("-")
        )
        return f"{provider}/{model_normalized}"

    # No provider - need to detect and add it
    name_lower = name.lower().replace(" ", "-").replace("_", "-")

    # Find matching provider
    detected_provider = None
    for keyword, provider in PROVIDER_MAPPING.items():
        if keyword in name_lower:
            detected_provider = provider
            break

    if not detected_provider:
        detected_provider = "Unknown"

    # Normalize model name: title case with hyphens
    # Keep version numbers and special terms intact
    model_parts = name.replace(" ", "-").replace("_", "-").split("-")
    normalized_parts = []
    for part in model_parts:
        if not part:
            continue
        # Keep uppercase terms (like "OSS", "W4A16", "B200")
        if part.isupper() or any(c.isdigit() for c in part):
            normalized_parts.append(part.upper() if part.isalpha() else part)
        else:
            normalized_parts.append(part.title())

    model_normalized = "-".join(normalized_parts)

    return f"{detected_provider}/{model_normalized}"


def format_display_name(raw_name: str) -> str:
    """Format model name for display (uppercase, spaces instead of hyphens).

    Examples:
        "OpenAI/GPT-OSS-120B" → "OPENAI / GPT OSS 120B"
        "Moonshot/Kimi-K2-Thinking" → "MOONSHOT / KIMI K2 THINKING"
    """
    normalized = normalize_model_name(raw_name)
    if "/" in normalized:
        provider, model = normalized.split("/", 1)
        model_display = model.replace("-", " ")
        return f"{provider.upper()} / {model_display.upper()}"
    return normalized.upper().replace("-", " ")


def format_use_case_name(use_case: str) -> str:
    """Format use case name with proper capitalization for acronyms."""
    if not use_case:
        return "Unknown"
    # Replace underscores and title case
    formatted = use_case.replace("_", " ").title()
    # Fix common acronyms
    acronyms = {
        "Rag": "RAG",
        "Llm": "LLM",
        "Ai": "AI",
        "Api": "API",
        "Gpu": "GPU",
        "Cpu": "CPU",
        "Slo": "SLO",
        "Qps": "QPS",
        "Rps": "RPS",
    }
    for wrong, right in acronyms.items():
        formatted = formatted.replace(wrong, right)
    return formatted


def get_scores(rec: dict) -> dict:
    """Extract normalized scores from a backend recommendation."""
    backend_scores = rec.get("scores", {}) or {}
    return {
        "accuracy": backend_scores.get("accuracy_score", 0),
        "latency": backend_scores.get("latency_score", 0),
        "cost": backend_scores.get("price_score", 0),
        "complexity": backend_scores.get("complexity_score", 0),
        "final": backend_scores.get("balanced_score", 0),
    }


def format_gpu_config(gpu_config: dict) -> str:
    """Format GPU configuration for display.

    Example: "2x A100 (TP=2, R=1)"
    """
    if not isinstance(gpu_config, dict):
        return "Unknown"
    gpu_type = gpu_config.get("gpu_type", "Unknown")
    gpu_count = gpu_config.get("gpu_count", 1)
    tp = gpu_config.get("tensor_parallel", 1)
    replicas = gpu_config.get("replicas", 1)
    return f"{gpu_count}x {gpu_type} (TP={tp}, R={replicas})"
