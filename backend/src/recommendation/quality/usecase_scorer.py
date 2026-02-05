"""Use-case specific model quality scoring based on Artificial Analysis benchmarks.

This module provides quality/accuracy scores for models based on their performance
on task-specific benchmarks (MMLU-Pro, LiveCodeBench, IFBench, etc.).

Integration with NeuralNav:
- This REPLACES the size-based accuracy heuristic in model_evaluator.score_model()
- Andre's latency/throughput benchmarks from PostgreSQL are KEPT as-is
- The final recommendation combines: Our quality + Andre's latency/cost/complexity
"""

import csv
import logging
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Base path for weighted scores CSVs
# Path: quality/usecase_scorer.py -> recommendation -> src -> backend -> project root -> data
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data")
WEIGHTED_SCORES_DIR = os.path.join(DATA_DIR, "business_context", "use_case", "weighted_scores")


class UseCaseQualityScorer:
    """Score models based on use-case specific benchmark performance.
    
    Uses pre-calculated weighted scores from Artificial Analysis benchmarks.
    Each use case has different weights for different benchmarks:
    - chatbot_conversational: MMLU-Pro (30%), IFBench (30%), HLE (20%), etc.
    - code_completion: LiveCodeBench (35%), SciCode (30%), etc.
    - See USE_CASE_METHODOLOGY.md for full details.
    """
    
    # Mapping from use case to CSV filename
    USE_CASE_FILES = {
        "chatbot_conversational": "opensource_chatbot_conversational.csv",
        "code_completion": "opensource_code_completion.csv",
        "code_generation_detailed": "opensource_code_generation_detailed.csv",
        "translation": "opensource_translation.csv",
        "content_generation": "opensource_content_generation.csv",
        "summarization_short": "opensource_summarization_short.csv",
        "document_analysis_rag": "opensource_document_analysis_rag.csv",
        "long_document_summarization": "opensource_long_document_summarization.csv",
        "research_legal_analysis": "opensource_research_legal_analysis.csv",
    }
    
    def __init__(self):
        """Initialize the scorer with cached data."""
        self._cache: Dict[str, Dict[str, float]] = {}
        self._load_all_scores()
    
    def _load_all_scores(self):
        """Pre-load all use case scores into memory."""
        for use_case, filename in self.USE_CASE_FILES.items():
            filepath = os.path.join(WEIGHTED_SCORES_DIR, filename)
            if os.path.exists(filepath):
                self._cache[use_case] = self._load_csv_scores(filepath)
                logger.info(f"Loaded {len(self._cache[use_case])} model scores for {use_case}")
            else:
                logger.warning(f"Weighted scores file not found: {filepath}")
                self._cache[use_case] = {}
    
    def _load_csv_scores(self, filepath: str) -> Dict[str, float]:
        """Load scores from a weighted_scores CSV file.
        
        Returns:
            Dict mapping model name (lowercase) to score (0-100)
        """
        scores = {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Support both old format (Model Name, Use Case Score) and new format (model_name, weighted_score)
                    model_name = row.get('model_name', row.get('Model Name', '')).strip()
                    score_str = row.get('weighted_score', row.get('Use Case Score', '0'))
                    
                    # Parse score (handle percentage strings)
                    try:
                        if '%' in str(score_str):
                            score = float(score_str.replace('%', ''))
                        else:
                            score = float(score_str) * 100 if float(score_str) <= 1 else float(score_str)
                    except (ValueError, TypeError):
                        score = 0.0
                    
                    if model_name:
                        # Store with lowercase key for easier matching
                        scores[model_name.lower()] = min(100, max(0, score))
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
        
        return scores
    
    # Benchmark model variant to AA model mapping (for ALL 40 Red Hat DB models)
    BENCHMARK_TO_AA_MAP = {
        # === GPT-OSS (61.62%, 55.23%) ===
        "gpt-oss-120b": "gpt-oss-120b (high)",
        "gpt-oss-20b": "gpt-oss-20b (high)",
        
        # === DeepSeek R1 (52.20%) ===
        "deepseek-r1-0528-quantized.w4a16": "deepseek r1 0528 (may '25)",
        
        # === Kimi K2 (49.61%) ===
        "kimi-k2-instruct-quantized.w4a16": "kimi k2",
        
        # === Llama 4 Maverick (46.86%) ===
        "llama-4-maverick-17b-128e-instruct": "llama 4 maverick",
        "llama-4-maverick-17b-128e-instruct-fp8": "llama 4 maverick",
        
        # === Qwen 2.5 7B (44.71%) ===
        "qwen2.5-7b-instruct": "qwen2.5 max",
        "qwen2.5-7b-instruct-fp8-dynamic": "qwen2.5 max",
        "qwen2.5-7b-instruct-quantized.w4a16": "qwen2.5 max",
        "qwen2.5-7b-instruct-quantized.w8a8": "qwen2.5 max",
        
        # === Llama 3.3 70B (42.99%) ===
        "llama-3.3-70b-instruct": "llama 3.3 instruct 70b",
        "llama-3.3-70b-instruct-fp8-dynamic": "llama 3.3 instruct 70b",
        "llama-3.3-70b-instruct-quantized.w4a16": "llama 3.3 instruct 70b",
        "llama-3.3-70b-instruct-quantized.w8a8": "llama 3.3 instruct 70b",
        
        # === Llama 4 Scout (42.42%) ===
        "llama-4-scout-17b-16e-instruct": "llama 4 scout",
        "llama-4-scout-17b-16e-instruct-fp8-dynamic": "llama 4 scout",
        "llama-4-scout-17b-16e-instruct-quantized.w4a16": "llama 4 scout",
        
        # === Qwen3 8B (40.37%) ===
        "qwen3-8b-fp8": "qwen3 8b (reasoning)",
        "qwen3-8b-fp8-dynamic": "qwen3 8b (reasoning)",
        
        # === NVIDIA Nemotron Nano 9B (39.89%) ===
        "nvidia-nemotron-nano-9b-v2-fp8-dynamic": "nvidia nemotron nano 9b v2 (reasoning)",
        
        # === Qwen3 Coder 480B (38.75% - using Jamba Reasoning) ===
        "qwen3-coder-480b-a35b-instruct-fp8": "qwen3 coder 480b a35b instruct",
        
        # === Llama 3.1 Nemotron 70B (36.69%) ===
        "llama-3.1-nemotron-70b-instruct-hf": "llama 3.1 nemotron instruct 70b",
        "llama-3.1-nemotron-70b-instruct-hf-fp8-dynamic": "llama 3.1 nemotron instruct 70b",
        
        # === Mistral Small 3.1 (35.70%) ===
        "mistral-small-3.1-24b-instruct-2503": "mistral small 3.1",
        "mistral-small-3.1-24b-instruct-2503-fp8-dynamic": "mistral small 3.1",
        "mistral-small-3.1-24b-instruct-2503-quantized.w4a16": "mistral small 3.1",
        "mistral-small-3.1-24b-instruct-2503-quantized.w8a8": "mistral small 3.1",
        
        # === Phi-4 (35.57%) ===
        "phi-4": "phi-4",
        "phi-4-fp8-dynamic": "phi-4",
        "phi-4-quantized.w4a16": "phi-4",
        "phi-4-quantized.w8a8": "phi-4",
        
        # === Mistral Small 24B (33.79%) ===
        "mistral-small-24b-instruct-2501": "mistral small 3",
        
        # === Llama 3.1 8B (27.71%) ===
        "llama-3.1-8b-instruct": "llama 3.1 instruct 8b",
        "meta-llama-3.1-8b-instruct-fp8-dynamic": "llama 3.1 instruct 8b",
        
        # === Gemma 3n (27.69%) ===
        "gemma-3n-e4b-it-fp8-dynamic": "gemma 3n e4b instruct",
        
        # === Granite 3.1 8B (25.57%) ===
        "granite-3.1-8b-instruct": "granite 3.3 8b (non-reasoning)",
        "granite-3.1-8b-instruct-fp8-dynamic": "granite 3.3 8b (non-reasoning)",
        "granite-3.1-8b-instruct-quantized.w4a16": "granite 3.3 8b (non-reasoning)",
        
        # === Mixtral 8x7B (20.51%) ===
        "mixtral-8x7b-instruct-v0.1": "mixtral 8x7b instruct",
        
        # === SmolLM3 3B (estimated ~20%) ===
        "smollm3-3b-fp8-dynamic": "smollm3 3b",
    }
    
    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name by removing quantization suffixes and org prefixes."""
        name = model_name.lower()
        
        # Remove org prefixes
        if '/' in name:
            name = name.split('/')[-1]
        
        # Remove quantization suffixes
        suffixes_to_remove = [
            '-fp8-dynamic', '-fp8', 
            '-quantized.w4a16', '-quantized.w8a8',
            '-instruct-2501', '-instruct-2503', '-instruct-hf',
            '-instruct-v0.1', '-instruct'
        ]
        for suffix in suffixes_to_remove:
            name = name.replace(suffix, '')
        
        return name.strip('-').strip()
    
    def get_quality_score(self, model_name: str, use_case: str) -> float:
        """Get quality score for a model on a specific use case.
        
        Args:
            model_name: Model name (e.g., "Llama 3.1 Instruct 8B", "meta-llama/llama-3.1-8b-instruct")
            use_case: Use case identifier (e.g., "code_completion")
            
        Returns:
            Quality score 0-100 (higher is better), or 0 if no valid AA data
        """
        # Normalize use case
        use_case_normalized = use_case.lower().replace(" ", "_").replace("-", "_")
        
        if use_case_normalized not in self._cache:
            logger.warning(f"Unknown use case: {use_case}, using chatbot_conversational")
            use_case_normalized = "chatbot_conversational"
        
        scores = self._cache.get(use_case_normalized, {})
        
        # Normalize the model name
        model_lower = model_name.lower()
        base_model = self._normalize_model_name(model_name)
        
        # Try exact match first
        if model_lower in scores:
            return scores[model_lower]
        
        # Try benchmark to AA mapping (for known valid models)
        for benchmark_pattern, aa_name in self.BENCHMARK_TO_AA_MAP.items():
            if benchmark_pattern in base_model:
                if aa_name in scores:
                    logger.debug(f"Matched {model_name} -> {aa_name} via benchmark mapping")
                    return scores[aa_name]
        
        # Try partial matching (for HuggingFace repo names)
        # Find BEST match - prioritize matches that include model size (7b, 20b, 120b, etc.)
        import re
        size_pattern = re.compile(r'\b(\d+(?:\.\d+)?[bB])\b')
        model_sizes = set(s.lower() for s in size_pattern.findall(base_model))
        
        best_match = None
        best_score = 0.0
        best_common_count = 0
        best_has_size_match = False
        
        for cached_name, score in scores.items():
            model_words = set(base_model.replace("-", " ").replace("/", " ").replace("_", " ").split())
            cached_words = set(cached_name.replace("-", " ").replace("/", " ").replace("_", " ").split())
            
            common_words = model_words & cached_words
            if len(common_words) >= 2:
                # Check if this match includes the model size
                cached_sizes = set(s.lower() for s in size_pattern.findall(cached_name))
                has_size_match = bool(model_sizes & cached_sizes)
                
                # Prefer matches with size match, then by common word count
                is_better = False
                if has_size_match and not best_has_size_match:
                    is_better = True  # Size match beats no size match
                elif has_size_match == best_has_size_match:
                    if len(common_words) > best_common_count:
                        is_better = True  # More common words is better
                
                if is_better:
                    best_match = cached_name
                    best_score = score
                    best_common_count = len(common_words)
                    best_has_size_match = has_size_match
        
        if best_match:
            logger.debug(f"Partial match {model_name} -> {best_match} (size_match={best_has_size_match})")
            return best_score
        
        # No valid AA data found - return 0 to indicate missing data
        # This allows filtering out models without quality scores
        logger.debug(f"No AA score found for {model_name} (base: {base_model})")
        return 0.0  # Return 0 so min_accuracy filter can exclude these
    
    def get_top_models_for_usecase(self, use_case: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get top N models for a specific use case."""
        use_case_normalized = use_case.lower().replace(" ", "_").replace("-", "_")
        scores = self._cache.get(use_case_normalized, {})
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_models[:top_n]
    
    def get_available_use_cases(self) -> List[str]:
        """Get list of available use cases."""
        return list(self.USE_CASE_FILES.keys())


# Singleton instance
_scorer_instance: Optional[UseCaseQualityScorer] = None


def get_quality_scorer() -> UseCaseQualityScorer:
    """Get the singleton quality scorer instance."""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = UseCaseQualityScorer()
    return _scorer_instance


def score_model_quality(model_name: str, use_case: str) -> float:
    """Convenience function to get quality score."""
    return get_quality_scorer().get_quality_score(model_name, use_case)

