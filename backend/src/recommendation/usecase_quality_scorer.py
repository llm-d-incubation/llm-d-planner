"""Use-case specific model quality scoring based on Artificial Analysis benchmarks.

This module provides quality/accuracy scores for models based on their performance
on task-specific benchmarks (MMLU-Pro, LiveCodeBench, IFBench, etc.).

Integration with Compass:
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
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
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
                    model_name = row.get('Model Name', '').strip()
                    score_str = row.get('Use Case Score', '0')
                    
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
    
    def get_quality_score(self, model_name: str, use_case: str) -> float:
        """Get quality score for a model on a specific use case.
        
        Args:
            model_name: Model name (e.g., "Llama 3.1 Instruct 8B", "meta-llama/llama-3.1-8b-instruct")
            use_case: Use case identifier (e.g., "code_completion")
            
        Returns:
            Quality score 0-100 (higher is better)
        """
        # Normalize use case
        use_case_normalized = use_case.lower().replace(" ", "_").replace("-", "_")
        
        if use_case_normalized not in self._cache:
            logger.warning(f"Unknown use case: {use_case}, using chatbot_conversational")
            use_case_normalized = "chatbot_conversational"
        
        scores = self._cache.get(use_case_normalized, {})
        
        # Try exact match first
        model_lower = model_name.lower()
        if model_lower in scores:
            return scores[model_lower]
        
        # Try partial matching (for HuggingFace repo names)
        for cached_name, score in scores.items():
            model_words = set(model_lower.replace("-", " ").replace("/", " ").replace("_", " ").split())
            cached_words = set(cached_name.replace("-", " ").replace("/", " ").replace("_", " ").split())
            
            common_words = model_words & cached_words
            if len(common_words) >= 3:
                return score
        
        # Fallback: return median score for the use case
        if scores:
            median_score = sorted(scores.values())[len(scores) // 2]
            logger.debug(f"No score found for {model_name}, using median: {median_score:.1f}")
            return median_score
        
        return 50.0  # Default fallback
    
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

