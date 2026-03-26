"""Model recommendation engine.

INTEGRATION NOTE (Yuval's Quality Scoring):
This module integrates use-case-specific quality scores from Artificial Analysis
benchmarks (weighted_scores CSVs). Quality scoring uses actual benchmark data
(MMLU-Pro, LiveCodeBench, IFBench, etc.) rather than model size heuristics.

Andre's latency/throughput benchmarks from PostgreSQL are kept as-is.
The final recommendation combines: Yuval's quality + Andre's latency/cost/complexity.
"""

import logging
from typing import Optional

from ..context_intent.schema import DeploymentIntent
from ..knowledge_base.model_catalog import ModelCatalog, ModelInfo

logger = logging.getLogger(__name__)

# Try to import use-case quality scorer (Yuval's contribution)
try:
    from .usecase_quality_scorer import score_model_quality, get_quality_scorer
    USE_CASE_QUALITY_AVAILABLE = True
    logger.info("Use-case quality scorer loaded (Artificial Analysis benchmarks)")
except ImportError:
    USE_CASE_QUALITY_AVAILABLE = False
    logger.warning("Use-case quality scorer not available, using size-based heuristics")


class ModelEvaluator:
    """Evaluate models for deployment intent and calculate accuracy scores.
    
    Quality Scoring (updated):
    - If use-case quality data is available: Uses Artificial Analysis benchmark scores
      weighted by use case (e.g., code_completion uses LiveCodeBench 35%, SciCode 30%)
    - Fallback: Uses model size heuristics if quality data unavailable
    """

    def __init__(self, catalog: "Optional[ModelCatalog]" = None):
        """
        Initialize model evaluator.

        Args:
            catalog: Model catalog (creates default if not provided)
        """
        self.catalog = catalog or ModelCatalog()
        self._quality_scorer = get_quality_scorer() if USE_CASE_QUALITY_AVAILABLE else None

    def score_model(self, model: ModelInfo, intent: DeploymentIntent) -> float:
        """
        Score a model for the given intent.

        Args:
            model: Model to score
            intent: Deployment intent

        Returns:
            Score (0-100, higher is better)
        """
        score = 0.0

        # 1. Use case quality match (50 points) - ENHANCED with Artificial Analysis data
        quality_score = self._get_usecase_quality_score(model.name, intent.use_case)
        score += 50 * (quality_score / 100)  # Normalize to 50 points max

        # 2. Domain specialization match (15 points)
        if intent.domain_specialization:
            domain_overlap = set(intent.domain_specialization) & set(model.domain_specialization)
            if domain_overlap:
                score += 15 * (len(domain_overlap) / len(intent.domain_specialization))

        # 3. Latency requirement vs model size (20 points) - Andre's logic preserved
        size_score = self._score_model_size_for_latency(
            model.size_parameters, intent.latency_requirement
        )
        score += 20 * size_score

        # 4. Budget constraint (10 points) - Andre's logic preserved
        budget_score = self._score_model_for_budget(model.size_parameters, intent.budget_constraint)
        score += 10 * budget_score

        # 5. Context length requirement (5 points)
        if intent.use_case in ["summarization", "qa_retrieval", "document_analysis_rag", 
                               "long_document_summarization", "research_legal_analysis"]:
            if model.context_length >= 32000:
                score += 5
            elif model.context_length >= 8192:
                score += 2.5

        logger.debug(f"Scored {model.name}: {score:.1f} (quality: {quality_score:.1f})")
        return score

    def _get_usecase_quality_score(self, model_name: str, use_case: str) -> float:
        """
        Get use-case-specific quality score from Artificial Analysis benchmarks.
        
        Args:
            model_name: Model name
            use_case: Use case identifier
            
        Returns:
            Quality score 0-100 (from weighted benchmarks or fallback heuristic)
        """
        if self._quality_scorer:
            return self._quality_scorer.get_quality_score(model_name, use_case)
        
        # Fallback to simple heuristic if quality scorer not available
        return 60.0  # Default moderate score

    def _score_model_size_for_latency(self, size_str: str, latency_requirement: str) -> float:
        """
        Score model size appropriateness for latency requirement.

        Args:
            size_str: Model size (e.g., "8B", "70B", "8x7B")
            latency_requirement: Latency sensitivity

        Returns:
            Score 0-1
        """
        # Extract approximate parameter count
        param_count = self._extract_param_count(size_str)

        # Latency requirement to preferred size mapping
        preference_map = {
            "very_high": (0, 10),  # Prefer <10B
            "high": (0, 15),  # Prefer <15B
            "medium": (7, 80),  # 7-80B is fine (allow larger for quality)
            "low": (20, 200),  # Larger models preferred
        }

        min_pref, max_pref = preference_map.get(latency_requirement, (0, 100))

        # Score based on how close to preference range
        if min_pref <= param_count <= max_pref:
            return 1.0
        elif param_count < min_pref:
            # Too small - might not have enough capability
            return 0.7 + 0.3 * (param_count / min_pref)
        else:
            # Too large - might be too slow
            excess = param_count - max_pref
            return max(0.3, 1.0 - (excess / 100))

    def _score_model_for_budget(self, size_str: str, budget_constraint: str) -> float:
        """
        Score model appropriateness for budget constraint.

        Args:
            size_str: Model size
            budget_constraint: Budget sensitivity

        Returns:
            Score 0-1
        """
        param_count = self._extract_param_count(size_str)

        # Budget constraint to preferred size mapping
        preference_map = {
            "strict": (0, 10),  # Prefer small models
            "moderate": (7, 30),  # Mid-size OK
            "flexible": (20, 200),  # Prefer larger models for quality
            "none": (30, 200),  # Prefer largest models
        }

        min_pref, max_pref = preference_map.get(budget_constraint, (0, 100))

        if min_pref <= param_count <= max_pref:
            return 1.0
        elif param_count > max_pref:
            # Over budget is OK if flexible/none
            if budget_constraint in ["flexible", "none"]:
                return 1.0
            excess = param_count - max_pref
            return max(0.2, 1.0 - (excess / 100))
        else:
            # Under budget - penalize for flexible/none (want larger models)
            if budget_constraint in ["flexible", "none"]:
                return 0.5
            return 0.8  # Smaller than needed is OK for strict/moderate budget

    def _extract_param_count(self, size_str: str) -> float:
        """
        Extract approximate parameter count from size string.

        Args:
            size_str: Size string (e.g., "8B", "70B", "8x7B")

        Returns:
            Approximate parameter count in billions
        """
        try:
            # Handle "8B", "70B" format
            if "B" in size_str and "x" not in size_str:
                return float(size_str.replace("B", ""))

            # Handle "8x7B" MoE format (approximate as total params)
            if "x" in size_str and "B" in size_str:
                parts = size_str.replace("B", "").split("x")
                return float(parts[0]) * float(parts[1])

            # Fallback
            return 10.0
        except Exception:
            logger.warning(f"Could not parse size string: {size_str}")
            return 10.0
