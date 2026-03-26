"""Solution scoring for multi-criteria recommendation ranking.

Scores deployment configurations on 4 criteria (0-100 scale):
- Accuracy/Quality: Model capability (from Artificial Analysis benchmarks or param count fallback)
- Price: Cost efficiency (inverse of cost, normalized)  
- Latency: SLO compliance and headroom (from Andre's PostgreSQL benchmarks)
- Complexity: Deployment simplicity (fewer GPUs = simpler)

INTEGRATION NOTE:
- Quality scoring: Uses Yuval's weighted_scores CSVs (Artificial Analysis benchmarks)
- Latency/Price/Complexity: Uses Andre's scoring logic and benchmark data
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import use-case quality scorer
try:
    from .usecase_quality_scorer import score_model_quality
    USE_CASE_QUALITY_AVAILABLE = True
except ImportError:
    USE_CASE_QUALITY_AVAILABLE = False


class SolutionScorer:
    """Score deployment configurations on 4 criteria (0-100 scale)."""

    # Accuracy tiers based on model parameter count (in billions)
    # Larger models generally have higher accuracy/capability
    ACCURACY_TIERS = {
        3: 40,
        4: 45,
        7: 55,
        8: 60,
        9: 62,
        14: 70,
        17: 72,
        20: 75,
        24: 78,
        27: 80,
        70: 85,
        120: 90,
        405: 95,
        480: 98,
    }

    # Complexity scores based on total GPU count
    COMPLEXITY_SCORES = {
        1: 100,
        2: 90,
        3: 82,
        4: 75,
        5: 70,
        6: 65,
        7: 62,
        8: 60,
    }

    # Default weights for balanced score
    DEFAULT_WEIGHTS = {
        "accuracy": 0.40,
        "price": 0.40,
        "latency": 0.10,
        "complexity": 0.10,
    }

    def score_accuracy(self, model_size_str: str, model_name: Optional[str] = None, 
                        use_case: Optional[str] = None) -> int:
        """
        Score model accuracy/quality.
        
        Priority:
        1. Use-case specific benchmark score (Artificial Analysis data) if available
        2. Fallback to model size-based heuristic (Andre's original logic)

        Args:
            model_size_str: Model size string (e.g., "8B", "70B", "8x7B")
            model_name: Optional model name for use-case-specific scoring
            use_case: Optional use case for benchmark-based scoring

        Returns:
            Score 0-100
        """
        # Try use-case-specific quality scoring first (Yuval's contribution)
        if USE_CASE_QUALITY_AVAILABLE and model_name and use_case:
            quality_score = score_model_quality(model_name, use_case)
            if quality_score > 0:
                logger.debug(f"Quality score for {model_name} ({use_case}): {quality_score:.1f}")
                return int(quality_score)
        
        # Fallback to size-based heuristic (Andre's original logic)
        return self._score_accuracy_by_size(model_size_str)
    
    def _score_accuracy_by_size(self, model_size_str: str) -> int:
        """
        Score model accuracy based on parameter count tier (fallback).

        Args:
            model_size_str: Model size string (e.g., "8B", "70B", "8x7B")

        Returns:
            Score 0-100
        """
        param_count = self._extract_param_count(model_size_str)

        # Find the closest tier at or below the param count
        best_score = 40  # minimum score
        for tier_size, tier_score in sorted(self.ACCURACY_TIERS.items()):
            if param_count >= tier_size:
                best_score = tier_score
            else:
                break

        logger.debug(f"Accuracy score for {model_size_str} ({param_count}B): {best_score}")
        return best_score

    def score_price(self, cost_per_month: float, min_cost: float, max_cost: float) -> int:
        """
        Score price using normalized inverse cost formula.

        Formula: 100 * (max_cost - config_cost) / (max_cost - min_cost)

        Args:
            cost_per_month: Configuration cost in USD/month
            min_cost: Minimum cost among all configurations
            max_cost: Maximum cost among all configurations

        Returns:
            Score 0-100 (100 = cheapest, 0 = most expensive)
        """
        if max_cost == min_cost:
            # All configs have same cost
            return 100

        # Clamp cost to range
        cost = max(min_cost, min(max_cost, cost_per_month))

        score = int(100 * (max_cost - cost) / (max_cost - min_cost))
        logger.debug(f"Price score for ${cost_per_month:.0f}/mo: {score}")
        return score

    def score_latency(
        self,
        predicted_ttft_ms: int,
        predicted_itl_ms: int,
        predicted_e2e_ms: int,
        target_ttft_ms: int,
        target_itl_ms: int,
        target_e2e_ms: int,
    ) -> tuple[int, str]:
        """
        Score latency based on SLO compliance and headroom.

        Args:
            predicted_ttft_ms: Predicted TTFT p95 in ms
            predicted_itl_ms: Predicted ITL p95 in ms
            predicted_e2e_ms: Predicted E2E p95 in ms
            target_ttft_ms: Target TTFT p95 in ms
            target_itl_ms: Target ITL p95 in ms
            target_e2e_ms: Target E2E p95 in ms

        Returns:
            Tuple of (score 0-100, slo_status)
            - slo_status: "compliant", "near_miss", or "exceeds"
        """
        # Calculate ratio of predicted to target for each metric
        # Lower ratio = better (more headroom)
        ratios = []

        if target_ttft_ms > 0:
            ratios.append(predicted_ttft_ms / target_ttft_ms)
        if target_itl_ms > 0:
            ratios.append(predicted_itl_ms / target_itl_ms)
        if target_e2e_ms > 0:
            ratios.append(predicted_e2e_ms / target_e2e_ms)

        if not ratios:
            return 90, "compliant"

        # Use worst (highest) ratio to determine compliance
        worst_ratio = max(ratios)

        if worst_ratio <= 1.0:
            # SLO compliant - score 90-100 based on headroom
            # ratio of 1.0 = score 90 (just meeting SLO)
            # ratio of 0.5 = score 100 (50% headroom)
            headroom_bonus = int(10 * (1.0 - worst_ratio))
            score = 90 + headroom_bonus
            slo_status = "compliant"
        elif worst_ratio <= 1.2:
            # Near miss (within 20%) - score 70-89
            # ratio of 1.0 = 89, ratio of 1.2 = 70
            score = int(89 - (worst_ratio - 1.0) * 95)
            score = max(70, min(89, score))
            slo_status = "near_miss"
        else:
            # Exceeds SLO by more than 20% - score 0-69
            # ratio of 1.2 = 69, ratio of 2.0 = 0
            score = int(69 - (worst_ratio - 1.2) * 86)
            score = max(0, min(69, score))
            slo_status = "exceeds"

        logger.debug(
            f"Latency score: {score} ({slo_status}) - "
            f"TTFT={predicted_ttft_ms}/{target_ttft_ms}, "
            f"ITL={predicted_itl_ms}/{target_itl_ms}, "
            f"E2E={predicted_e2e_ms}/{target_e2e_ms}"
        )
        return score, slo_status

    def score_complexity(self, total_gpu_count: int) -> int:
        """
        Score complexity based on deployment topology.

        Args:
            total_gpu_count: Total GPUs required (tensor_parallel * replicas)

        Returns:
            Score 0-100 (100 = simplest, lower = more complex)
        """
        # Use predefined scores or calculate for larger counts
        if total_gpu_count in self.COMPLEXITY_SCORES:
            score = self.COMPLEXITY_SCORES[total_gpu_count]
        elif total_gpu_count > 8:
            # Linear decay for very large deployments
            score = max(40, 60 - (total_gpu_count - 8) * 2)
        else:
            score = 60

        logger.debug(f"Complexity score for {total_gpu_count} GPUs: {score}")
        return score

    def score_balanced(
        self,
        accuracy_score: int,
        price_score: int,
        latency_score: int,
        complexity_score: int,
        weights: Optional[dict] = None,
    ) -> float:
        """
        Calculate weighted composite score.

        Args:
            accuracy_score: Accuracy score (0-100)
            price_score: Price score (0-100)
            latency_score: Latency score (0-100)
            complexity_score: Complexity score (0-100)
            weights: Optional custom weights (default: 40% accuracy, 40% price,
                     10% latency, 10% complexity)

        Returns:
            Weighted composite score (0-100)
        """
        w = weights or self.DEFAULT_WEIGHTS

        balanced = (
            accuracy_score * w["accuracy"]
            + price_score * w["price"]
            + latency_score * w["latency"]
            + complexity_score * w["complexity"]
        )

        logger.debug(
            f"Balanced score: {balanced:.1f} "
            f"(A={accuracy_score}, P={price_score}, L={latency_score}, C={complexity_score})"
        )
        return round(balanced, 1)

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
            if "B" in size_str and "x" not in size_str.lower():
                # Extract number before B
                match = re.search(r"(\d+\.?\d*)\s*B", size_str, re.IGNORECASE)
                if match:
                    return float(match.group(1))

            # Handle "8x7B" MoE format (use total params)
            if "x" in size_str.lower() and "B" in size_str.upper():
                match = re.search(r"(\d+)\s*x\s*(\d+\.?\d*)\s*B", size_str, re.IGNORECASE)
                if match:
                    return float(match.group(1)) * float(match.group(2))

            # Fallback: try to extract any number
            match = re.search(r"(\d+\.?\d*)", size_str)
            if match:
                return float(match.group(1))

            return 10.0  # Default fallback
        except Exception:
            logger.warning(f"Could not parse size string: {size_str}")
            return 10.0
