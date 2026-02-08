"""Ranking service for multi-criteria recommendation sorting.

ACCURACY-FIRST STRATEGY:
1. Get top N unique models by raw accuracy (quality baseline)
2. Filter all hardware configs to only these high-quality models
3. Best Latency/Cost/etc. are ranked WITHIN this quality tier

This ensures:
- All recommendations show HIGH QUALITY models
- No "fast but useless" or "cheap but terrible" recommendations
- Cards show different trade-offs within the same quality tier

TASK-SPECIFIC BONUSES (Balanced card only):
- Different model types get bonuses for specific use cases
- E.g., Code models get +8 for code tasks, multilingual for translation
- This diversifies recommendations across use cases
"""

import logging

from src.shared.schemas import DeploymentRecommendation

logger = logging.getLogger(__name__)


# Task-specific bonuses for Balanced score diversification
# Format: {use_case: {model_keyword: bonus_points}}
#
# STRATEGY: Kimi K2 has high base accuracy (~71-78), so it needs LOW or ZERO
# bonus in most tasks to allow task-specific models to win.
# Task-specialist models need HIGH bonuses to overcome accuracy gaps.
#
TASK_BONUSES = {
    "code_completion": {
        # Code-specialized models get HIGHEST bonus (DeepSeek excels at code)
        "coder": 20,
        "starcoder": 20,
        "codellama": 20,
        "deepseek": 20,  # DeepSeek R1 excellent for code
        "gpt-oss": 12,
        "qwen": 10,
        # Kimi: NO bonus (already high accuracy, not code-specialized)
    },
    "code_generation_detailed": {
        "coder": 20,
        "starcoder": 20,
        "codellama": 20,
        "deepseek": 20,
        "gpt-oss": 12,
        "qwen": 10,
    },
    "chatbot_conversational": {
        # GPT-OSS is good for general chat
        "gpt-oss": 15,
        "llama": 12,
        "qwen": 12,
        "mistral": 12,
        "instruct": 8,
        "chat": 8,
        "gemma": 8,
        # Kimi: small bonus (good but not specialized for casual chat)
        "kimi": 5,
    },
    "translation": {
        # Multilingual specialists get highest bonus
        "qwen": 20,  # Qwen is strongly multilingual
        "aya": 20,
        "nllb": 20,
        "llama": 10,
        "gpt-oss": 8,
        "mistral": 8,
        # Kimi: small bonus (not primarily multilingual)
        "kimi": 5,
    },
    "content_generation": {
        # Creative/general models
        "llama": 18,
        "gpt-oss": 15,
        "mistral": 15,
        "qwen": 12,
        "gemma": 10,
        # Kimi: reasoning-focused, not creative writing
        "kimi": 3,
    },
    "summarization_short": {
        # Fast, efficient summarizers - favor smaller/faster models
        "llama": 15,
        "qwen": 15,
        "mistral": 15,
        "gpt-oss": 12,
        "gemma": 10,
        "kimi": 5,
    },
    "long_document_summarization": {
        # Long-context specialists
        "qwen": 18,  # Qwen has good long context
        "minimax": 18,  # MiniMax designed for long context
        "llama": 12,
        "mistral": 12,
        "gpt-oss": 10,
        "kimi": 8,  # Kimi has good context but others specialize
    },
    "document_analysis_rag": {
        # Analytical and retrieval-augmented
        "llama": 15,
        "qwen": 15,
        "gpt-oss": 12,
        "mistral": 12,
        "gemma": 10,
        "kimi": 8,
    },
    "research_legal_analysis": {
        # Complex reasoning - Kimi CAN get bonus here (reasoning is its strength)
        "kimi": 15,  # Kimi Thinking excels at reasoning
        "deepseek": 18,  # DeepSeek R1 also excellent for reasoning
        "qwen": 12,
        "gpt-oss": 10,
        "llama": 10,
        "mistral": 8,
    },
}


def get_task_bonus(model_name: str, use_case: str) -> int:
    """
    Get task-specific bonus for a model based on use case.

    Args:
        model_name: Model name (e.g., "Moonshot/Kimi-K2-Thinking")
        use_case: Use case identifier (e.g., "code_completion")

    Returns:
        Bonus points (0-10) to add to balanced score
    """
    if not model_name or not use_case:
        return 0

    model_lower = model_name.lower()
    bonuses = TASK_BONUSES.get(use_case, {})

    # Find matching bonus (first match wins)
    for keyword, bonus in bonuses.items():
        if keyword in model_lower:
            return bonus

    return 0


class Analyzer:
    """Generate ranked recommendation lists from scored configurations."""

    def generate_ranked_lists(
        self,
        configurations: list[DeploymentRecommendation],
        min_accuracy: int | None = None,
        max_cost: float | None = None,
        top_n: int = 5,
        weights: dict[str, int] | None = None,
        use_case: str | None = None,
    ) -> dict[str, list[DeploymentRecommendation]]:
        """
        Generate 5 ranked lists using ACCURACY-FIRST strategy.

        Strategy:
        1. Get top N unique models by raw accuracy (quality baseline)
        2. Filter ALL hardware configs to only these high-quality models
        3. Best Latency = fastest hardware among high-quality models
        4. Best Cost = cheapest hardware among high-quality models
        5. Balanced = best weighted score among high-quality models

        Args:
            configurations: List of scored DeploymentRecommendations
            min_accuracy: Minimum accuracy score filter (0-100)
            max_cost: Maximum monthly cost filter (USD)
            top_n: Number of top configurations to return per list
            weights: Optional custom weights for balanced score (0-10 scale)
                     Keys: accuracy, price, latency, complexity

        Returns:
            Dict with keys: best_accuracy, lowest_cost, lowest_latency,
                           simplest, balanced
        """
        # Apply filters
        filtered = self._apply_filters(configurations, min_accuracy, max_cost)

        # # Recalculate balanced scores with custom weights and task bonuses
        # if filtered:
        #     self._recalculate_balanced_scores(filtered, weights or {}, use_case)

        if not filtered:
            logger.warning("No configurations remain after filtering")
            return {
                "best_accuracy": [],
                "lowest_cost": [],
                "lowest_latency": [],
                "simplest": [],
                "balanced": [],
            }

        # =====================================================================
        # STEP 1: Get top N UNIQUE MODELS by raw accuracy (quality baseline)
        # Tie-breakers follow priority order: Accuracy → Cost → Latency → Complexity
        # =====================================================================

        # Helper functions for sort keys (higher score = better, lower cost = better)
        def get_accuracy(x):
            return x.scores.accuracy_score if x.scores else 0

        def get_cost_inverted(x):
            # Invert cost so lower cost sorts higher (better)
            cost = x.cost_per_month_usd or float("inf")
            return -cost

        def get_latency(x):
            return x.scores.latency_score if x.scores else 0

        def get_complexity(x):
            return x.scores.complexity_score if x.scores else 0

        def get_balanced(x):
            return x.scores.balanced_score if x.scores else 0.0

        seen_models = set()
        unique_accuracy_configs = []
        # Primary: Accuracy, Tie-breakers: Cost → Latency → Complexity
        sorted_by_accuracy = sorted(
            filtered,
            key=lambda x: (
                get_accuracy(x),
                get_cost_inverted(x),
                get_latency(x),
                get_complexity(x),
            ),
            reverse=True,
        )

        for config in sorted_by_accuracy:
            model_name = config.model_name or config.model_id or "Unknown"
            if model_name not in seen_models:
                seen_models.add(model_name)
                unique_accuracy_configs.append(config)
                if len(unique_accuracy_configs) >= top_n:
                    break

        # Get the model names of top accuracy models
        top_accuracy_model_names = {c.model_name or c.model_id for c in unique_accuracy_configs}

        logger.info(
            f"ACCURACY-FIRST: Top {len(top_accuracy_model_names)} models by accuracy: "
            f"{list(top_accuracy_model_names)[:5]}"
        )

        # =====================================================================
        # STEP 2: Filter ALL configs to only high-quality models
        # This ensures Best Latency and Best Cost show HIGH QUALITY models
        # =====================================================================
        high_quality_configs = [
            c for c in filtered if (c.model_name or c.model_id) in top_accuracy_model_names
        ]

        logger.info(
            f"ACCURACY-FIRST: {len(high_quality_configs)} configs from top {len(top_accuracy_model_names)} models"
        )

        # =====================================================================
        # STEP 3: Generate ranked lists from HIGH-QUALITY configs only
        # Tie-breaker order: Accuracy → Cost → Latency → Complexity
        # Each list excludes its primary criterion from tie-breakers
        # =====================================================================
        ranked_lists = {
            # Best Accuracy: Top N unique models (one config per model)
            # Already sorted with tie-breakers: Cost → Latency → Complexity
            "best_accuracy": unique_accuracy_configs[:top_n],
            # Best Cost: Primary=Cost, Tie-breakers: Accuracy → Latency → Complexity
            "lowest_cost": sorted(
                high_quality_configs,
                key=lambda x: (
                    get_cost_inverted(x),
                    get_accuracy(x),
                    get_latency(x),
                    get_complexity(x),
                ),
                reverse=True,
            )[:top_n],
            # Best Latency: Primary=Latency, Tie-breakers: Accuracy → Cost → Complexity
            "lowest_latency": sorted(
                high_quality_configs,
                key=lambda x: (
                    get_latency(x),
                    get_accuracy(x),
                    get_cost_inverted(x),
                    get_complexity(x),
                ),
                reverse=True,
            )[:top_n],
            # Simplest: Primary=Complexity, Tie-breakers: Accuracy → Cost → Latency
            "simplest": sorted(
                high_quality_configs,
                key=lambda x: (
                    get_complexity(x),
                    get_accuracy(x),
                    get_cost_inverted(x),
                    get_latency(x),
                ),
                reverse=True,
            )[:top_n],
            # Balanced: Primary=Balanced, Tie-breakers: Accuracy → Cost → Latency → Complexity
            "balanced": sorted(
                high_quality_configs,
                key=lambda x: (
                    get_balanced(x),
                    get_accuracy(x),
                    get_cost_inverted(x),
                    get_latency(x),
                    get_complexity(x),
                ),
                reverse=True,
            )[:top_n],
        }

        logger.info(
            f"Generated ranked lists (ACCURACY-FIRST): {len(filtered)} total configs, "
            f"{len(high_quality_configs)} high-quality, top {top_n} per criterion"
        )

        return ranked_lists

    def _apply_filters(
        self,
        configs: list[DeploymentRecommendation],
        min_accuracy: int | None,
        max_cost: float | None,
    ) -> list[DeploymentRecommendation]:
        """
        Apply accuracy and cost filters to configurations.

        Args:
            configs: List of configurations to filter
            min_accuracy: Minimum accuracy score (0-100), None = no filter
            max_cost: Maximum monthly cost (USD), None = no filter

        Returns:
            Filtered list of configurations
        """
        filtered = configs

        # Filter by minimum accuracy
        if min_accuracy is not None and min_accuracy > 0:
            filtered = [c for c in filtered if c.scores and c.scores.accuracy_score >= min_accuracy]
            logger.debug(f"After min_accuracy={min_accuracy} filter: {len(filtered)} configs")

        # Filter by maximum cost
        if max_cost is not None and max_cost > 0:
            filtered = [
                c
                for c in filtered
                if c.cost_per_month_usd is not None and c.cost_per_month_usd <= max_cost
            ]
            logger.debug(f"After max_cost=${max_cost} filter: {len(filtered)} configs")

        return filtered

    def _recalculate_balanced_scores(
        self,
        configs: list[DeploymentRecommendation],
        weights: dict[str, int],
        use_case: str | None = None,
    ) -> None:
        """
        Recalculate balanced scores with task-specific bonuses.

        Formula: Balanced = (Accuracy + Task_Bonus) × 70% + (Latency + Cost) / 2 × 30%

        Task bonuses diversify recommendations by boosting models suited for specific tasks:
        - Code models get +8 for code tasks
        - Multilingual models get +8 for translation
        - General models get bonuses for chatbot/content tasks

        Args:
            configs: List of configurations to update (modified in place)
            weights: Dict with keys: accuracy, price, latency (complexity ignored)
                     Values are integers 0-10
            use_case: Use case identifier for task-specific bonuses
        """
        logger.info(f"Recalculating balanced scores for use_case={use_case} with task bonuses")

        # Track bonuses applied for logging
        bonuses_applied = {}

        for config in configs:
            if config.scores:
                # Get base scores
                acc = config.scores.accuracy_score
                lat = config.scores.latency_score
                cost = config.scores.price_score

                # Get task-specific bonus for this model
                model_name = config.model_name or config.model_id or ""
                task_bonus = get_task_bonus(model_name, use_case) if use_case else 0

                # Track for logging
                if task_bonus > 0:
                    bonuses_applied[model_name] = task_bonus

                # BALANCED SCORE WITH TASK BONUS
                # (Accuracy + Task_Bonus) × 70% + Operational × 30%
                # Task bonus is capped to not exceed +10 points effect
                adjusted_acc = min(acc + task_bonus, 100)  # Cap at 100

                # Operational score = average of latency and cost
                operational_avg = (lat + cost) / 2

                # Balanced = 70% adjusted accuracy + 30% operational
                balanced = adjusted_acc * 0.7 + operational_avg * 0.3
                config.scores.balanced_score = round(balanced, 1)

        if bonuses_applied:
            logger.info(f"Task bonuses applied for {use_case}: {bonuses_applied}")

    def get_unique_configs_count(
        self, ranked_lists: dict[str, list[DeploymentRecommendation]]
    ) -> int:
        """
        Count unique configurations across all ranked lists.

        Since the same configuration may appear in multiple lists
        (e.g., best accuracy AND lowest cost), this counts unique ones.

        Args:
            ranked_lists: Dict of ranked lists

        Returns:
            Count of unique configurations
        """
        seen = set()
        for configs in ranked_lists.values():
            for config in configs:
                # Use model_id + gpu_config as unique key
                if config.gpu_config:
                    key = (
                        config.model_id,
                        config.gpu_config.gpu_type,
                        config.gpu_config.gpu_count,
                        config.gpu_config.tensor_parallel,
                        config.gpu_config.replicas,
                    )
                    seen.add(key)
        return len(seen)
