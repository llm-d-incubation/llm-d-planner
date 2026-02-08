"""Workflow orchestration for end-to-end recommendation flow."""

import logging

from src.intent_extraction import IntentExtractor
from src.llm.ollama_client import OllamaClient
from src.recommendation.analyzer import Analyzer
from src.recommendation.config_finder import ConfigFinder
from src.shared.schemas import (
    ConversationMessage,
    DeploymentRecommendation,
    RankedRecommendationsResponse,
)
from src.specification import TrafficProfileGenerator

logger = logging.getLogger(__name__)


class RecommendationWorkflow:
    """Orchestrate the full recommendation workflow."""

    def __init__(
        self,
        llm_client: OllamaClient | None = None,
        intent_extractor: IntentExtractor | None = None,
        traffic_generator: TrafficProfileGenerator | None = None,
        config_finder: ConfigFinder | None = None,
    ):
        """
        Initialize workflow orchestrator.

        Args:
            llm_client: Ollama client (creates default if not provided)
            intent_extractor: Intent extractor
            traffic_generator: Traffic profile generator
            config_finder: Configuration finder
        """
        self.llm_client = llm_client or OllamaClient()
        self.intent_extractor = intent_extractor or IntentExtractor(self.llm_client)
        self.traffic_generator = traffic_generator or TrafficProfileGenerator()
        self.config_finder = config_finder or ConfigFinder()

    def generate_specification(
        self, user_message: str, conversation_history: list[ConversationMessage] | None = None
    ) -> tuple:
        """
        Generate deployment specification from user message.

        This extracts intent and generates traffic/SLO specs from conversation.
        Model recommendation happens later in generate_recommendation_from_specs.

        Returns:
            Tuple of (DeploymentSpecification, intent, traffic_profile, slo_targets)
        """
        from src.shared.schemas import DeploymentSpecification

        logger.info("Step 1: Extracting deployment intent")
        intent = self.intent_extractor.extract_intent(user_message, conversation_history)
        intent = self.intent_extractor.infer_missing_fields(intent)
        logger.info(f"Intent extracted: {intent.use_case}, {intent.user_count} users")

        logger.info("Step 2: Generating traffic profile and SLO targets")
        traffic_profile = self.traffic_generator.generate_profile(intent)
        slo_targets = self.traffic_generator.generate_slo_targets(intent)
        logger.info(
            f"Traffic profile: ({traffic_profile.prompt_tokens}→{traffic_profile.output_tokens}), "
            f"{traffic_profile.expected_qps} QPS"
        )
        logger.info(
            f"SLO targets (p95): TTFT={slo_targets.ttft_p95_target_ms}ms, "
            f"ITL={slo_targets.itl_p95_target_ms}ms, E2E={slo_targets.e2e_p95_target_ms}ms"
        )

        specification = DeploymentSpecification(
            intent=intent,
            traffic_profile=traffic_profile,
            slo_targets=slo_targets,
        )

        return specification, intent, traffic_profile, slo_targets

    def generate_recommendation(
        self, user_message: str, conversation_history: list[ConversationMessage] | None = None
    ) -> DeploymentRecommendation:
        """
        Generate deployment recommendation from user message.

        This is the main workflow that orchestrates all components:
        1. Extract intent from conversation
        2. Generate traffic profile and SLO targets
        3. Delegate to generate_recommendation_from_specs for recommendation

        Args:
            user_message: User's deployment request
            conversation_history: Optional conversation context

        Returns:
            DeploymentRecommendation

        Raises:
            ValueError: If recommendation cannot be generated
        """
        logger.info("Starting recommendation workflow")

        # Generate specification from conversation
        specification, intent, traffic_profile, slo_targets = self.generate_specification(
            user_message, conversation_history
        )

        # Convert to dict format and delegate to single recommendation path
        specs = {
            "intent": intent.model_dump(),
            "traffic_profile": traffic_profile.model_dump(),
            "slo_targets": slo_targets.model_dump(),
        }
        return self.generate_recommendation_from_specs(specs)

    def generate_recommendation_from_specs(self, specifications: dict) -> DeploymentRecommendation:
        """
        Generate recommendation from specifications.

        This is the single entry point for recommendation generation.
        Used by both:
        - generate_recommendation() - after extracting specs from conversation
        - Direct calls with user-edited specifications

        Args:
            specifications: Dict with keys: intent, traffic_profile, slo_targets

        Returns:
            DeploymentRecommendation

        Raises:
            ValueError: If recommendation cannot be generated
        """
        from src.shared.schemas import DeploymentIntent, SLOTargets, TrafficProfile

        logger.info("Generating recommendation from specifications")

        # Infer experience_class if not provided in intent
        intent_data = specifications["intent"].copy()
        if "experience_class" not in intent_data or not intent_data.get("experience_class"):
            # Use the same inference logic as the extractor
            use_case = intent_data.get("use_case", "")
            if use_case == "code_completion":
                intent_data["experience_class"] = "instant"
            elif use_case in [
                "chatbot_conversational",
                "code_generation_detailed",
                "translation",
                "content_generation",
                "summarization_short",
            ]:
                intent_data["experience_class"] = "conversational"
            elif use_case == "document_analysis_rag":
                intent_data["experience_class"] = "interactive"
            elif use_case == "long_document_summarization":
                intent_data["experience_class"] = "deferred"
            elif use_case == "research_legal_analysis":
                intent_data["experience_class"] = "batch"
            else:
                intent_data["experience_class"] = "conversational"  # Default

        # Parse specifications into proper schema objects
        intent = DeploymentIntent(**intent_data)
        traffic_profile = TrafficProfile(**specifications["traffic_profile"])
        slo_targets = SLOTargets(**specifications["slo_targets"])

        logger.info(
            f"Specs: {intent.use_case}, {intent.user_count} users, "
            f"{traffic_profile.expected_qps} QPS, "
            f"TTFT target={slo_targets.ttft_p95_target_ms}ms (p95)"
        )

        # Generate all viable configurations with full scoring
        # No model pre-filtering - all benchmark configs meeting SLO are scored
        logger.info("Generating all viable configurations")
        all_configs = self.config_finder.plan_all_capacities(
            traffic_profile=traffic_profile,
            slo_targets=slo_targets,
            intent=intent,
            include_near_miss=False,  # Strict SLO for best recommendation
        )

        if not all_configs:
            # Build helpful error message with context
            error_msg = (
                f"No viable deployment configurations found meeting SLO targets.\n\n"
                f"**Requirements:**\n"
                f"- Use case: {intent.use_case} ({intent.experience_class} experience)\n"
                f"- Scale: {intent.user_count:,} users\n\n"
                f"**Traffic profile:**\n"
                f"- {traffic_profile.prompt_tokens} prompt tokens -> {traffic_profile.output_tokens} output tokens\n"
                f"- Expected load: {traffic_profile.expected_qps} queries/second\n\n"
                f"**SLO targets (p95):**\n"
                f"- TTFT <= {slo_targets.ttft_p95_target_ms}ms\n"
                f"- ITL <= {slo_targets.itl_p95_target_ms}ms\n"
                f"- E2E <= {slo_targets.e2e_p95_target_ms}ms\n\n"
                f"No configurations can meet the SLO targets with available hardware configurations. "
                f"Try relaxing SLO targets or considering a different use case."
            )
            raise ValueError(error_msg)

        # Sort by balanced score and pick best
        all_configs.sort(key=lambda x: x.scores.balanced_score if x.scores else 0, reverse=True)
        best_recommendation = all_configs[0]

        logger.info(
            f"Selected: {best_recommendation.model_name} on "
            f"{best_recommendation.gpu_config.gpu_count}x {best_recommendation.gpu_config.gpu_type} "
            f"(balanced score: {best_recommendation.scores.balanced_score if best_recommendation.scores else 0:.1f})"
        )

        # Add top 3 alternatives
        if len(all_configs) > 1:
            best_recommendation.alternative_options = [
                rec.to_alternative_dict() for rec in all_configs[1:4]
            ]
            logger.info(f"Added {len(best_recommendation.alternative_options)} alternative options")

        return best_recommendation

    def generate_ranked_recommendations(
        self,
        user_message: str,
        conversation_history: list[ConversationMessage] | None = None,
        min_accuracy: int | None = None,
        max_cost: float | None = None,
        include_near_miss: bool = True,
        weights: dict[str, int] | None = None,
    ) -> RankedRecommendationsResponse:
        """
        Generate ranked recommendation lists from user message.

        This is the enhanced workflow that returns multiple ranked views
        instead of a single best recommendation. Useful for exploring
        trade-offs between accuracy, cost, latency, and complexity.

        Args:
            user_message: User's deployment request
            conversation_history: Optional conversation context
            min_accuracy: Minimum accuracy score filter (0-100)
            max_cost: Maximum monthly cost filter (USD)
            include_near_miss: Whether to include near-SLO configurations
            weights: Optional custom weights for balanced score (0-10 scale)
                     Keys: accuracy, price, latency, complexity

        Returns:
            RankedRecommendationsResponse with 5 ranked lists
        """
        logger.info("Starting ranked recommendation workflow")

        # Generate specification from conversation
        specification, intent, traffic_profile, slo_targets = self.generate_specification(
            user_message, conversation_history
        )

        # Get ALL configurations with scores
        # No model pre-filtering - all benchmark configs meeting SLO are scored
        logger.info("Planning capacity for all model/GPU combinations")
        all_configs = self.config_finder.plan_all_capacities(
            traffic_profile=traffic_profile,
            slo_targets=slo_targets,
            intent=intent,
            include_near_miss=include_near_miss,
        )

        if not all_configs:
            logger.warning("No viable configurations found")
            return RankedRecommendationsResponse(
                min_accuracy_threshold=min_accuracy,
                max_cost_ceiling=max_cost,
                include_near_miss=include_near_miss,
                specification=specification,
                total_configs_evaluated=0,
                configs_after_filters=0,
            )

        # Generate ranked lists (top 10 solutions per criterion)
        # Pass use_case for task-specific bonuses on Balanced card
        analyzer = Analyzer()
        ranked_lists = analyzer.generate_ranked_lists(
            configurations=all_configs,
            min_accuracy=min_accuracy,
            max_cost=max_cost,
            top_n=5,  # Top 5 accuracy models only
            weights=weights,
            use_case=intent.use_case,  # Task bonuses for Balanced
        )

        # Count configs after filtering
        configs_after_filters = analyzer.get_unique_configs_count(ranked_lists)

        logger.info(
            f"Generated ranked recommendations: {len(all_configs)} total configs, "
            f"{configs_after_filters} after filters"
        )

        return RankedRecommendationsResponse(
            min_accuracy_threshold=min_accuracy,
            max_cost_ceiling=max_cost,
            include_near_miss=include_near_miss,
            specification=specification,
            best_accuracy=ranked_lists["best_accuracy"],
            lowest_cost=ranked_lists["lowest_cost"],
            lowest_latency=ranked_lists["lowest_latency"],
            simplest=ranked_lists["simplest"],
            balanced=ranked_lists["balanced"],
            total_configs_evaluated=len(all_configs),
            configs_after_filters=configs_after_filters,
        )

    def generate_ranked_recommendations_from_spec(
        self,
        specifications: dict,
        min_accuracy: int | None = None,
        max_cost: float | None = None,
        include_near_miss: bool = True,
        weights: dict[str, int] | None = None,
    ) -> RankedRecommendationsResponse:
        """
        Generate ranked recommendation lists from pre-built specifications.

        This bypasses intent extraction and uses the provided specs directly.
        Used when UI has already extracted and potentially edited the specs.

        Args:
            specifications: Dict with keys: intent, traffic_profile, slo_targets
            min_accuracy: Minimum accuracy score filter (0-100)
            max_cost: Maximum monthly cost filter (USD)
            include_near_miss: Whether to include near-SLO configurations
            weights: Optional custom weights for balanced score (0-10 scale)
                     Keys: accuracy, price, latency, complexity

        Returns:
            RankedRecommendationsResponse with 5 ranked lists
        """
        from src.shared.schemas import (
            DeploymentIntent,
            DeploymentSpecification,
            SLOTargets,
            TrafficProfile,
        )

        logger.info("Starting ranked recommendation workflow from specifications")

        # Infer experience_class if not provided in intent
        intent_data = specifications["intent"].copy()
        if "experience_class" not in intent_data or not intent_data.get("experience_class"):
            use_case = intent_data.get("use_case", "")
            if use_case == "code_completion":
                intent_data["experience_class"] = "instant"
            elif use_case in [
                "chatbot_conversational",
                "code_generation_detailed",
                "translation",
                "content_generation",
                "summarization_short",
            ]:
                intent_data["experience_class"] = "conversational"
            elif use_case == "document_analysis_rag":
                intent_data["experience_class"] = "interactive"
            elif use_case == "long_document_summarization":
                intent_data["experience_class"] = "deferred"
            elif use_case == "research_legal_analysis":
                intent_data["experience_class"] = "batch"
            else:
                intent_data["experience_class"] = "conversational"

        # Parse specifications into schema objects
        intent = DeploymentIntent(**intent_data)
        traffic_profile = TrafficProfile(**specifications["traffic_profile"])
        slo_targets = SLOTargets(**specifications["slo_targets"])

        specification = DeploymentSpecification(
            intent=intent,
            traffic_profile=traffic_profile,
            slo_targets=slo_targets,
        )

        logger.info(
            f"Specs: {intent.use_case}, {intent.user_count} users, "
            f"{traffic_profile.expected_qps} QPS, "
            f"TTFT target={slo_targets.ttft_p95_target_ms}ms (p95)"
        )

        # Get ALL configurations with scores
        logger.info("Planning capacity for all model/GPU combinations")
        logger.info(f"Using weights for balanced scoring: {weights}")
        all_configs = self.config_finder.plan_all_capacities(
            traffic_profile=traffic_profile,
            slo_targets=slo_targets,
            intent=intent,
            include_near_miss=include_near_miss,
            weights=weights,
        )

        if not all_configs:
            logger.warning("No viable configurations found")
            return RankedRecommendationsResponse(
                min_accuracy_threshold=min_accuracy,
                max_cost_ceiling=max_cost,
                include_near_miss=include_near_miss,
                specification=specification,
                total_configs_evaluated=0,
                configs_after_filters=0,
            )

        # Generate ranked lists (top 10 solutions per criterion)
        # Pass use_case for task-specific bonuses on Balanced card
        analyzer = Analyzer()
        ranked_lists = analyzer.generate_ranked_lists(
            configurations=all_configs,
            min_accuracy=min_accuracy,
            max_cost=max_cost,
            top_n=10,  # Top 10 accuracy models only
            weights=weights,
            use_case=intent.use_case,  # Task bonuses for Balanced
        )

        # Count configs after filtering
        configs_after_filters = analyzer.get_unique_configs_count(ranked_lists)

        logger.info(
            f"Generated ranked recommendations from spec: {len(all_configs)} total configs, "
            f"{configs_after_filters} after filters"
        )

        return RankedRecommendationsResponse(
            min_accuracy_threshold=min_accuracy,
            max_cost_ceiling=max_cost,
            include_near_miss=include_near_miss,
            specification=specification,
            best_accuracy=ranked_lists["best_accuracy"],
            lowest_cost=ranked_lists["lowest_cost"],
            lowest_latency=ranked_lists["lowest_latency"],
            simplest=ranked_lists["simplest"],
            balanced=ranked_lists["balanced"],
            total_configs_evaluated=len(all_configs),
            configs_after_filters=configs_after_filters,
        )
