"""FastAPI routes for the Compass API."""

import csv
import logging
import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..context_intent.schema import (
    ConversationMessage,
    DeploymentRecommendation,
    RankedRecommendationsResponse,
)
from ..deployment.cluster import KubernetesClusterManager, KubernetesDeploymentError
from ..deployment.generator import DeploymentGenerator
from ..deployment.validator import ValidationError, YAMLValidator
from ..knowledge_base.model_catalog import ModelCatalog
from ..knowledge_base.slo_templates import SLOTemplateRepository
from ..orchestration.workflow import RecommendationWorkflow

# Configure logging - check for DEBUG environment variable
debug_mode = os.getenv("COMPASS_DEBUG", "false").lower() == "true"
log_level = logging.DEBUG if debug_mode else logging.INFO
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info(f"Compass API starting with log level: {logging.getLevelName(log_level)}")

# Create FastAPI app
app = FastAPI(
    title="Compass API", description="API for LLM deployment recommendations", version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize workflow (singleton for POC)
workflow = RecommendationWorkflow()
model_catalog = ModelCatalog()
slo_repo = SLOTemplateRepository()
# Use simulator mode by default (no GPU required for development)
deployment_generator = DeploymentGenerator(simulator_mode=True)
yaml_validator = YAMLValidator()

# Initialize cluster manager (will be None if cluster not accessible)
try:
    cluster_manager = KubernetesClusterManager(namespace="default")
    logger.info("Kubernetes cluster manager initialized successfully")
except KubernetesDeploymentError as e:
    logger.warning(f"Kubernetes cluster not accessible: {e}")
    cluster_manager = None


# Request/Response models
class RecommendationRequest(BaseModel):
    """Request for deployment recommendation."""

    user_message: str
    conversation_history: list[ConversationMessage] | None = None


class SimpleRecommendationRequest(BaseModel):
    """Simple request for deployment recommendation (UI compatibility)."""

    message: str


class RecommendationResponse(BaseModel):
    """Response with deployment recommendation."""

    recommendation: DeploymentRecommendation
    success: bool = True
    message: str | None = None


class DeploymentRequest(BaseModel):
    """Request to generate deployment YAML files."""

    recommendation: DeploymentRecommendation
    namespace: str = "default"


class DeploymentResponse(BaseModel):
    """Response with generated deployment files."""

    deployment_id: str
    namespace: str
    files: dict
    success: bool = True
    message: str | None = None


class DeploymentStatusResponse(BaseModel):
    """Mock deployment status response."""

    deployment_id: str
    status: str
    slo_compliance: dict
    resource_utilization: dict
    cost_analysis: dict
    traffic_patterns: dict
    recommendations: list[str] | None = None


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "compass"}


# Main recommendation endpoint
@app.post("/api/v1/recommend", response_model=RecommendationResponse)
async def get_recommendation(request: RecommendationRequest):
    """
    Generate deployment recommendation from user message.

    Args:
        request: Recommendation request with user message

    Returns:
        Deployment recommendation

    Raises:
        HTTPException: If recommendation fails
    """
    try:
        # Log user request with clear delimiters
        logger.info("=" * 80)
        logger.info("[USER REQUEST] New recommendation request")
        logger.info(f"[USER MESSAGE] {request.user_message}")
        if request.conversation_history:
            logger.info(
                f"[CONVERSATION HISTORY] {len(request.conversation_history)} previous messages"
            )
        logger.info("=" * 80)

        # Always generate specification first (this cannot fail)
        specification = workflow.generate_specification(
            user_message=request.user_message, conversation_history=request.conversation_history
        )[0]

        # Try to find viable recommendations
        try:
            recommendation = workflow.generate_recommendation(
                user_message=request.user_message, conversation_history=request.conversation_history
            )

            # Validate recommendation
            is_valid = workflow.validate_recommendation(recommendation)
            if not is_valid:
                logger.warning("Generated recommendation failed validation")

            return RecommendationResponse(
                recommendation=recommendation,
                success=True,
                message="Recommendation generated successfully",
            )

        except ValueError as e:
            # No viable configurations found - return specification only
            logger.warning(f"No viable configurations found: {e}")

            # Create a partial recommendation with specification but no config
            from ..context_intent.schema import DeploymentRecommendation

            partial_recommendation = DeploymentRecommendation(
                intent=specification.intent,
                traffic_profile=specification.traffic_profile,
                slo_targets=specification.slo_targets,
                model_id=None,
                model_name=None,
                gpu_config=None,
                predicted_ttft_p95_ms=None,
                predicted_itl_p95_ms=None,
                predicted_e2e_p95_ms=None,
                predicted_throughput_qps=None,
                cost_per_hour_usd=None,
                cost_per_month_usd=None,
                meets_slo=False,
                reasoning=str(e),  # Include the detailed error message
                alternative_options=None,
            )

            return RecommendationResponse(
                recommendation=partial_recommendation,
                success=True,
                message="Specification generated, but no viable configurations found",
            )

    except Exception as e:
        logger.error(f"Failed to generate recommendation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate recommendation: {str(e)}"
        ) from e


# Get available models
@app.get("/api/v1/models")
async def list_models():
    """Get list of available models."""
    try:
        models = model_catalog.get_all_models()
        return {"models": [model.to_dict() for model in models], "count": len(models)}
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Get GPU types
@app.get("/api/v1/gpu-types")
async def list_gpu_types():
    """Get list of available GPU types."""
    try:
        gpu_types = model_catalog.get_all_gpu_types()
        return {"gpu_types": [gpu.to_dict() for gpu in gpu_types], "count": len(gpu_types)}
    except Exception as e:
        logger.error(f"Failed to list GPU types: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Get use case templates
@app.get("/api/v1/use-cases")
async def list_use_cases():
    """Get list of supported use cases with SLO templates."""
    try:
        templates = slo_repo.get_all_templates()
        return {
            "use_cases": {use_case: template.to_dict() for use_case, template in templates.items()},
            "count": len(templates),
        }
    except Exception as e:
        logger.error(f"Failed to list use cases: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Get benchmark data
@app.get("/api/v1/benchmarks")
async def get_benchmarks():
    """Get all 206 models benchmark data from opensource_all_benchmarks.csv."""
    try:
        # Get the CSV file path relative to the backend directory
        csv_path = Path(__file__).parent.parent.parent.parent / "data" / "benchmarks" / "models" / "opensource_all_benchmarks.csv"

        if not csv_path.exists():
            logger.error(f"Benchmark CSV not found at: {csv_path}")
            raise HTTPException(status_code=404, detail="Benchmark data file not found")

        # Read CSV using built-in csv module
        records = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter out rows with empty/missing Model Name
                if row.get('Model Name') and row['Model Name'].strip():
                    records.append(row)

        logger.info(f"Loaded {len(records)} benchmark records from CSV")

        return {
            "success": True,
            "count": len(records),
            "benchmarks": records
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load benchmarks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load benchmarks: {str(e)}") from e


# Get weighted scores for a use case
@app.get("/api/v1/weighted-scores/{use_case}")
async def get_weighted_scores(use_case: str):
    """Get use-case-specific weighted scores from CSV."""
    try:
        # Map use case to CSV filename
        use_case_to_file = {
            "chatbot_conversational": "opensource_chatbot_conversational.csv",
            "code_completion": "opensource_code_completion.csv",
            "code_generation_detailed": "opensource_code_generation_detailed.csv",
            "document_analysis_rag": "opensource_document_analysis_rag.csv",
            "summarization_short": "opensource_summarization_short.csv",
            "long_document_summarization": "opensource_long_document_summarization.csv",
            "translation": "opensource_translation.csv",
            "content_generation": "opensource_content_generation.csv",
            "research_legal_analysis": "opensource_research_legal_analysis.csv",
        }

        filename = use_case_to_file.get(use_case)
        if not filename:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid use case: {use_case}. Valid options: {list(use_case_to_file.keys())}"
            )

        # Get the CSV file path
        csv_path = Path(__file__).parent.parent.parent.parent / "data" / "business_context" / "use_case" / "weighted_scores" / filename

        if not csv_path.exists():
            logger.error(f"Weighted scores CSV not found at: {csv_path}")
            raise HTTPException(status_code=404, detail=f"Weighted scores file not found for use case: {use_case}")

        # Read CSV using built-in csv module
        records = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Include all rows (no filtering)
                records.append(row)

        logger.info(f"Loaded {len(records)} weighted score records for use case: {use_case}")

        return {
            "success": True,
            "use_case": use_case,
            "count": len(records),
            "scores": records
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load weighted scores: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load weighted scores: {str(e)}") from e


# Simplified recommendation endpoint for UI
@app.post("/api/recommend")
async def simple_recommend(request: SimpleRecommendationRequest):
    """
    Simplified recommendation endpoint for UI compatibility.

    Args:
        request: Simple request with message field

    Returns:
        Recommendation as JSON dict with auto-generated YAML
    """
    try:
        logger.info(f"Received UI recommendation request: {request.message[:100]}...")

        # Always generate specification first (this cannot fail)
        specification = workflow.generate_specification(
            user_message=request.message, conversation_history=None
        )[0]

        # Try to find viable recommendations
        try:
            recommendation = workflow.generate_recommendation(
                user_message=request.message, conversation_history=None
            )

            # Auto-generate deployment YAML
            try:
                yaml_result = deployment_generator.generate_all(
                    recommendation=recommendation, namespace="default"
                )
                deployment_id = yaml_result["deployment_id"]
                yaml_files = yaml_result["files"]
                logger.info(
                    f"Auto-generated YAML files for {deployment_id}: {list(yaml_files.keys())}"
                )
                yaml_generated = True
            except Exception as yaml_error:
                logger.warning(f"Failed to auto-generate YAML: {yaml_error}")
                deployment_id = None
                yaml_files = {}
                yaml_generated = False

            # Return recommendation as dict with YAML info
            result = recommendation.model_dump()
            result["deployment_id"] = deployment_id
            result["yaml_generated"] = yaml_generated
            result["yaml_files"] = list(yaml_files.keys()) if yaml_files else []

            return result

        except ValueError as e:
            # No viable configurations found - return specification only
            logger.warning(f"No viable configurations found: {e}")

            # Create a partial recommendation with specification but no config
            from ..context_intent.schema import DeploymentRecommendation

            partial_recommendation = DeploymentRecommendation(
                intent=specification.intent,
                traffic_profile=specification.traffic_profile,
                slo_targets=specification.slo_targets,
                model_id=None,
                model_name=None,
                gpu_config=None,
                predicted_ttft_p95_ms=None,
                predicted_itl_p95_ms=None,
                predicted_e2e_p95_ms=None,
                predicted_throughput_qps=None,
                cost_per_hour_usd=None,
                cost_per_month_usd=None,
                meets_slo=False,
                reasoning=str(e),  # Include the detailed error message
                alternative_options=None,
            )

            # No YAML for partial recommendations
            result = partial_recommendation.model_dump()
            result["deployment_id"] = None
            result["yaml_generated"] = False
            result["yaml_files"] = []

            return result

    except Exception as e:
        logger.error(f"Failed to generate recommendation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate recommendation: {str(e)}"
        ) from e


class BalancedWeights(BaseModel):
    """Weights for balanced score calculation (0-10 scale)."""

    accuracy: int = 4
    price: int = 4
    latency: int = 1
    complexity: int = 1


class RankedRecommendationRequest(BaseModel):
    """Request for ranked recommendations with multi-criteria scoring."""

    message: str
    min_accuracy: int | None = None
    max_cost: float | None = None
    include_near_miss: bool = True
    weights: BalancedWeights | None = None


class RankedRecommendationFromSpecRequest(BaseModel):
    """Request for ranked recommendations from pre-built specification.

    Used when UI has already extracted/edited the specification and wants
    ranked recommendations without re-running intent extraction.
    """

    # Intent fields
    use_case: str
    user_count: int
    latency_requirement: str = "high"  # "very_high", "high", "medium", "low"
    budget_constraint: str = "moderate"  # "strict", "moderate", "flexible"
    hardware_preference: str | None = None

    # Traffic profile fields
    prompt_tokens: int
    output_tokens: int
    expected_qps: float

    # SLO target fields
    ttft_p95_target_ms: int
    itl_p95_target_ms: int
    e2e_p95_target_ms: int

    # Ranking options
    min_accuracy: int | None = None
    max_cost: float | None = None
    include_near_miss: bool = True
    weights: BalancedWeights | None = None


@app.post("/api/ranked-recommend")
async def ranked_recommend(request: RankedRecommendationRequest):
    """
    Generate ranked recommendation lists with multi-criteria scoring.

    Returns 5 ranked views of deployment configurations:
    - best_accuracy: Top configs sorted by model capability
    - lowest_cost: Top configs sorted by price efficiency
    - lowest_latency: Top configs sorted by SLO headroom
    - simplest: Top configs sorted by deployment simplicity
    - balanced: Top configs sorted by weighted composite score

    Args:
        request: Request with message and optional filters

    Returns:
        RankedRecommendationsResponse with 5 ranked lists
    """
    try:
        logger.info(f"Received ranked recommendation request: {request.message[:100]}...")
        if request.min_accuracy:
            logger.info(f"  Filter: min_accuracy >= {request.min_accuracy}")
        if request.max_cost:
            logger.info(f"  Filter: max_cost <= ${request.max_cost}")
        logger.info(f"  Include near-miss: {request.include_near_miss}")
        if request.weights:
            logger.info(
                f"  Weights: A={request.weights.accuracy}, P={request.weights.price}, "
                f"L={request.weights.latency}, C={request.weights.complexity}"
            )

        # Convert weights to dict for workflow
        weights_dict = None
        if request.weights:
            weights_dict = {
                "accuracy": request.weights.accuracy,
                "price": request.weights.price,
                "latency": request.weights.latency,
                "complexity": request.weights.complexity,
            }

        # Generate ranked recommendations
        response = workflow.generate_ranked_recommendations(
            user_message=request.message,
            conversation_history=None,
            min_accuracy=request.min_accuracy,
            max_cost=request.max_cost,
            include_near_miss=request.include_near_miss,
            weights=weights_dict,
        )

        logger.info(
            f"Ranked recommendation complete: {response.total_configs_evaluated} configs, "
            f"{response.configs_after_filters} after filters"
        )

        return response.model_dump()

    except Exception as e:
        logger.error(f"Failed to generate ranked recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate ranked recommendations: {str(e)}"
        ) from e


@app.post("/api/ranked-recommend-from-spec")
async def ranked_recommend_from_spec(request: RankedRecommendationFromSpecRequest):
    """
    Generate ranked recommendations from pre-built specification.

    This endpoint is optimized for the UI workflow where specifications
    have already been extracted and potentially edited by the user.
    Skips intent extraction and uses provided specs directly.

    Returns 5 ranked views of deployment configurations:
    - balanced: Weighted composite score
    - best_accuracy: Top configs by model capability
    - lowest_cost: Top configs by price efficiency
    - lowest_latency: Top configs by SLO headroom
    - simplest: Top configs by deployment simplicity

    Args:
        request: Request with pre-built specification and optional filters

    Returns:
        RankedRecommendationsResponse with 5 ranked lists
    """
    try:
        logger.info(
            f"Received ranked recommendation from spec: use_case={request.use_case}, "
            f"user_count={request.user_count}, qps={request.expected_qps}"
        )
        logger.info(
            f"  SLO targets: TTFT={request.ttft_p95_target_ms}ms, "
            f"ITL={request.itl_p95_target_ms}ms, E2E={request.e2e_p95_target_ms}ms"
        )
        logger.info(
            f"  Token config: {request.prompt_tokens} -> {request.output_tokens}"
        )
        if request.weights:
            logger.info(
                f"  Weights: A={request.weights.accuracy}, P={request.weights.price}, "
                f"L={request.weights.latency}, C={request.weights.complexity}"
            )

        # Build specifications dict for workflow
        specifications = {
            "intent": {
                "use_case": request.use_case,
                "user_count": request.user_count,
                "latency_requirement": request.latency_requirement,
                "budget_constraint": request.budget_constraint,
                "throughput_priority": "medium",
                "domain_specialization": ["general"],
            },
            "traffic_profile": {
                "prompt_tokens": request.prompt_tokens,
                "output_tokens": request.output_tokens,
                "expected_qps": request.expected_qps,
            },
            "slo_targets": {
                "ttft_p95_target_ms": request.ttft_p95_target_ms,
                "itl_p95_target_ms": request.itl_p95_target_ms,
                "e2e_p95_target_ms": request.e2e_p95_target_ms,
            },
        }

        # Convert weights to dict for workflow
        weights_dict = None
        if request.weights:
            weights_dict = {
                "accuracy": request.weights.accuracy,
                "price": request.weights.price,
                "latency": request.weights.latency,
                "complexity": request.weights.complexity,
            }

        # Generate ranked recommendations from specs
        response = workflow.generate_ranked_recommendations_from_spec(
            specifications=specifications,
            min_accuracy=request.min_accuracy,
            max_cost=request.max_cost,
            include_near_miss=request.include_near_miss,
            weights=weights_dict,
        )

        logger.info(
            f"Ranked recommendation from spec complete: {response.total_configs_evaluated} configs, "
            f"{response.configs_after_filters} after filters"
        )

        return response.model_dump()

    except Exception as e:
        logger.error(f"Failed to generate ranked recommendations from spec: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate ranked recommendations: {str(e)}"
        ) from e


class ReRecommendationRequest(BaseModel):
    """Request for re-recommendation with edited specifications."""

    specifications: dict


@app.post("/api/re-recommend")
async def re_recommend(request: ReRecommendationRequest):
    """
    Re-generate recommendation with user-edited specifications.

    Args:
        request: Request with edited specifications (intent, traffic_profile, slo_targets)

    Returns:
        New recommendation as JSON dict with auto-generated YAML
    """
    try:
        logger.info("Received re-recommendation request with edited specifications")
        logger.debug(f"Edited specs: {request.specifications}")

        # Try to find viable recommendations with edited specs
        try:
            # Re-run recommendation workflow with edited specifications
            recommendation = workflow.generate_recommendation_from_specs(request.specifications)

            # Auto-generate deployment YAML
            try:
                yaml_result = deployment_generator.generate_all(
                    recommendation=recommendation, namespace="default"
                )
                deployment_id = yaml_result["deployment_id"]
                yaml_files = yaml_result["files"]
                logger.info(
                    f"Auto-generated YAML files for {deployment_id}: {list(yaml_files.keys())}"
                )
                yaml_generated = True
            except Exception as yaml_error:
                logger.warning(f"Failed to auto-generate YAML: {yaml_error}")
                deployment_id = None
                yaml_files = {}
                yaml_generated = False

            # Return recommendation as dict with YAML info
            result = recommendation.model_dump()
            result["deployment_id"] = deployment_id
            result["yaml_generated"] = yaml_generated
            result["yaml_files"] = list(yaml_files.keys()) if yaml_files else []

            logger.info(
                f"Re-recommendation complete: {recommendation.model_name} on "
                f"{recommendation.gpu_config.gpu_count}x {recommendation.gpu_config.gpu_type}"
            )

            return result

        except ValueError as e:
            # No viable configurations found - return partial recommendation
            logger.warning(f"No viable configurations found with edited specs: {e}")

            # Extract specifications from request
            from ..context_intent.schema import (
                DeploymentIntent,
                DeploymentRecommendation,
                SLOTargets,
                TrafficProfile,
            )

            # Infer experience_class if not provided
            intent_data = request.specifications["intent"].copy()
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

            intent = DeploymentIntent(**intent_data)
            traffic_profile = TrafficProfile(**request.specifications["traffic_profile"])
            slo_targets = SLOTargets(**request.specifications["slo_targets"])

            partial_recommendation = DeploymentRecommendation(
                intent=intent,
                traffic_profile=traffic_profile,
                slo_targets=slo_targets,
                model_id=None,
                model_name=None,
                gpu_config=None,
                predicted_ttft_p95_ms=None,
                predicted_itl_p95_ms=None,
                predicted_e2e_p95_ms=None,
                predicted_throughput_qps=None,
                cost_per_hour_usd=None,
                cost_per_month_usd=None,
                meets_slo=False,
                reasoning=str(e),
                alternative_options=None,
            )

            # No YAML for partial recommendations
            result = partial_recommendation.model_dump()
            result["deployment_id"] = None
            result["yaml_generated"] = False
            result["yaml_files"] = []

            return result

    except Exception as e:
        logger.error(f"Failed to re-generate recommendation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to re-generate recommendation: {str(e)}"
        ) from e


class RegenerateRequest(BaseModel):
    """Request for regenerating traffic profile and SLOs from requirements."""

    intent: dict


@app.post("/api/regenerate-and-recommend")
async def regenerate_and_recommend(request: RegenerateRequest):
    """
    Regenerate traffic profile and SLO targets from edited requirements, then recommend.

    This is the full workflow: requirements → profile/SLOs → recommendation

    Args:
        request: Request with edited intent/requirements

    Returns:
        New recommendation as JSON dict with auto-generated YAML
    """
    try:
        from ..context_intent.schema import DeploymentIntent

        logger.info("Received regenerate-and-recommend request with edited requirements")
        logger.debug(f"Edited intent: {request.intent}")

        # Infer experience_class if not provided
        intent_data = request.intent.copy()
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

        # Parse intent into schema
        intent = DeploymentIntent(**intent_data)

        # Generate traffic profile and SLO targets from intent
        from ..context_intent.traffic_profile import TrafficProfileGenerator

        traffic_generator = TrafficProfileGenerator()
        traffic_profile = traffic_generator.generate_profile(intent)
        slo_targets = traffic_generator.generate_slo_targets(intent)

        logger.info(
            f"Regenerated traffic profile: {traffic_profile.expected_qps} QPS, "
            f"{traffic_profile.prompt_tokens}→{traffic_profile.output_tokens} tokens"
        )
        logger.info(
            f"Regenerated SLO targets (p95): TTFT={slo_targets.ttft_p95_target_ms}ms, "
            f"ITL={slo_targets.itl_p95_target_ms}ms, E2E={slo_targets.e2e_p95_target_ms}ms"
        )

        # Build specifications dict
        specifications = {
            "intent": intent.model_dump(),
            "traffic_profile": traffic_profile.model_dump(),
            "slo_targets": slo_targets.model_dump(),
        }

        # Try to find viable recommendations with regenerated specifications
        try:
            # Re-run recommendation workflow with regenerated specifications
            recommendation = workflow.generate_recommendation_from_specs(specifications)

            # Auto-generate deployment YAML
            try:
                yaml_result = deployment_generator.generate_all(
                    recommendation=recommendation, namespace="default"
                )
                deployment_id = yaml_result["deployment_id"]
                yaml_files = yaml_result["files"]
                logger.info(
                    f"Auto-generated YAML files for {deployment_id}: {list(yaml_files.keys())}"
                )
                yaml_generated = True
            except Exception as yaml_error:
                logger.warning(f"Failed to auto-generate YAML: {yaml_error}")
                deployment_id = None
                yaml_files = {}
                yaml_generated = False

            # Return recommendation as dict with YAML info
            result = recommendation.model_dump()
            result["deployment_id"] = deployment_id
            result["yaml_generated"] = yaml_generated
            result["yaml_files"] = list(yaml_files.keys()) if yaml_files else []

            logger.info(
                f"Regenerate-and-recommend complete: {recommendation.model_name} on "
                f"{recommendation.gpu_config.gpu_count}x {recommendation.gpu_config.gpu_type}"
            )

            return result

        except ValueError as e:
            # No viable configurations found - return partial recommendation
            logger.warning(f"No viable configurations found with regenerated specs: {e}")

            from ..context_intent.schema import DeploymentRecommendation

            partial_recommendation = DeploymentRecommendation(
                intent=intent,
                traffic_profile=traffic_profile,
                slo_targets=slo_targets,
                model_id=None,
                model_name=None,
                gpu_config=None,
                predicted_ttft_p95_ms=None,
                predicted_itl_p95_ms=None,
                predicted_e2e_p95_ms=None,
                predicted_throughput_qps=None,
                cost_per_hour_usd=None,
                cost_per_month_usd=None,
                meets_slo=False,
                reasoning=str(e),
                alternative_options=None,
            )

            # No YAML for partial recommendations
            result = partial_recommendation.model_dump()
            result["deployment_id"] = None
            result["yaml_generated"] = False
            result["yaml_files"] = []

            return result

    except Exception as e:
        logger.error(f"Failed to regenerate and recommend: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to regenerate and recommend: {str(e)}"
        ) from e


# Simple test endpoint for quick validation
@app.post("/api/v1/test")
async def test_endpoint(message: str = "I need a chatbot for 1000 users"):
    """
    Quick test endpoint for validation.

    Args:
        message: Test message (optional)

    Returns:
        Simplified recommendation
    """
    try:
        recommendation = workflow.generate_recommendation(message)

        return {
            "success": True,
            "model": recommendation.model_name,
            "gpu_config": f"{recommendation.gpu_config.gpu_count}x {recommendation.gpu_config.gpu_type}",
            "cost_per_month": f"${recommendation.cost_per_month_usd:.2f}",
            "meets_slo": recommendation.meets_slo,
            "reasoning": recommendation.reasoning,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# Deployment endpoints
@app.post("/api/deploy", response_model=DeploymentResponse)
async def deploy_model(request: DeploymentRequest):
    """
    Generate deployment YAML files from recommendation.

    Args:
        request: Deployment request with recommendation

    Returns:
        Deployment response with file paths

    Raises:
        HTTPException: If deployment generation fails
    """
    try:
        logger.info(f"Generating deployment for model: {request.recommendation.model_name}")

        # Generate all YAML files
        result = deployment_generator.generate_all(
            recommendation=request.recommendation, namespace=request.namespace
        )

        # Validate generated files
        try:
            yaml_validator.validate_all(result["files"])
            logger.info(f"All YAML files validated for deployment: {result['deployment_id']}")
        except ValidationError as e:
            logger.error(f"YAML validation failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Generated YAML validation failed: {str(e)}"
            ) from e

        return DeploymentResponse(
            deployment_id=result["deployment_id"],
            namespace=result["namespace"],
            files=result["files"],
            success=True,
            message=f"Deployment files generated successfully for {result['deployment_id']}",
        )

    except Exception as e:
        logger.error(f"Failed to generate deployment: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate deployment: {str(e)}"
        ) from e


@app.get("/api/deployments/{deployment_id}/status", response_model=DeploymentStatusResponse)
async def get_deployment_status(deployment_id: str):
    """
    Get mock deployment status for observability demonstration.

    This endpoint returns simulated observability data to demonstrate
    Component 9 (Inference Observability & SLO Monitoring).

    Args:
        deployment_id: Deployment identifier

    Returns:
        Mock deployment status with observability metrics

    Raises:
        HTTPException: If deployment not found
    """
    try:
        logger.info(f"Fetching mock status for deployment: {deployment_id}")

        # Mock observability data (in production, this would query Prometheus/Grafana)
        import random

        # Simulate SLO compliance with some variance
        base_ttft = 185
        base_tpot = 48
        base_e2e = 1850

        mock_status = DeploymentStatusResponse(
            deployment_id=deployment_id,
            status="running",
            slo_compliance={
                "ttft_p90_ms": base_ttft + random.randint(-10, 15),
                "ttft_target_ms": 200,
                "ttft_compliant": True,
                "tpot_p90_ms": base_tpot + random.randint(-3, 5),
                "tpot_target_ms": 50,
                "tpot_compliant": True,
                "e2e_p90_ms": base_e2e + random.randint(-50, 100),
                "e2e_target_ms": 2000,
                "e2e_compliant": True,
                "throughput_qps": 122 + random.randint(-5, 10),
                "throughput_target_qps": 100,
                "throughput_compliant": True,
                "uptime_pct": 99.94 + random.uniform(-0.05, 0.05),
                "uptime_target_pct": 99.9,
                "uptime_compliant": True,
            },
            resource_utilization={
                "gpu_utilization_pct": 78 + random.randint(-5, 10),
                "gpu_memory_used_gb": 14.2 + random.uniform(-1, 2),
                "gpu_memory_total_gb": 24,
                "avg_batch_size": 18 + random.randint(-3, 5),
                "queue_depth": random.randint(0, 5),
                "token_throughput_per_gpu": 3500 + random.randint(-200, 300),
            },
            cost_analysis={
                "actual_cost_per_hour_usd": 0.95 + random.uniform(-0.05, 0.1),
                "predicted_cost_per_hour_usd": 1.00,
                "actual_cost_per_month_usd": 812 + random.randint(-30, 50),
                "predicted_cost_per_month_usd": 800,
                "cost_per_1k_tokens_usd": 0.042 + random.uniform(-0.002, 0.005),
                "predicted_cost_per_1k_tokens_usd": 0.040,
            },
            traffic_patterns={
                "avg_prompt_tokens": 165 + random.randint(-10, 20),
                "predicted_prompt_tokens": 150,
                "avg_generation_tokens": 220 + random.randint(-15, 25),
                "predicted_generation_tokens": 200,
                "peak_qps": 95 + random.randint(-5, 10),
                "predicted_peak_qps": 100,
                "requests_last_hour": 7200 + random.randint(-500, 800),
                "requests_last_24h": 172800 + random.randint(-5000, 10000),
            },
            recommendations=[
                "GPU utilization is 78%, below the 80% efficiency target. Consider downsizing to reduce cost.",
                "Actual traffic is 10% higher than predicted. Monitor for potential capacity constraints.",
                "All SLO targets are being met with headroom. Configuration is performing well.",
            ],
        )

        return mock_status

    except Exception as e:
        logger.error(f"Failed to get deployment status: {e}")
        raise HTTPException(status_code=404, detail=f"Deployment not found: {deployment_id}") from e


@app.post("/api/deploy-to-cluster")
async def deploy_to_cluster(request: DeploymentRequest):
    """
    Deploy model to Kubernetes cluster.

    This endpoint generates YAML files AND applies them to the cluster.

    Args:
        request: Deployment request with recommendation and namespace

    Returns:
        Deployment result with status

    Raises:
        HTTPException: If cluster not accessible or deployment fails
    """
    # Try to initialize cluster manager if it wasn't available at startup
    manager = cluster_manager
    if manager is None:
        try:
            manager = KubernetesClusterManager(namespace=request.namespace)
        except KubernetesDeploymentError as e:
            raise HTTPException(
                status_code=503, detail=f"Kubernetes cluster not accessible: {str(e)}"
            ) from e

    try:
        logger.info(f"Deploying model to cluster: {request.recommendation.model_name}")

        # Step 1: Generate YAML files
        result = deployment_generator.generate_all(
            recommendation=request.recommendation, namespace=request.namespace
        )

        deployment_id = result["deployment_id"]
        files = result["files"]

        # Step 2: Validate generated files
        try:
            yaml_validator.validate_all(files)
            logger.info(f"YAML validation passed for: {deployment_id}")
        except ValidationError as e:
            logger.error(f"YAML validation failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Generated YAML validation failed: {str(e)}"
            ) from e

        # Step 3: Deploy to cluster
        # Note: generator creates keys like "inferenceservice", "vllm-config", "autoscaling", etc.
        # Skip ServiceMonitor for now (requires Prometheus Operator)
        yaml_file_paths = [files["inferenceservice"], files["autoscaling"]]

        deployment_result = manager.deploy_all(yaml_file_paths)

        if not deployment_result["success"]:
            logger.error(f"Deployment failed: {deployment_result['errors']}")
            raise HTTPException(
                status_code=500, detail=f"Deployment failed: {deployment_result['errors']}"
            )

        logger.info(f"Successfully deployed {deployment_id} to cluster")

        return {
            "success": True,
            "deployment_id": deployment_id,
            "namespace": request.namespace,
            "files": files,
            "deployment_result": deployment_result,
            "message": f"Successfully deployed {deployment_id} to Kubernetes cluster",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to deploy to cluster: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to deploy to cluster: {str(e)}") from e


@app.get("/api/cluster-status")
async def get_cluster_status():
    """
    Get Kubernetes cluster status.

    Returns:
        Cluster accessibility and basic info
    """
    # Always try to check cluster status dynamically, don't rely on cached cluster_manager
    try:
        temp_manager = KubernetesClusterManager(namespace="default")
        # List existing deployments
        deployments = temp_manager.list_inferenceservices()

        return {
            "accessible": True,
            "namespace": temp_manager.namespace,
            "inference_services": deployments,
            "count": len(deployments),
            "message": "Cluster accessible",
        }
    except Exception as e:
        logger.error(f"Failed to query cluster status: {e}")
        return {"accessible": False, "error": str(e)}


@app.get("/api/deployments/{deployment_id}/k8s-status")
async def get_k8s_deployment_status(deployment_id: str):
    """
    Get actual Kubernetes deployment status (not mock data).

    Args:
        deployment_id: InferenceService name

    Returns:
        Real deployment status from cluster

    Raises:
        HTTPException: If cluster not accessible
    """
    # Try to initialize cluster manager if it wasn't available at startup
    manager = cluster_manager
    if manager is None:
        try:
            manager = KubernetesClusterManager(namespace="default")
        except KubernetesDeploymentError as e:
            raise HTTPException(
                status_code=503, detail=f"Kubernetes cluster not accessible: {str(e)}"
            ) from e

    try:
        # Get InferenceService status
        isvc_status = manager.get_inferenceservice_status(deployment_id)

        # Get associated pods
        pods = manager.get_deployment_pods(deployment_id)

        return {
            "deployment_id": deployment_id,
            "inferenceservice": isvc_status,
            "pods": pods,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get K8s deployment status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get deployment status: {str(e)}"
        ) from e


@app.get("/api/deployments/{deployment_id}/yaml")
async def get_deployment_yaml(deployment_id: str):
    """
    Retrieve generated YAML files for a deployment.

    Args:
        deployment_id: Deployment identifier

    Returns:
        Dictionary with YAML file contents

    Raises:
        HTTPException: If YAML files not found
    """
    try:
        # Get the output directory from deployment generator

        output_dir = deployment_generator.output_dir

        # Find all YAML files for this deployment
        yaml_files = {}
        for file_path in output_dir.glob(f"{deployment_id}*.yaml"):
            with open(file_path) as f:
                yaml_files[file_path.name] = f.read()

        if not yaml_files:
            raise HTTPException(
                status_code=404, detail=f"No YAML files found for deployment {deployment_id}"
            )

        return {"deployment_id": deployment_id, "files": yaml_files, "count": len(yaml_files)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve YAML files: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve YAML files: {str(e)}"
        ) from e


@app.delete("/api/deployments/{deployment_id}")
async def delete_deployment(deployment_id: str):
    """
    Delete a deployment from the cluster.

    Args:
        deployment_id: InferenceService name to delete

    Returns:
        Deletion result

    Raises:
        HTTPException: If cluster not accessible or deletion fails
    """
    # Try to initialize cluster manager if it wasn't available at startup
    manager = cluster_manager
    if manager is None:
        try:
            manager = KubernetesClusterManager(namespace="default")
        except KubernetesDeploymentError as e:
            raise HTTPException(
                status_code=503, detail=f"Kubernetes cluster not accessible: {str(e)}"
            ) from e

    try:
        result = manager.delete_inferenceservice(deployment_id)

        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete deployment: {result.get('error', 'Unknown error')}",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete deployment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete deployment: {str(e)}") from e


@app.get("/api/deployments")
async def list_all_deployments():
    """
    List all InferenceServices in the cluster with their detailed status.

    Returns:
        List of deployments with status information

    Raises:
        HTTPException: If cluster not accessible
    """
    # Try to initialize cluster manager if it wasn't available at startup
    manager = cluster_manager
    if manager is None:
        try:
            manager = KubernetesClusterManager(namespace="default")
        except KubernetesDeploymentError as e:
            raise HTTPException(
                status_code=503, detail=f"Kubernetes cluster not accessible: {str(e)}"
            ) from e

    try:
        # Get list of all InferenceService names
        deployment_ids = manager.list_inferenceservices()

        # Get detailed status for each
        deployments = []
        for deployment_id in deployment_ids:
            status = manager.get_inferenceservice_status(deployment_id)
            pods = manager.get_deployment_pods(deployment_id)

            deployments.append({"deployment_id": deployment_id, "status": status, "pods": pods})

        return {
            "success": True,
            "count": len(deployments),
            "deployments": deployments,
            "namespace": manager.namespace,
        }

    except Exception as e:
        logger.error(f"Failed to list deployments: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list deployments: {str(e)}") from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
