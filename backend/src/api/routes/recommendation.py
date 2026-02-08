"""Recommendation endpoints."""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.dependencies import get_deployment_generator, get_workflow
from src.shared.schemas import DeploymentRecommendation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["recommendation"])


class SimpleRecommendationRequest(BaseModel):
    """Simple request for deployment recommendation (UI compatibility)."""

    message: str


class BalancedWeights(BaseModel):
    """Weights for balanced score calculation (0-10 scale)."""

    accuracy: int = 4
    price: int = 4
    latency: int = 1
    complexity: int = 1


class RankedRecommendationFromSpecRequest(BaseModel):
    """Request for ranked recommendations from pre-built specification.

    Used when UI has already extracted/edited the specification and wants
    ranked recommendations without re-running intent extraction.
    """

    # Intent fields
    use_case: str
    user_count: int
    preferred_gpu_types: list[str] | None = None  # GPU filter list (empty/None = any GPU)

    # Traffic profile fields
    prompt_tokens: int
    output_tokens: int
    expected_qps: float

    # SLO target fields (required - explicit SLOs must always be provided)
    ttft_target_ms: int
    itl_target_ms: int
    e2e_target_ms: int
    percentile: str = "p95"  # "mean", "p90", "p95", "p99"

    # Ranking options
    min_accuracy: int | None = None
    max_cost: float | None = None
    include_near_miss: bool = True
    weights: BalancedWeights | None = None


@router.post("/recommend")
async def simple_recommend(request: SimpleRecommendationRequest):
    """
    Simplified recommendation endpoint for UI compatibility.

    Args:
        request: Simple request with message field

    Returns:
        Recommendation as JSON dict with auto-generated YAML
    """
    try:
        workflow = get_workflow()
        deployment_generator = get_deployment_generator()

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
                reasoning=str(e),
                alternative_options=None,
            )

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


@router.post("/ranked-recommend-from-spec")
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
        workflow = get_workflow()

        # Log complete request for debugging
        logger.info("=" * 60)
        logger.info("RANKED-RECOMMEND-FROM-SPEC REQUEST")
        logger.info("=" * 60)
        logger.info(f"  use_case: {request.use_case}")
        logger.info(f"  user_count: {request.user_count}")
        logger.info(f"  preferred_gpu_types: {request.preferred_gpu_types}")
        logger.info(f"  prompt_tokens: {request.prompt_tokens}")
        logger.info(f"  output_tokens: {request.output_tokens}")
        logger.info(f"  expected_qps: {request.expected_qps}")
        logger.info(f"  percentile: {request.percentile}")
        logger.info(f"  ttft_target_ms: {request.ttft_target_ms}ms")
        logger.info(f"  itl_target_ms: {request.itl_target_ms}ms")
        logger.info(f"  e2e_target_ms: {request.e2e_target_ms}ms")
        logger.info(f"  min_accuracy: {request.min_accuracy}")
        logger.info(f"  max_cost: {request.max_cost}")
        logger.info(f"  include_near_miss: {request.include_near_miss}")
        if request.weights:
            logger.info(
                f"  weights: accuracy={request.weights.accuracy}, price={request.weights.price}, "
                f"latency={request.weights.latency}, complexity={request.weights.complexity}"
            )
        else:
            logger.info("  weights: None (using defaults)")
        logger.info("=" * 60)

        # Build specifications dict for workflow
        specifications = {
            "intent": {
                "use_case": request.use_case,
                "user_count": request.user_count,
                "domain_specialization": ["general"],
                "preferred_gpu_types": request.preferred_gpu_types or [],
            },
            "traffic_profile": {
                "prompt_tokens": request.prompt_tokens,
                "output_tokens": request.output_tokens,
                "expected_qps": request.expected_qps,
            },
            "slo_targets": {
                "ttft_p95_target_ms": request.ttft_target_ms,
                "itl_p95_target_ms": request.itl_target_ms,
                "e2e_p95_target_ms": request.e2e_target_ms,
                "percentile": request.percentile,
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


@router.post("/test")
async def test_endpoint(message: str = "I need a chatbot for 1000 users"):
    """
    Quick test endpoint for validation.

    Args:
        message: Test message (optional)

    Returns:
        Simplified recommendation
    """
    try:
        workflow = get_workflow()
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
