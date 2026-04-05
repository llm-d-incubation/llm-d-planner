"""Test script for Sprint 4: YAML generation and deployment functionality.

This script tests the complete workflow:
1. Generate a recommendation
2. Generate YAML deployment files
3. Validate the generated YAMLs
4. Fetch mock deployment status
"""

import logging

from planner.configuration.generator import DeploymentGenerator
from planner.configuration.validator import YAMLValidator
from planner.shared.schemas import (
    DeploymentIntent,
    DeploymentRecommendation,
    GPUConfig,
    SLOTargets,
    TrafficProfile,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_recommendation() -> DeploymentRecommendation:
    """Create a test recommendation for a chatbot deployment."""

    intent = DeploymentIntent(
        use_case="chatbot_conversational",
        experience_class="conversational",
        user_count=5000,
        domain_specialization=["general"],
        additional_context=None,
    )

    traffic_profile = TrafficProfile(prompt_tokens=512, output_tokens=256, expected_qps=50.0)

    slo_targets = SLOTargets(ttft_p95_target_ms=200, itl_p95_target_ms=50, e2e_p95_target_ms=2000)

    gpu_config = GPUConfig(gpu_type="A100-80", gpu_count=2, tensor_parallel=2, replicas=1)

    recommendation = DeploymentRecommendation(
        intent=intent,
        traffic_profile=traffic_profile,
        slo_targets=slo_targets,
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        model_name="Llama 3.1 8B Instruct",
        model_uri="meta-llama/Llama-3.1-8B-Instruct",
        gpu_config=gpu_config,
        predicted_ttft_p95_ms=185,
        predicted_itl_p95_ms=48,
        predicted_e2e_p95_ms=1850,
        predicted_throughput_qps=122.0,
        cost_per_hour_usd=9.00,
        cost_per_month_usd=6570.0,
        meets_slo=True,
        reasoning="Llama 3.1 8B Instruct provides excellent latency for chatbot use cases. "
        "2x A100-80 GPUs in tensor parallel configuration meets all SLO targets "
        "with headroom for traffic spikes. Cost-effective for 5000 concurrent users.",
        alternative_options=None,
    )

    return recommendation


def test_yaml_generation():
    """Test YAML generation and validation."""

    logger.info("=" * 80)
    logger.info("SPRINT 4 TEST: YAML Generation and Deployment")
    logger.info("=" * 80)

    # Step 1: Create test recommendation
    logger.info("\n[1/4] Creating test recommendation...")
    recommendation = create_test_recommendation()
    assert recommendation.gpu_config is not None
    logger.info(
        f"✓ Recommendation created: {recommendation.model_name} on "
        f"{recommendation.gpu_config.gpu_count}x {recommendation.gpu_config.gpu_type}"
    )

    # Step 2: Generate YAML files
    logger.info("\n[2/4] Generating deployment YAML files...")
    generator = DeploymentGenerator()

    result = generator.generate_all(recommendation, namespace="default")
    assert result is not None
    assert "deployment_id" in result
    assert "files" in result

    # Step 3: Validate generated YAMLs
    validator = YAMLValidator()
    validation_results = validator.validate_all(result["files"])
    for config_type, valid in validation_results.items():
        assert valid, f"YAML validation failed for {config_type}"
