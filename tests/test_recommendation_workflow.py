"""Test script for end-to-end recommendation workflow."""

import json
import logging
from pathlib import Path

import pytest

from planner.orchestration.workflow import RecommendationWorkflow

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _load_scenarios():
    """Load demo scenarios from JSON file."""
    scenarios_path = Path(__file__).parent.parent / "data" / "configuration" / "demo_scenarios.json"
    with open(scenarios_path) as f:
        data = json.load(f)
    return data["scenarios"]


@pytest.fixture(scope="module")
def workflow():
    """Create a shared RecommendationWorkflow instance."""
    return RecommendationWorkflow()


@pytest.mark.integration
@pytest.mark.parametrize("scenario", _load_scenarios(), ids=lambda s: s["id"])
def test_scenario(workflow, scenario):
    """Test a single demo scenario."""
    print("\n" + "=" * 80)
    print(f"SCENARIO: {scenario['name']}")
    print("=" * 80)
    print(f"\nDescription: {scenario['description']}")
    print(f"\nUser Message: {scenario['user_description']}\n")

    # Generate recommendation
    recommendation = workflow.generate_recommendation(user_message=scenario["user_description"])

    # Verify we got a recommendation back
    assert recommendation is not None
    assert recommendation.model_name
    assert recommendation.gpu_config.gpu_count >= 1

    # Display results
    print("\n--- RECOMMENDATION ---")
    print(f"Model: {recommendation.model_name}")
    print(
        f"GPU Config: {recommendation.gpu_config.gpu_count}x {recommendation.gpu_config.gpu_type}"
    )
    print(f"  - Tensor Parallel: {recommendation.gpu_config.tensor_parallel}")
    print(f"  - Replicas: {recommendation.gpu_config.replicas}")

    print("\nCost:")
    print(f"  - Per Hour: ${recommendation.cost_per_hour_usd:.2f}")
    print(f"  - Per Month: ${recommendation.cost_per_month_usd:.2f}")

    print("\nPredicted Performance:")
    print(
        f"  - TTFT p95: {recommendation.predicted_ttft_p95_ms}ms (target: {recommendation.slo_targets.ttft_p95_target_ms}ms)"
    )
    print(
        f"  - ITL p95: {recommendation.predicted_itl_p95_ms}ms (target: {recommendation.slo_targets.itl_p95_target_ms}ms)"
    )
    print(
        f"  - E2E p95: {recommendation.predicted_e2e_p95_ms}ms (target: {recommendation.slo_targets.e2e_p95_target_ms}ms)"
    )
    print(f"  - Throughput: {recommendation.predicted_throughput_qps:.1f} QPS")

    print("\nTraffic Profile:")
    print(f"  - Expected QPS: {recommendation.traffic_profile.expected_qps:.1f}")
    print(f"  - Prompt Tokens: {recommendation.traffic_profile.prompt_tokens} tokens")
    print(f"  - Output Tokens: {recommendation.traffic_profile.output_tokens} tokens")

    print(f"\nMeets SLO: {'YES' if recommendation.meets_slo else 'NO'}")
    print(f"\nReasoning: {recommendation.reasoning}")

    # Check against expected recommendation if provided
    if "expected_recommendation" in scenario:
        expected = scenario["expected_recommendation"]
        print("\n--- COMPARISON WITH EXPECTED ---")

        model_match = recommendation.model_id == expected["model_id"]
        print(f"Model Match: {'yes' if model_match else 'no'} (expected: {expected['model_id']})")

        gpu_match = recommendation.gpu_config.gpu_type == expected["gpu_config"]["gpu_type"]
        print(
            f"GPU Type Match: {'yes' if gpu_match else 'no'} (expected: {expected['gpu_config']['gpu_type']})"
        )
