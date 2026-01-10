"""Test script for end-to-end recommendation workflow."""

import json
import logging
import pytest
from pathlib import Path

from src.orchestration.workflow import RecommendationWorkflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@pytest.mark.integration
def test_scenario(workflow: RecommendationWorkflow, scenario: dict):
    """Test a single demo scenario."""
    print("\n" + "="*80)
    print(f"SCENARIO: {scenario['name']}")
    print("="*80)
    print(f"\nDescription: {scenario['description']}")
    print(f"\nUser Message: {scenario['user_description']}\n")

    try:
        # Generate recommendation
        recommendation = workflow.generate_recommendation(
            user_message=scenario['user_description']
        )

        # Display results
        print("\n--- RECOMMENDATION ---")
        print(f"Model: {recommendation.model_name}")
        print(f"GPU Config: {recommendation.gpu_config.gpu_count}x {recommendation.gpu_config.gpu_type}")
        print(f"  - Tensor Parallel: {recommendation.gpu_config.tensor_parallel}")
        print(f"  - Replicas: {recommendation.gpu_config.replicas}")

        print(f"\nCost:")
        print(f"  - Per Hour: ${recommendation.cost_per_hour_usd:.2f}")
        print(f"  - Per Month: ${recommendation.cost_per_month_usd:.2f}")

        print(f"\nPredicted Performance:")
        print(f"  - TTFT p95: {recommendation.predicted_ttft_p95_ms}ms (target: {recommendation.slo_targets.ttft_p95_target_ms}ms)")
        print(f"  - ITL p95: {recommendation.predicted_itl_p95_ms}ms (target: {recommendation.slo_targets.itl_p95_target_ms}ms)")
        print(f"  - E2E p95: {recommendation.predicted_e2e_p95_ms}ms (target: {recommendation.slo_targets.e2e_p95_target_ms}ms)")
        print(f"  - Throughput: {recommendation.predicted_throughput_qps:.1f} QPS")

        print(f"\nTraffic Profile:")
        print(f"  - Expected QPS: {recommendation.traffic_profile.expected_qps:.1f}")
        print(f"  - Prompt Tokens: {recommendation.traffic_profile.prompt_tokens} tokens")
        print(f"  - Output Tokens: {recommendation.traffic_profile.output_tokens} tokens")

        print(f"\nMeets SLO: {'✅ YES' if recommendation.meets_slo else '❌ NO'}")
        print(f"\nReasoning: {recommendation.reasoning}")

        # Check against expected recommendation
        if 'expected_recommendation' in scenario:
            expected = scenario['expected_recommendation']
            print("\n--- COMPARISON WITH EXPECTED ---")

            model_match = recommendation.model_id == expected['model_id']
            print(f"Model Match: {'✅' if model_match else '❌'} (expected: {expected['model_id']})")

            gpu_match = recommendation.gpu_config.gpu_type == expected['gpu_config']['gpu_type']
            print(f"GPU Type Match: {'✅' if gpu_match else '❌'} (expected: {expected['gpu_config']['gpu_type']})")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.error(f"Failed to process scenario: {e}", exc_info=True)
        return False


def main():
    """Run end-to-end tests with demo scenarios."""
    print("\n" + "="*80)
    print("NEURALNAV - END-TO-END TEST")
    print("="*80)

    # Initialize workflow
    print("\nInitializing workflow...")
    workflow = RecommendationWorkflow()
    print("✅ Workflow initialized")

    # Load demo scenarios
    scenarios_path = Path(__file__).parent.parent / "data" / "demo_scenarios.json"
    with open(scenarios_path) as f:
        data = json.load(f)
        scenarios = data["scenarios"]

    print(f"\nLoaded {len(scenarios)} demo scenarios")

    # Test each scenario
    results = []
    for scenario in scenarios:
        success = test_scenario(workflow, scenario)
        results.append((scenario['name'], success))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    total = len(results)
    passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {passed}/{total} scenarios passed")

    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
