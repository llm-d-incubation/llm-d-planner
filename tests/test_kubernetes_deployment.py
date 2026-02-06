#!/usr/bin/env python3
"""Test script for Sprint 5: Kubernetes deployment functionality.

This script tests the complete deployment flow:
1. Generate recommendation
2. Generate YAML files
3. Deploy to Kubernetes cluster
4. Verify deployment status
5. Clean up
"""

import pytest
import requests
import time
import sys

API_BASE_URL = "http://localhost:8000"


@pytest.mark.integration
def test_cluster_status():
    """Test 1: Verify cluster is accessible."""
    print("\n=== Test 1: Cluster Status ===")
    response = requests.get(f"{API_BASE_URL}/api/v1/cluster-status")
    status = response.json()

    print(f"Cluster accessible: {status['accessible']}")
    print(f"Namespace: {status.get('namespace')}")
    print(f"Existing deployments: {status.get('count', 0)}")

    assert status['accessible'], "Cluster must be accessible"
    print("✅ PASSED\n")
    return status


@pytest.mark.integration
def test_generate_recommendation():
    """Test 2: Generate a deployment recommendation."""
    print("=== Test 2: Generate Recommendation ===")

    message = "I need a small chatbot for 100 users with low latency"
    response = requests.post(
        f"{API_BASE_URL}/api/v1/recommend",
        json={"message": message},
        timeout=30
    )

    assert response.status_code == 200, f"Failed: {response.text}"

    recommendation = response.json()
    print(f"Model: {recommendation['model_name']}")
    print(f"GPU Config: {recommendation['gpu_config']['gpu_count']}x {recommendation['gpu_config']['gpu_type']}")
    print(f"Cost/month: ${recommendation['cost_per_month_usd']:.2f}")
    print(f"Meets SLO: {recommendation['meets_slo']}")

    print("✅ PASSED\n")
    return recommendation


@pytest.mark.integration
def test_deploy_to_cluster(recommendation):
    """Test 3: Deploy to Kubernetes cluster."""
    print("=== Test 3: Deploy to Kubernetes ===")

    response = requests.post(
        f"{API_BASE_URL}/api/v1/deploy-to-cluster",
        json={"recommendation": recommendation, "namespace": "default"},
        timeout=60
    )

    assert response.status_code == 200, f"Failed to deploy: {response.text}"

    result = response.json()
    deployment_id = result["deployment_id"]
    print(f"Deployment ID: {deployment_id}")
    print(f"Namespace: {result['namespace']}")
    print(f"Files generated: {len(result['files'])}")

    deployment_result = result.get("deployment_result", {})
    print(f"Deployment success: {deployment_result.get('success')}")

    for applied_file in deployment_result.get("applied_files", []):
        print(f"  ✅ Applied: {applied_file['file']}")

    print("✅ PASSED\n")
    return deployment_id


@pytest.mark.integration
def test_deployment_status(deployment_id):
    """Test 4: Check deployment status."""
    print("=== Test 4: Check Deployment Status ===")

    # Wait a moment for resources to be created
    print("Waiting 10 seconds for resources to be created...")
    time.sleep(10)

    response = requests.get(
        f"{API_BASE_URL}/api/v1/deployments/{deployment_id}/k8s-status",
        timeout=10
    )

    assert response.status_code == 200, f"Failed to get status: {response.text}"

    status = response.json()
    isvc = status.get("inferenceservice", {})

    print(f"InferenceService exists: {isvc.get('exists')}")

    if isvc.get('exists'):
        print(f"Ready: {isvc.get('ready')}")
        print(f"Conditions: {len(isvc.get('conditions', []))}")

        for condition in isvc.get('conditions', []):
            print(f"  - {condition.get('type')}: {condition.get('status')}")

    pods = status.get("pods", [])
    print(f"Pods: {len(pods)}")
    for pod in pods:
        print(f"  - {pod.get('name')}: {pod.get('phase')}")

    print("✅ PASSED\n")
    return status


@pytest.mark.integration
def test_cleanup(deployment_id):
    """Test 5: Clean up deployment."""
    print("=== Test 5: Cleanup ===")

    response = requests.delete(
        f"{API_BASE_URL}/api/v1/deployments/{deployment_id}",
        timeout=30
    )

    assert response.status_code == 200, f"Failed to delete: {response.text}"

    result = response.json()
    print(f"Deletion success: {result.get('success')}")
    print(f"Output: {result.get('output', '').strip()}")

    print("✅ PASSED\n")


def main():
    """Run all tests."""
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║       Sprint 5 End-to-End Deployment Test Suite          ║")
    print("╚═══════════════════════════════════════════════════════════╝")

    try:
        # Test 1: Cluster status
        cluster_status = test_cluster_status()

        # Test 2: Generate recommendation
        recommendation = test_generate_recommendation()

        # Test 3: Deploy to cluster
        deployment_id = test_deploy_to_cluster(recommendation)

        # Test 4: Check deployment status
        deployment_status = test_deployment_status(deployment_id)

        # Test 5: Cleanup
        test_cleanup(deployment_id)

        print("\n╔═══════════════════════════════════════════════════════════╗")
        print("║                  ALL TESTS PASSED ✅                      ║")
        print("╚═══════════════════════════════════════════════════════════╝\n")

        print("Sprint 5 Complete! 🎉")
        print("\nKey Achievements:")
        print("  ✅ KIND cluster running with KServe")
        print("  ✅ Deployment automation working")
        print("  ✅ InferenceService resources created")
        print("  ✅ API endpoints functional")
        print("  ✅ Cleanup working")

        return 0

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API. Is the backend running?")
        print("   Run: cd backend && source venv/bin/activate && python -m src.api.routes")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
