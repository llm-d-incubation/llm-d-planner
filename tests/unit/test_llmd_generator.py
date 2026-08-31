"""Unit tests for llm-d deployment generator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from planner.api.routes import configuration_router
from planner.configuration import DeploymentGenerator, YAMLValidator
from planner.configuration.llmd_generator import LlmdDeploymentGenerator
from planner.shared.schemas.recommendation import (
    DeploymentConfiguration,
    GPUConfig,
)


@pytest.fixture
def client(tmp_path) -> TestClient:
    """Create a test client with mocked app state (no DB or disk side-effects)."""
    app = FastAPI()

    with patch("planner.configuration.generator.ModelCatalog"):
        app.state.deployment_generator = DeploymentGenerator(
            output_dir=str(tmp_path / "vllm"), simulator_mode=False
        )
    app.state.llmd_deployment_generator = LlmdDeploymentGenerator(output_dir=str(tmp_path / "llmd"))
    app.state.yaml_validator = YAMLValidator()
    app.state.cluster_managers = {}
    app.state.cluster_manager_lock = MagicMock()

    app.include_router(configuration_router)

    return TestClient(app)


@pytest.fixture
def sample_config() -> DeploymentConfiguration:
    return DeploymentConfiguration(
        model_id="meta-llama/Llama-3-8B-Instruct",
        model_name="Llama-3-8B-Instruct",
        model_uri=None,
        gpu_config=GPUConfig(
            gpu_type="NVIDIA-A100-80GB",
            gpu_count=6,
            tensor_parallel=2,
            replicas=3,
        ),
        use_case="chatbot_conversational",
        expected_qps=9.0,
        prompt_tokens=512,
        output_tokens=256,
        e2e_target_ms=7000,
    )


@pytest.fixture
def llmd_generator(tmp_path) -> LlmdDeploymentGenerator:
    """Create an LlmdDeploymentGenerator writing to a temporary directory."""
    return LlmdDeploymentGenerator(output_dir=str(tmp_path))


@pytest.mark.unit
class TestLlmdGeneratorOutput:
    def test_invalid_model_id_raises(self, llmd_generator: LlmdDeploymentGenerator) -> None:
        """Test that invalid model_id format raises ValueError."""
        config = DeploymentConfiguration(
            model_id='bad-model"\nmalicious: code',
            model_name="bad-model",
            model_uri=None,
            gpu_config=GPUConfig(
                gpu_type="NVIDIA-A100-80GB",
                gpu_count=2,
                tensor_parallel=2,
                replicas=3,
            ),
            use_case="chatbot_conversational",
            expected_qps=9.0,
            prompt_tokens=512,
            output_tokens=256,
            e2e_target_ms=7000,
        )
        with pytest.raises(ValueError, match="Invalid model_id format"):
            llmd_generator.generate_all(config)

    def test_invalid_namespace_raises(self, llmd_generator: LlmdDeploymentGenerator) -> None:
        """Test that invalid namespace format raises ValueError."""
        config = DeploymentConfiguration(
            model_id="meta-llama/Llama-3-8B-Instruct",
            model_name="Llama-3-8B-Instruct",
            model_uri=None,
            gpu_config=GPUConfig(
                gpu_type="NVIDIA-A100-80GB",
                gpu_count=2,
                tensor_parallel=2,
                replicas=3,
            ),
            use_case="chatbot_conversational",
            expected_qps=9.0,
            prompt_tokens=512,
            output_tokens=256,
            e2e_target_ms=7000,
        )
        with pytest.raises(ValueError, match="Invalid namespace format"):
            llmd_generator.generate_all(config, namespace="INVALID NS!")

    def test_generate_all_returns_three_files(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        from pathlib import Path

        result = llmd_generator.generate_all(sample_config, namespace="prod")

        assert set(result["files"].keys()) == {
            "kustomization",
            "patch_vllm",
            "helm_values",
        }
        assert set(result["contents"].keys()) == {
            "kustomization",
            "patch_vllm",
            "helm_values",
        }
        for path in result["files"].values():
            assert Path(path).exists()

    def test_generate_all_returns_deployment_id_and_namespace(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        result = llmd_generator.generate_all(sample_config, namespace="myns")

        assert result["deployment_id"]
        assert result["namespace"] == "myns"

    def test_all_outputs_are_valid_yaml(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        result = llmd_generator.generate_all(sample_config)

        for key in ("kustomization", "patch_vllm", "helm_values"):
            parsed = yaml.safe_load(result["contents"][key])
            assert parsed is not None, f"{key} rendered as empty YAML"


@pytest.mark.unit
class TestKustomizationOutput:
    def test_references_llmd_base(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        result = llmd_generator.generate_all(sample_config)
        parsed = yaml.safe_load(result["contents"]["kustomization"])

        assert any("llm-d/llm-d" in r for r in parsed["resources"])

    def test_sets_name_prefix_from_deployment_id(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        result = llmd_generator.generate_all(sample_config)
        parsed = yaml.safe_load(result["contents"]["kustomization"])

        assert parsed["namePrefix"].startswith("chatbot")
        assert parsed["namePrefix"].endswith("-")

    def test_sets_app_label(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        result = llmd_generator.generate_all(sample_config)
        parsed = yaml.safe_load(result["contents"]["kustomization"])

        label_pairs = parsed["labels"][0]["pairs"]
        assert "app" in label_pairs


@pytest.mark.unit
class TestPatchVllmOutput:
    def test_uses_model_id(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        result = llmd_generator.generate_all(sample_config)
        parsed = yaml.safe_load(result["contents"]["patch_vllm"])

        container = parsed["spec"]["template"]["spec"]["containers"][0]
        assert "meta-llama/Llama-3-8B-Instruct" in container["args"]

    def test_uses_tensor_parallel(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        result = llmd_generator.generate_all(sample_config)
        parsed = yaml.safe_load(result["contents"]["patch_vllm"])

        container = parsed["spec"]["template"]["spec"]["containers"][0]
        assert "--tensor-parallel-size=2" in container["args"]

    def test_uses_replicas(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        result = llmd_generator.generate_all(sample_config)
        parsed = yaml.safe_load(result["contents"]["patch_vllm"])

        assert parsed["spec"]["replicas"] == 3

    def test_sets_gpu_resources_per_replica(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        result = llmd_generator.generate_all(sample_config)
        parsed = yaml.safe_load(result["contents"]["patch_vllm"])

        container = parsed["spec"]["template"]["spec"]["containers"][0]
        # gpu_count=6, tensor_parallel=2, replicas=3 — each pod gets tensor_parallel GPUs
        assert container["resources"]["requests"]["nvidia.com/gpu"] == "2"
        assert container["resources"]["limits"]["nvidia.com/gpu"] == "2"


@pytest.mark.unit
class TestHelmValuesOutput:
    def test_contains_inference_extension(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        result = llmd_generator.generate_all(sample_config)
        parsed = yaml.safe_load(result["contents"]["helm_values"])

        assert "inferenceExtension" in parsed
        assert (
            parsed["inferenceExtension"]["image"]["repository"] == "llm-d/llm-d-inference-scheduler"
        )

    def test_contains_inference_pool_selector(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        result = llmd_generator.generate_all(sample_config)
        parsed = yaml.safe_load(result["contents"]["helm_values"])

        pool = parsed["inferencePool"]
        assert "app" in pool["modelServers"]["matchLabels"]

    def test_contains_default_epp_config(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        result = llmd_generator.generate_all(sample_config)
        epp_config = _extract_epp_config(result)
        assert epp_config["kind"] == "EndpointPickerConfig"
        assert len(epp_config["plugins"]) == 4
        assert epp_config["schedulingProfiles"][0]["name"] == "default"

    def test_target_ports(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        result = llmd_generator.generate_all(sample_config)
        parsed = yaml.safe_load(result["contents"]["helm_values"])

        assert parsed["inferencePool"]["targetPorts"] == [{"number": 8000}]


@pytest.mark.unit
class TestGenerateDeploymentEndpointStack:
    def test_generate_deployment_with_stack_llmd(
        self, client: TestClient, sample_config: DeploymentConfiguration
    ) -> None:
        response = client.post(
            "/api/v1/generate-deployment",
            json={
                "configuration": sample_config.model_dump(),
                "namespace": "test-ns",
                "stack": "llm-d",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "kustomization" in data["files"]
        assert "helm_values" in data["files"]
        assert "patch_vllm" in data["files"]

    def test_generate_deployment_with_stack_llmd_with_routing_profile(
        self, client: TestClient, sample_config: DeploymentConfiguration
    ) -> None:
        response = client.post(
            "/api/v1/generate-deployment",
            json={
                "configuration": sample_config.model_dump(),
                "namespace": "test-ns",
                "stack": "llm-d",
                "routing_profile": "session-affinity",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "kustomization" in data["files"]

    def test_generate_deployment_with_invalid_routing_profile_returns_422(
        self, client: TestClient, sample_config: DeploymentConfiguration
    ) -> None:
        """API returns 422 for an invalid routing profile."""
        response = client.post(
            "/api/v1/generate-deployment",
            json={
                "configuration": sample_config.model_dump(),
                "namespace": "test-ns",
                "stack": "llm-d",
                "routing_profile": "nonexistent",
            },
        )
        assert response.status_code == 422

    def test_generate_deployment_with_stack_vllm_is_default(
        self, client: TestClient, sample_config: DeploymentConfiguration
    ) -> None:
        response = client.post(
            "/api/v1/generate-deployment",
            json={
                "configuration": sample_config.model_dump(),
                "namespace": "test-ns",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "inferenceservice" in data["files"]


def _extract_epp_config(result: dict) -> dict:
    """Parse the EndpointPickerConfig from generated helm values."""
    parsed = yaml.safe_load(result["contents"]["helm_values"])
    custom_config = parsed["inferenceExtension"]["pluginsCustomConfig"]
    config_key = list(custom_config.keys())[0]
    epp_config: dict = yaml.safe_load(custom_config[config_key])
    return epp_config


@pytest.mark.unit
class TestRoutingProfiles:
    """Tests for configurable routing profiles."""

    def test_default_routing_profile_unchanged(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        """Default profile should produce identical output to pre-profile behavior."""
        result = llmd_generator.generate_all(sample_config)
        epp_config = _extract_epp_config(result)

        plugin_types = [p["type"] for p in epp_config["plugins"]]
        assert plugin_types == [
            "prefix-cache-scorer",
            "decode-filter",
            "max-score-picker",
            "single-profile-handler",
        ]
        sched = epp_config["schedulingProfiles"][0]
        assert sched["name"] == "default"
        sched_refs = [p["pluginRef"] for p in sched["plugins"]]
        assert sched_refs == ["decode-filter", "max-score-picker", "prefix-cache-scorer"]
        weighted = [p for p in sched["plugins"] if p.get("weight")]
        assert len(weighted) == 1
        assert weighted[0]["pluginRef"] == "prefix-cache-scorer"
        assert weighted[0]["weight"] == 2

    def test_session_affinity_profile(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        """Session-affinity profile includes session-affinity-scorer plugin."""
        result = llmd_generator.generate_all(sample_config, routing_profile="session-affinity")
        epp_config = _extract_epp_config(result)

        plugin_types = [p["type"] for p in epp_config["plugins"]]
        assert "session-affinity-scorer" in plugin_types
        assert "prefix-cache-scorer" in plugin_types

        sched = epp_config["schedulingProfiles"][0]
        sched_refs = [p["pluginRef"] for p in sched["plugins"]]
        assert "session-affinity-scorer" in sched_refs
        sa_plugin = [p for p in sched["plugins"] if p["pluginRef"] == "session-affinity-scorer"][0]
        assert sa_plugin["weight"] == 3

    def test_throughput_optimized_profile(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        """Throughput-optimized profile uses load-aware-scorer instead of prefix-cache-scorer."""
        result = llmd_generator.generate_all(sample_config, routing_profile="throughput-optimized")
        epp_config = _extract_epp_config(result)

        plugin_types = [p["type"] for p in epp_config["plugins"]]
        assert "load-aware-scorer" in plugin_types
        assert "prefix-cache-scorer" not in plugin_types

        sched = epp_config["schedulingProfiles"][0]
        sched_refs = [p["pluginRef"] for p in sched["plugins"]]
        assert "load-aware-scorer" in sched_refs
        la_plugin = [p for p in sched["plugins"] if p["pluginRef"] == "load-aware-scorer"][0]
        assert la_plugin["weight"] == 3

    def test_invalid_routing_profile_raises(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        """Unknown routing profile should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown routing profile"):
            llmd_generator.generate_all(sample_config, routing_profile="nonexistent")
