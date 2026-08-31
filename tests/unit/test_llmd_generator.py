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
from planner.shared.schemas.intent import DeploymentIntent
from planner.shared.schemas.recommendation import (
    DeploymentConfiguration,
    DeploymentRecommendation,
    GPUConfig,
)
from planner.shared.schemas.specification import SLOTargets, TrafficProfile


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
def sample_recommendation() -> DeploymentRecommendation:
    return DeploymentRecommendation(
        intent=DeploymentIntent(
            use_case="chatbot_conversational",
            user_count=100,
        ),
        traffic_profile=TrafficProfile(prompt_tokens=512, output_tokens=256, expected_qps=9.0),
        slo_targets=SLOTargets(
            ttft_target_ms=150,
            itl_target_ms=25,
            e2e_target_ms=7000,
        ),
        model_id="meta-llama/Llama-3-8B-Instruct",
        model_name="Llama-3-8B-Instruct",
        model_uri=None,
        meets_slo=True,
        gpu_config=GPUConfig(
            gpu_type="NVIDIA-A100-80GB",
            gpu_count=6,
            tensor_parallel=2,
            replicas=3,
        ),
        reasoning="test recommendation",
    )


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
        parsed = yaml.safe_load(result["contents"]["helm_values"])

        custom_config = parsed["inferenceExtension"]["pluginsCustomConfig"]
        config_key = list(custom_config.keys())[0]
        epp_config = yaml.safe_load(custom_config[config_key])
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


@pytest.mark.unit
class TestPDDisaggregation:
    """Tests for P/D (prefill/decode) disaggregation output."""

    def test_pd_disabled_produces_single_patch(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        """Default (pd_enabled=False) has patch_vllm, no patch_prefill/patch_decode."""
        result = llmd_generator.generate_all(sample_config)

        assert "patch_vllm" in result["contents"]
        assert "patch_prefill" not in result["contents"]
        assert "patch_decode" not in result["contents"]

    def test_pd_disabled_patch_targets_decode_deployment(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        """Non-PD patch must use name 'decode' to match the llm-d base Deployment."""
        result = llmd_generator.generate_all(sample_config)
        parsed = yaml.safe_load(result["contents"]["patch_vllm"])

        assert parsed["metadata"]["name"] == "decode"

    def test_pd_enabled_produces_prefill_and_decode_patches(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        """pd_enabled=True produces patch_prefill and patch_decode, no patch_vllm."""
        result = llmd_generator.generate_all(sample_config, pd_enabled=True)

        assert "patch_prefill" in result["contents"]
        assert "patch_decode" in result["contents"]
        assert "patch_vllm" not in result["contents"]

    def test_pd_prefill_patch_has_correct_replicas(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        """Prefill patch uses prefill_replicas value."""
        result = llmd_generator.generate_all(sample_config, pd_enabled=True, prefill_replicas=2)
        parsed = yaml.safe_load(result["contents"]["patch_prefill"])

        assert parsed["spec"]["replicas"] == 2
        assert parsed["metadata"]["name"] == "prefill"

    def test_pd_decode_patch_has_correct_replicas(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        """Decode patch uses decode_replicas value."""
        result = llmd_generator.generate_all(sample_config, pd_enabled=True, decode_replicas=3)
        parsed = yaml.safe_load(result["contents"]["patch_decode"])

        assert parsed["spec"]["replicas"] == 3
        assert parsed["metadata"]["name"] == "decode"

    def test_pd_kustomization_references_both_patches(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        """Kustomization patches list has patch-prefill.yaml and patch-decode.yaml."""
        result = llmd_generator.generate_all(sample_config, pd_enabled=True)
        parsed = yaml.safe_load(result["contents"]["kustomization"])

        patch_paths = [p["path"] for p in parsed["patches"]]
        assert "patch-prefill.yaml" in patch_paths
        assert "patch-decode.yaml" in patch_paths
        assert "patch-vllm.yaml" not in patch_paths

    def test_pd_all_outputs_valid_yaml(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        """All contents parse as valid YAML when pd_enabled=True."""
        result = llmd_generator.generate_all(sample_config, pd_enabled=True)

        for key, content in result["contents"].items():
            parsed = yaml.safe_load(content)
            assert parsed is not None, f"{key} rendered as empty YAML"

    def test_rejects_zero_prefill_replicas(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        """generate_all() raises ValueError when prefill_replicas < 1."""
        with pytest.raises(ValueError, match="must be between 1 and 32"):
            llmd_generator.generate_all(sample_config, pd_enabled=True, prefill_replicas=0)

    def test_rejects_zero_decode_replicas(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        """generate_all() raises ValueError when decode_replicas < 1."""
        with pytest.raises(ValueError, match="must be between 1 and 32"):
            llmd_generator.generate_all(sample_config, pd_enabled=True, decode_replicas=0)

    def test_rejects_prefill_replicas_above_max(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        """generate_all() raises ValueError when prefill_replicas > 32."""
        with pytest.raises(ValueError, match="must be between 1 and 32"):
            llmd_generator.generate_all(sample_config, pd_enabled=True, prefill_replicas=33)

    def test_rejects_decode_replicas_above_max(
        self,
        llmd_generator: LlmdDeploymentGenerator,
        sample_config: DeploymentConfiguration,
    ) -> None:
        """generate_all() raises ValueError when decode_replicas > 32."""
        with pytest.raises(ValueError, match="must be between 1 and 32"):
            llmd_generator.generate_all(sample_config, pd_enabled=True, decode_replicas=33)


@pytest.mark.unit
class TestDeployAPINewParams:
    """Tests for new parameters exposed in the deploy API endpoint."""

    def test_generate_deployment_llmd_with_pd_enabled(
        self, client: TestClient, sample_config: DeploymentConfiguration
    ) -> None:
        """POST with pd_enabled=True should return patch_prefill and patch_decode."""
        response = client.post(
            "/api/v1/generate-deployment",
            json={
                "configuration": sample_config.model_dump(),
                "namespace": "test-ns",
                "stack": "llm-d",
                "pd_enabled": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "patch_prefill" in data["files"]
        assert "patch_decode" in data["files"]

    @pytest.mark.parametrize(
        "field",
        ["prefill_replicas", "decode_replicas"],
    )
    def test_generate_deployment_rejects_zero_replicas(
        self,
        client: TestClient,
        sample_config: DeploymentConfiguration,
        field: str,
    ) -> None:
        """API returns 422 when replica count is < 1."""
        payload = {
            "configuration": sample_config.model_dump(),
            "namespace": "test-ns",
            "stack": "llm-d",
            "pd_enabled": True,
            field: 0,
        }
        response = client.post("/api/v1/generate-deployment", json=payload)
        assert response.status_code == 422

    def test_generate_deployment_rejects_replicas_above_max(
        self,
        client: TestClient,
        sample_config: DeploymentConfiguration,
    ) -> None:
        """API returns 422 when replica count exceeds 32."""
        response = client.post(
            "/api/v1/generate-deployment",
            json={
                "configuration": sample_config.model_dump(),
                "namespace": "test-ns",
                "stack": "llm-d",
                "pd_enabled": True,
                "prefill_replicas": 33,
            },
        )
        assert response.status_code == 422
