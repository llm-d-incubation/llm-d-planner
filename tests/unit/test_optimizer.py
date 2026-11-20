"""Real optimizer calculations retain their results and use request-scoped HF credentials."""

import json
import os
from dataclasses import asdict
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import patch

import pytest

if find_spec("llm_optimizer") is None:
    pytest.skip("llm-optimizer is not installed; run make setup-backend", allow_module_level=True)
else:
    from huggingface_hub import constants
    from huggingface_hub.utils import build_hf_headers
    from llm_optimizer import common, performance

    from planner import optimizer
    from planner.capacity_planner import get_model_config_from_hf, get_model_info_from_hf
    from planner.gpu_recommender import GPURecommender

pytestmark = pytest.mark.unit


@pytest.fixture
def model_config():
    return {
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "intermediate_size": 512,
        "vocab_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "torch_dtype": "float16",
    }


@pytest.mark.parametrize(
    ("configured_token", "environment_token", "expected_header"),
    [
        ("hf_configured", "hf_environment", "Bearer hf_configured"),
        (None, "hf_environment", "Bearer hf_environment"),
        (None, None, None),
        ("", "hf_environment", None),
    ],
)
def test_gpu_results_authenticate_real_optimizer_download(
    configured_token, environment_token, expected_header, model_config, monkeypatch, tmp_path
):
    if environment_token is None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HF_TOKEN", environment_token)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(constants, "HF_TOKEN_PATH", str(tmp_path / "no-saved-token"))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(model_config))
    headers = []

    def download(*, repo_id, filename, token=None):
        assert repo_id == "test/private-model"
        assert filename == "config.json"
        headers.append(build_hf_headers(token=token).get("authorization"))
        return str(config_path)

    get_model_config_from_hf.cache_clear()
    get_model_info_from_hf.cache_clear()
    try:
        with (
            patch("planner.capacity_planner.HfApi"),
            patch(
                "planner.capacity_planner.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(),
            ),
            patch.object(common, "hf_hub_download", side_effect=download) as anonymous_download,
            patch.object(optimizer, "hf_hub_download", side_effect=download),
            patch(
                "huggingface_hub.list_repo_files", side_effect=AssertionError("Unexpected HF call")
            ) as anonymous_lookup,
        ):
            recommender = GPURecommender(
                "test/private-model", 16, 8, gpu_list=["H100"], hf_token=configured_token
            )
            results, failures = recommender.get_gpu_results()
        assert failures == {}
        assert results["H100"].best_configs["best_latency"].output_throughput_tps > 0
        assert results["H100"].tuning_commands["advanced"]["vllm"]["commands"]
        assert headers == [expected_header]
        anonymous_download.assert_not_called()
        anonymous_lookup.assert_not_called()
        assert os.getenv("HF_TOKEN") == environment_token
    finally:
        get_model_config_from_hf.cache_clear()
        get_model_info_from_hf.cache_clear()


@pytest.mark.parametrize("constraints", [None, "ttft:p95<=1000000ms;itl:p95<=1000000ms"])
def test_estimation_matches_optimizer_runner(model_config, constraints, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(model_config))
    params = performance.PerformanceEstimationParams(
        model="test/model",
        input_len=16,
        output_len=8,
        gpu="H100",
        num_gpus=1,
        framework="vllm",
        target="throughput",
        constraints=constraints,
    )
    with patch.object(common, "hf_hub_download", return_value=str(config_path)):
        expected_params, expected_result = performance.run_performance_estimation(params)
    with patch.object(optimizer, "hf_hub_download", return_value=str(config_path)):
        config = optimizer.get_optimizer_model_config(params.model, "hf_configured")
        actual_params, actual_result = optimizer.run_performance_estimation(params, config)
    assert actual_params == expected_params
    assert asdict(actual_result) == asdict(expected_result)


@pytest.mark.parametrize(
    ("changes", "model_id", "bytes_per_param", "expected_precision"),
    [
        ({"quantization_config": {"quant_method": "fp8"}}, "test/model-fp16", None, "fp8"),
        ({"quantization_config": {"quant_method": "awq"}}, "test/model-bf16", None, "bf16"),
        ({"quantization_config": {"bits": 8}}, "test/model-fp16", None, "fp8"),
        (
            {"quantization_config": {"quant_method": "unrecognized"}},
            "test/model-fp16",
            None,
            "fp16",
        ),
        ({}, "test/model-fp8", None, "fp8"),
        ({"torch_dtype": "bfloat16"}, "test/model-fp16", None, "fp16"),
        ({"torch_dtype": "bfloat16"}, "test/model", None, "bf16"),
        ({"torch_dtype": None, "_name_or_path": "test/model-fp16"}, "test/model", None, "fp16"),
        ({"torch_dtype": None, "_name_or_path": "test/model-float16"}, "test/model", 2, "fp16"),
        ({"torch_dtype": None, "dtype": "bfloat16"}, "test/model", None, "bf16"),
        ({"torch_dtype": None, "dtype": "float16"}, "test/model", 2, "bf16"),
        ({"torch_dtype": None}, "test/model", 1.3, "fp8"),
        ({"torch_dtype": None}, "test/model", 2.5, "bf16"),
        ({"torch_dtype": None}, "test/model", 4.5, "fp32"),
        ({"torch_dtype": None}, "test/model", 5, "fp16"),
        ({"torch_dtype": None}, "test/model", None, "fp16"),
    ],
)
def test_model_conversion_matches_optimizer_precision(
    model_config, changes, model_id, bytes_per_param, expected_precision, tmp_path
):
    model_config.update(changes)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(model_config))
    size = (
        int(common.calculate_model_parameters_from_config(model_config) * bytes_per_param)
        if bytes_per_param is not None
        else None
    )
    info = SimpleNamespace(siblings=[SimpleNamespace(rfilename="weights.safetensors", size=size)])
    with (
        patch.object(common, "hf_hub_download", return_value=str(config_path)),
        patch("huggingface_hub.list_repo_files", return_value=["weights.safetensors"]) as files,
        patch("huggingface_hub.get_hf_file_metadata", return_value=SimpleNamespace(size=size)),
        patch("huggingface_hub.repo_info", return_value=info),
    ):
        expected = common.get_model_config_and_precision_from_hf(model_id)
    with (
        patch.object(optimizer, "hf_hub_download", return_value=str(config_path)) as download,
        patch.object(optimizer, "HfApi") as api,
        patch(
            "huggingface_hub.list_repo_files", side_effect=AssertionError("Unexpected HF call")
        ) as anonymous_lookup,
    ):
        api.return_value.model_info.return_value = info
        actual = optimizer.get_optimizer_model_config(model_id, "hf_configured")
    assert actual == expected
    assert actual.inferred_precision == expected_precision
    anonymous_lookup.assert_not_called()
    assert api.called == files.called
    download.assert_called_once_with(
        repo_id=model_id, filename="config.json", token="hf_configured"
    )
    if files.called:
        api.assert_called_once_with(token="hf_configured")
        api.return_value.model_info.assert_called_once_with(model_id, files_metadata=True)


def test_unsatisfiable_constraints_remain_a_failure(model_config, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(model_config))
    params = performance.PerformanceEstimationParams(
        model="test/model",
        input_len=16,
        output_len=8,
        gpu="H100",
        num_gpus=1,
        framework="vllm",
        constraints="ttft:p95<=0ms",
    )
    with (
        patch.object(common, "hf_hub_download", return_value=str(config_path)),
        pytest.raises(ValueError, match="Cannot satisfy"),
    ):
        performance.run_performance_estimation(params)
    with patch.object(optimizer, "hf_hub_download", return_value=str(config_path)):
        config = optimizer.get_optimizer_model_config(params.model, "hf_configured")
        with pytest.raises(ValueError, match="Cannot satisfy"):
            optimizer.run_performance_estimation(params, config)


@pytest.mark.parametrize(
    "files",
    [
        [("weights.safetensors", 2)],
        [("part-1.safetensors", 1), ("part-2.safetensors", 1), ("config.json", 10)],
        [("weights.safetensors", None), ("weights.safetensors.index.json", 10)],
        [],
    ],
)
def test_precision_size_fallback_matches_optimizer_file_selection(model_config, files):
    model_config.pop("torch_dtype")
    count = common.calculate_model_parameters_from_config(model_config)
    siblings = [
        SimpleNamespace(rfilename=name, size=int(multiplier * count) if multiplier else None)
        for name, multiplier in files
    ]
    info = SimpleNamespace(siblings=siblings)
    sizes = {sibling.rfilename: sibling.size for sibling in siblings}

    def metadata(url):
        return SimpleNamespace(size=sizes[url.rsplit("/", 1)[-1]])

    with (
        patch("huggingface_hub.list_repo_files", return_value=list(sizes)),
        patch("huggingface_hub.get_hf_file_metadata", side_effect=metadata),
        patch("huggingface_hub.repo_info", return_value=info),
    ):
        expected = common.infer_precision_from_config(model_config, "test/model")
    with patch.object(optimizer, "HfApi") as api:
        api.return_value.model_info.return_value = info
        actual = optimizer._infer_precision(model_config, "test/model", "hf_configured")
        api.assert_called_once_with(token="hf_configured")
    assert actual == expected


def test_precision_metadata_failure_keeps_optimizer_fallback(model_config):
    model_config.pop("torch_dtype")
    with patch("huggingface_hub.list_repo_files", side_effect=OSError("Unavailable")):
        expected = common.infer_precision_from_config(model_config, "test/model")
    with patch.object(optimizer, "HfApi") as api:
        api.return_value.model_info.side_effect = OSError("Unavailable")
        assert optimizer._infer_precision(model_config, "test/model", "hf_configured") == expected


@pytest.mark.parametrize("content", ["not json", "{}"])
def test_model_config_failures_match_optimizer(content, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(content)
    with (
        patch.object(common, "hf_hub_download", return_value=str(config_path)),
        pytest.raises((RuntimeError, KeyError)) as expected,
    ):
        common.get_model_config_and_precision_from_hf("test/model")
    with (
        patch.object(optimizer, "hf_hub_download", return_value=str(config_path)),
        pytest.raises(type(expected.value)) as actual,
    ):
        optimizer.get_optimizer_model_config("test/model", "hf_configured")
    assert str(actual.value) == str(expected.value)
