"""Unit tests for sweep_runner matrix expansion and orchestration — no K8s required."""
import importlib.util
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _import_runner():
    spec = importlib.util.spec_from_file_location(
        "sweep_runner",
        Path(__file__).parents[2] / "accuracy/scripts/sweep_runner.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MINIMAL_CONFIG = {
    "defaults": {
        "gpu": "H100-80GB", "max_model_len": 8192,
        "pp": 1, "dp": 1, "gpu_memory_utilization": "0.90",
        "namespace": "llmdplanner", "vllm_image": "vllm/vllm-openai:v0.19.0",
    },
    "runs": [
        {"model": "org/model-7b", "tp": [1, 2, 4]},
        {"model": "org/model-70b", "tp": 2, "max_model_len": 16384},
    ],
}


@pytest.mark.unit
def test_expand_list_tp_into_three_runs():
    runner = _import_runner()
    expanded = runner.expand_matrix(MINIMAL_CONFIG)
    tp1_runs = [r for r in expanded if r["model"] == "org/model-7b"]
    assert len(tp1_runs) == 3
    assert {r["tp"] for r in tp1_runs} == {1, 2, 4}


@pytest.mark.unit
def test_defaults_applied():
    runner = _import_runner()
    for run in runner.expand_matrix(MINIMAL_CONFIG):
        assert run["pp"] == 1
        assert run["dp"] == 1
        assert run["gpu"] == "H100-80GB"
        assert run["namespace"] == "llmdplanner"


@pytest.mark.unit
def test_run_level_override_wins():
    runner = _import_runner()
    expanded = runner.expand_matrix(MINIMAL_CONFIG)
    overridden = [r for r in expanded if r["model"] == "org/model-70b"]
    assert len(overridden) == 1
    assert overridden[0]["max_model_len"] == 16384


@pytest.mark.unit
def test_run_ids_unique_and_deterministic():
    runner = _import_runner()
    expanded = runner.expand_matrix(MINIMAL_CONFIG)
    ids = [runner.make_run_id(r) for r in expanded]
    assert len(ids) == len(set(ids))
    assert runner.make_run_id(expanded[0]) == runner.make_run_id(expanded[0])


@pytest.mark.unit
def test_run_id_is_valid_k8s_name():
    """Run IDs must be lowercase alphanumeric+hyphens and ≤52 chars."""
    runner = _import_runner()
    run = {"model": "meta-llama/Llama-3.1-8B-Instruct",
           "gpu": "H100-80GB", "tp": 1, "pp": 1, "dp": 1, "max_model_len": 8192}
    rid = runner.make_run_id(run)
    assert re.match(r'^[a-z0-9][a-z0-9\-]*[a-z0-9]$', rid), f"Invalid K8s name: {rid}"
    assert len(rid) <= 52


@pytest.mark.unit
def test_run_sweep_skips_existing_results(tmp_path):
    """run_sweep must skip a run if its JSON already exists."""
    runner = _import_runner()
    run = {"model": "org/model-7b", "gpu": "H100-80GB",
           "tp": 1, "pp": 1, "dp": 1, "max_model_len": 8192,
           "namespace": "llmdplanner", "vllm_image": "v", "results_pvc": "pvc",
           "gpu_memory_utilization": "0.90",
           "node_selector": {}}
    run_id = runner.make_run_id(run)
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    (runs_dir / f"{run_id}.json").write_text('{"status":"done"}')

    mock_parse_log = MagicMock()
    with patch.dict(sys.modules, {"parse_log": mock_parse_log}), \
         patch.object(runner, "_load_k8s", return_value=(MagicMock(), MagicMock())), \
         patch.object(runner, "_submit_sub_job") as mock_submit:
        runner.run_sweep([run], tmp_path)
        mock_submit.assert_not_called()


@pytest.mark.unit
def test_extract_version_standard():
    runner = _import_runner()
    assert runner._extract_version("vllm/vllm-openai:v0.19.0") == "v0.19.0"


@pytest.mark.unit
def test_extract_version_no_tag():
    runner = _import_runner()
    assert runner._extract_version("vllm-openai") == "unknown"


@pytest.mark.unit
def test_versioned_subfolders(tmp_path):
    """run_sweep creates <results>/<version>/logs/ and <results>/<version>/runs/."""
    runner = _import_runner()
    version = "v0.19.0"
    results_dir = tmp_path / version
    logs_dir = results_dir / "logs"
    runs_dir = results_dir / "runs"
    logs_dir.mkdir(parents=True)
    runs_dir.mkdir(parents=True)
    assert logs_dir.is_dir()
    assert runs_dir.is_dir()


@pytest.mark.unit
def test_build_job_manifest_sets_correct_gpu_count():
    runner = _import_runner()
    run = {"model": "org/model", "gpu": "H100-80GB",
           "tp": 4, "pp": 2, "dp": 1, "max_model_len": 8192,
           "namespace": "llmdplanner", "vllm_image": "vllm/vllm-openai:v0.19.0",
           "results_pvc": "vllm-mem-data", "gpu_memory_utilization": "0.90",
           "node_selector": {"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3"}}
    manifest = runner._build_job_manifest(runner.make_run_id(run), run)
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    # TP=4 × PP=2 = 8 GPUs
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 8
