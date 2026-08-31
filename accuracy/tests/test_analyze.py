"""Unit tests for analyze.py — pure data, no GPU or cluster required."""
import importlib.util
import json
import tempfile
from pathlib import Path

import pytest


def _import_analyze():
    spec = importlib.util.spec_from_file_location(
        "analyze",
        Path(__file__).parents[2] / "accuracy/scripts/analyze.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SAMPLE_RUN = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "gpu": "H100-80GB",
    "tp": 1, "pp": 1, "dp": 1, "max_model_len": 8192,
    "measured": {
        "weight_memory_gib": 14.23,
        "activation_memory_gib": 5.32,
        "non_torch_memory_gib": 0.14,
        "kv_cache_gib": 58.1,
    },
    "planner_predicted": {
        "weight_memory_gib": 14.23,
        "activation_memory_gib": 5.6,
        "non_torch_memory_gib": 0.15,
        "kv_cache_gib": 57.6,
    },
}


@pytest.mark.unit
def test_compute_error_pct():
    analyze = _import_analyze()
    errors = analyze.compute_error_pct(SAMPLE_RUN)
    assert abs(errors["weight_memory"] - 0.0) < 0.01
    assert abs(errors["activation_memory"] - 5.26) < 0.1
    assert abs(errors["non_torch_memory"] - 7.14) < 0.1
    assert errors["kv_cache"] < 0


@pytest.mark.unit
def test_load_runs_from_dir():
    analyze = _import_analyze()
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "run1.json").write_text(json.dumps(SAMPLE_RUN))
        (Path(d) / "run2.json").write_text(json.dumps({**SAMPLE_RUN, "tp": 2}))
        (Path(d) / "metadata.txt").write_text("ignored")
        runs = analyze.load_runs(d)
    assert len(runs) == 2


@pytest.mark.unit
def test_outlier_detection():
    analyze = _import_analyze()
    run_ok = {**SAMPLE_RUN, "error_pct": {
        "weight_memory": 0.0, "activation_memory": 5.0,
        "non_torch_memory": 7.0, "kv_cache": -0.9}}
    run_bad = {**SAMPLE_RUN, "model": "other/model", "error_pct": {
        "weight_memory": 0.0, "activation_memory": 15.0,
        "non_torch_memory": 7.0, "kv_cache": -0.9}}
    outliers = analyze.find_outliers([run_ok, run_bad], threshold_pct=10.0)
    assert len(outliers) == 1
    assert outliers[0]["model"] == "other/model"


@pytest.mark.unit
def test_markdown_report_contains_required_sections():
    analyze = _import_analyze()
    run = {**SAMPLE_RUN, "error_pct": {
        "weight_memory": 0.0, "activation_memory": 5.26,
        "non_torch_memory": 7.14, "kv_cache": -0.86}}
    report = analyze.generate_markdown_report([run])
    for section in ["## Per-component error", "## Per-architecture error",
                    "## Argument sensitivity", "## Outliers", "## Calibration decisions"]:
        assert section in report, f"Missing section: {section}"
