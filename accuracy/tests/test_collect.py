"""Unit tests for collect.py — no cluster or kubectl required."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _import_collect():
    spec = importlib.util.spec_from_file_location(
        "collect",
        Path(__file__).parents[2] / "accuracy/scripts/collect.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
def test_reader_pod_manifest():
    """Reader pod must mount the correct PVC, use busybox, and be named vllm-mem-reader."""
    collect = _import_collect()
    manifest = collect._reader_pod_manifest(namespace="llmdplanner", pvc="vllm-mem-data")
    assert manifest["metadata"]["name"] == "vllm-mem-reader"
    assert manifest["metadata"]["namespace"] == "llmdplanner"
    spec = manifest["spec"]
    assert spec["containers"][0]["image"] == "busybox"
    volumes = spec["volumes"]
    pvc_vol = next(v for v in volumes if "persistentVolumeClaim" in v)
    assert pvc_vol["persistentVolumeClaim"]["claimName"] == "vllm-mem-data"


@pytest.mark.unit
def test_resolve_version_from_arg():
    """Explicit --vllm-version wins over sweep.yaml."""
    collect = _import_collect()
    version = collect._resolve_version(
        vllm_version_arg="v0.99.0",
        sweep_path=None,
    )
    assert version == "v0.99.0"


@pytest.mark.unit
def test_resolve_version_from_sweep(tmp_path):
    """When --vllm-version omitted, version is extracted from sweep.yaml vllm_image."""
    collect = _import_collect()
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text("defaults:\n  vllm_image: vllm/vllm-openai:v0.19.0\n")
    version = collect._resolve_version(vllm_version_arg=None, sweep_path=str(sweep))
    assert version == "v0.19.0"


@pytest.mark.unit
def test_resolve_version_missing_raises():
    """No --vllm-version and no sweep.yaml must exit with a clear error."""
    collect = _import_collect()
    with pytest.raises(SystemExit):
        collect._resolve_version(vllm_version_arg=None, sweep_path=None)


@pytest.mark.unit
@patch("subprocess.run")
def test_dry_run_no_subprocess(mock_run, capsys, tmp_path):
    """--dry-run must print paths and make zero subprocess.run calls."""
    collect = _import_collect()
    collect.collect(
        vllm_version="v0.19.0",
        namespace="llmdplanner",
        pvc="vllm-mem-data",
        out=str(tmp_path),
        dry_run=True,
    )
    mock_run.assert_not_called()
    out = capsys.readouterr().out
    assert "v0.19.0" in out
