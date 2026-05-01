"""Unit tests for parse_log.py — no GPU or cluster required."""
import json
import os
import tempfile
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _import_parse():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "parse_log",
        Path(__file__).parents[2] / "accuracy/scripts/parse_log.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
def test_parse_all_fields():
    """Fixture log must yield exact expected output."""
    parse_log = _import_parse()
    result = parse_log.parse(FIXTURES / "llama_tp1_startup.log")
    expected = json.loads((FIXTURES / "llama_tp1_expected.json").read_text())
    assert result == expected


@pytest.mark.unit
def test_parse_missing_required_field_raises():
    """Any missing required field must raise ValueError naming the field."""
    parse_log = _import_parse()
    text = (FIXTURES / "llama_tp1_startup.log").read_text()
    # Remove the "GPU KV cache size" line — kv_cache_tokens becomes missing
    incomplete = "\n".join(l for l in text.splitlines() if "GPU KV cache size" not in l)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write(incomplete)
        tmp = f.name
    try:
        with pytest.raises(ValueError, match="kv_cache_tokens"):
            parse_log.parse(tmp)
    finally:
        os.unlink(tmp)


@pytest.mark.unit
def test_numeric_types():
    """Memory values must be float; block counts must be int."""
    parse_log = _import_parse()
    result = parse_log.parse(FIXTURES / "llama_tp1_startup.log")
    assert isinstance(result["weight_memory_gib"], float)
    assert isinstance(result["kv_cache_memory_gib"], float)
    assert isinstance(result["kv_cache_tokens"], int)
    assert isinstance(result["kv_cache_blocks"], int)
    assert isinstance(result["kv_block_size_bytes"], int)


@pytest.mark.unit
def test_parse_version_missing_is_nonfatal():
    """Missing version info is a warning, not an error — log may not have it."""
    parse_log = _import_parse()
    text = (FIXTURES / "llama_tp1_startup.log").read_text()
    no_version = "\n".join(l for l in text.splitlines() if "vLLM version" not in l)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write(no_version)
        tmp = f.name
    try:
        result = parse_log.parse(tmp)   # must not raise
        assert "vllm_version" not in result or result["vllm_version"] is None
    finally:
        os.unlink(tmp)
