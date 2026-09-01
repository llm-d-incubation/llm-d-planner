"""Tests for data file path resolver."""

from pathlib import Path

import pytest


@pytest.mark.unit
class TestDataPath:
    def test_resolves_bundled_config_file(self):
        from planner.data._resolver import data_path

        path = data_path("configuration/model_catalog.json")
        assert path.exists()
        assert path.name == "model_catalog.json"

    def test_resolves_bundled_quality_file(self):
        from quality_scoring.data._resolver import quality_data_path

        path = quality_data_path("arena_models.json")
        assert path.exists()

    def test_resolves_bundled_performance_file(self):
        from planner.data._resolver import data_path

        path = data_path("performance/benchmarks_BLIS.json")
        assert path.exists()

    def test_override_with_custom_dir(self, tmp_path):
        from planner.data._resolver import data_path

        custom = tmp_path / "configuration" / "test.json"
        custom.parent.mkdir(parents=True)
        custom.write_text("{}")
        result = data_path("configuration/test.json", data_dir=tmp_path)
        assert result == custom

    def test_default_returns_path_inside_package(self):
        from planner.data._resolver import data_path

        path = data_path("configuration/usecase_slo_workload.json")
        assert "planner/data" in str(path)
