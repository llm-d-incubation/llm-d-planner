"""Tests for catalog-to-optimizer GPU name mapping."""

import pytest

from planner.shared.utils import CATALOG_TO_OPTIMIZER_GPU, catalog_to_optimizer_gpu_name


@pytest.mark.unit
class TestCatalogToOptimizerGpuName:
    """Test catalog_to_optimizer_gpu_name translation function."""

    def test_a100_80_maps_to_a100(self):
        assert catalog_to_optimizer_gpu_name("A100-80") == "A100"

    def test_a100_40_maps_to_a100_40gb(self):
        assert catalog_to_optimizer_gpu_name("A100-40") == "A100-40GB"

    def test_h100_identity(self):
        assert catalog_to_optimizer_gpu_name("H100") == "H100"

    def test_unknown_gpu_passes_through(self):
        assert catalog_to_optimizer_gpu_name("MI300X") == "MI300X"

    def test_empty_string_passes_through(self):
        assert catalog_to_optimizer_gpu_name("") == ""


@pytest.mark.unit
class TestCatalogToOptimizerGpuMapping:
    """Test the shared CATALOG_TO_OPTIMIZER_GPU mapping completeness."""

    def test_contains_all_roofline_supported_gpus(self):
        expected = {"H100", "H200", "A100-80", "A100-40", "L40", "L20", "B100", "B200"}
        assert set(CATALOG_TO_OPTIMIZER_GPU.keys()) == expected

    def test_a100_variants_differ_from_catalog_name(self):
        assert CATALOG_TO_OPTIMIZER_GPU["A100-80"] != "A100-80"
        assert CATALOG_TO_OPTIMIZER_GPU["A100-40"] != "A100-40"
