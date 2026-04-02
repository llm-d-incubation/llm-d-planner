"""Unit tests for cluster GPU detection."""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestGPUProductMap:
    """Validate GPU_PRODUCT_MAP values match CANONICAL_GPUS."""

    def test_all_map_values_are_canonical(self):
        from planner.cluster.gpu_detector import GPU_PRODUCT_MAP
        from planner.shared.utils.gpu_normalizer import CANONICAL_GPUS

        for label, canonical in GPU_PRODUCT_MAP.items():
            assert (
                canonical in CANONICAL_GPUS
            ), f"GPU_PRODUCT_MAP['{label}'] = '{canonical}' is not in CANONICAL_GPUS"

    def test_all_canonical_gpus_have_at_least_one_mapping(self):
        from planner.cluster.gpu_detector import GPU_PRODUCT_MAP
        from planner.shared.utils.gpu_normalizer import CANONICAL_GPUS

        mapped_canonical = set(GPU_PRODUCT_MAP.values())
        for gpu in CANONICAL_GPUS:
            assert gpu in mapped_canonical, f"Canonical GPU '{gpu}' has no entry in GPU_PRODUCT_MAP"


def _make_node(name: str, gpu_label: str | None = None) -> MagicMock:
    """Helper: create a mock K8s V1Node with optional GPU label."""
    node = MagicMock()
    node.metadata.name = name
    labels = {}
    if gpu_label:
        labels["nvidia.com/gpu.product"] = gpu_label
    node.metadata.labels = labels
    return node


@pytest.mark.unit
class TestDetectClusterGPUs:
    """Test detect_cluster_gpus() with mocked kubernetes client."""

    def setup_method(self):
        from planner.cluster.gpu_detector import reset_gpu_cache

        reset_gpu_cache()

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", False)
    def test_returns_empty_when_kubernetes_not_installed(self):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        result = detect_cluster_gpus()
        assert result == []

    @patch.dict(os.environ, {"PLANNER_DETECT_CLUSTER_GPUS": "false"})
    def test_returns_empty_when_disabled_via_env(self):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        result = detect_cluster_gpus()
        assert result == []

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_detects_single_gpu_type(self, mock_list, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [_make_node("node1", "NVIDIA-H100-80GB-HBM3")]
        result = detect_cluster_gpus()
        assert result == ["H100"]

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_detects_multiple_gpu_types(self, mock_list, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [
            _make_node("node1", "NVIDIA-A100-SXM4-80GB"),
            _make_node("node2", "NVIDIA-H100-80GB-HBM3"),
            _make_node("node3", "NVIDIA-L4"),
        ]
        result = detect_cluster_gpus()
        assert result == ["A100-80", "H100", "L4"]

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_deduplicates_same_gpu_on_multiple_nodes(self, mock_list, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [
            _make_node("node1", "NVIDIA-H100-80GB-HBM3"),
            _make_node("node2", "NVIDIA-H100-SXM5-80GB"),
        ]
        result = detect_cluster_gpus()
        assert result == ["H100"]

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_skips_nodes_without_gpu_labels(self, mock_list, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [
            _make_node("cpu-node1"),
            _make_node("gpu-node1", "NVIDIA-H100-80GB-HBM3"),
            _make_node("cpu-node2"),
        ]
        result = detect_cluster_gpus()
        assert result == ["H100"]

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_skips_unknown_gpu_labels(self, mock_list, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [
            _make_node("node1", "NVIDIA-H100-80GB-HBM3"),
            _make_node("node2", "Tesla-V100-SXM2-32GB"),  # Unknown
        ]
        result = detect_cluster_gpus()
        assert result == ["H100"]

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_returns_empty_when_no_nodes(self, mock_list, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = []
        result = detect_cluster_gpus()
        assert result == []

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_returns_empty_when_no_gpu_labels_on_any_node(self, mock_list, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [_make_node("cpu1"), _make_node("cpu2")]
        result = detect_cluster_gpus()
        assert result == []

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config", side_effect=Exception("no config"))
    def test_returns_empty_on_config_error(self, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        result = detect_cluster_gpus()
        assert result == []

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes", side_effect=Exception("API error"))
    def test_returns_empty_on_api_error(self, mock_list, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        result = detect_cluster_gpus()
        assert result == []

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_case_insensitive_label_matching(self, mock_list, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [_make_node("node1", "nvidia-l4")]
        result = detect_cluster_gpus()
        assert result == ["L4"]

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_a100_40gb_variants(self, mock_list, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [
            _make_node("node1", "NVIDIA-A100-SXM4-40GB"),
            _make_node("node2", "NVIDIA-A100-40GB-PCIe"),
        ]
        result = detect_cluster_gpus()
        assert result == ["A100-40"]

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_cached_result_avoids_repeated_api_calls(self, mock_list, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [_make_node("node1", "NVIDIA-L4")]
        detect_cluster_gpus()
        detect_cluster_gpus()
        # Second call should use cache, not call list_nodes again
        assert mock_list.call_count == 1
