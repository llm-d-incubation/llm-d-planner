"""Test benchmark source selection via environment variable."""

from unittest.mock import patch

import pytest


@pytest.mark.unit
@patch.dict("os.environ", {}, clear=False)
def test_default_source_is_postgresql():
    """When NEURALNAV_BENCHMARK_SOURCE is not set, default to postgresql."""
    import os

    os.environ.pop("NEURALNAV_BENCHMARK_SOURCE", None)
    from neuralnav.api.dependencies import _get_benchmark_source_type

    assert _get_benchmark_source_type() == "postgresql"


@pytest.mark.unit
@patch.dict("os.environ", {"NEURALNAV_BENCHMARK_SOURCE": "postgresql"}, clear=False)
def test_explicit_postgresql_source():
    """When NEURALNAV_BENCHMARK_SOURCE=postgresql, return postgresql."""
    from neuralnav.api.dependencies import _get_benchmark_source_type

    assert _get_benchmark_source_type() == "postgresql"


@pytest.mark.unit
@patch.dict("os.environ", {"NEURALNAV_BENCHMARK_SOURCE": "model_catalog"}, clear=False)
def test_model_catalog_source():
    """When NEURALNAV_BENCHMARK_SOURCE=model_catalog, return model_catalog."""
    from neuralnav.api.dependencies import _get_benchmark_source_type

    assert _get_benchmark_source_type() == "model_catalog"


@pytest.mark.unit
@patch.dict("os.environ", {"NEURALNAV_BENCHMARK_SOURCE": "model_catalog"}, clear=False)
def test_model_catalog_workflow_creates_correct_components():
    """When source is model_catalog, get_workflow() should wire up Model Catalog components."""
    import neuralnav.api.dependencies as deps

    # Reset singleton so get_workflow() re-creates it
    deps._workflow = None

    # Patch at source modules since get_workflow() uses local imports
    with (
        patch(
            "neuralnav.knowledge_base.model_catalog_client.ModelCatalogClient"
        ) as mock_client_cls,
        patch(
            "neuralnav.knowledge_base.model_catalog_benchmarks.ModelCatalogBenchmarkSource"
        ) as mock_bench_src_cls,
        patch(
            "neuralnav.knowledge_base.model_catalog_models.ModelCatalogModelSource"
        ) as mock_model_src_cls,
        patch(
            "neuralnav.knowledge_base.model_catalog_quality.ModelCatalogQualityScorer"
        ) as mock_quality_cls,
        patch("neuralnav.recommendation.config_finder.ConfigFinder") as mock_finder_cls,
        patch("neuralnav.api.dependencies.RecommendationWorkflow") as mock_wf_cls,
    ):
        deps.get_workflow()

        # Verify client created once
        mock_client_cls.assert_called_once()
        client_instance = mock_client_cls.return_value

        # Verify all Model Catalog components wired with the same client
        mock_bench_src_cls.assert_called_once_with(client_instance)
        mock_model_src_cls.assert_called_once_with(client_instance)
        mock_quality_cls.assert_called_once_with(client_instance)

        # Verify ConfigFinder created with all three components
        mock_finder_cls.assert_called_once_with(
            benchmark_repo=mock_bench_src_cls.return_value,
            catalog=mock_model_src_cls.return_value,
            quality_scorer=mock_quality_cls.return_value,
        )

        # Verify workflow created with the custom config_finder
        mock_wf_cls.assert_called_once_with(config_finder=mock_finder_cls.return_value)

    # Clean up singleton
    deps._workflow = None


@pytest.mark.unit
@patch.dict("os.environ", {"NEURALNAV_BENCHMARK_SOURCE": "postgresql"}, clear=False)
def test_postgresql_workflow_uses_defaults():
    """When source is postgresql, get_workflow() creates default RecommendationWorkflow."""
    import neuralnav.api.dependencies as deps

    # Reset singleton
    deps._workflow = None

    with patch("neuralnav.api.dependencies.RecommendationWorkflow") as mock_wf_cls:
        deps.get_workflow()
        mock_wf_cls.assert_called_once_with()

    # Clean up singleton
    deps._workflow = None
