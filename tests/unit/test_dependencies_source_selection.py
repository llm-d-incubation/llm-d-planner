"""Test benchmark source selection and app state initialization."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import planner.api.dependencies as deps


@pytest.mark.unit
@patch.dict("os.environ", {}, clear=False)
def test_default_source_is_database():
    """When PLANNER_BENCHMARK_SOURCE is not set, default to database."""
    import os

    os.environ.pop("PLANNER_BENCHMARK_SOURCE", None)
    assert deps._get_benchmark_source_type() == "database"


@pytest.mark.unit
@patch.dict("os.environ", {"PLANNER_BENCHMARK_SOURCE": "database"}, clear=False)
def test_explicit_database_source():
    """When PLANNER_BENCHMARK_SOURCE=database, return database."""
    assert deps._get_benchmark_source_type() == "database"


@pytest.mark.unit
@patch.dict("os.environ", {"PLANNER_BENCHMARK_SOURCE": "model_catalog"}, clear=False)
def test_model_catalog_source():
    """When PLANNER_BENCHMARK_SOURCE=model_catalog, return model_catalog."""
    assert deps._get_benchmark_source_type() == "model_catalog"


@pytest.mark.unit
@patch.dict("os.environ", {"PLANNER_BENCHMARK_SOURCE": " Model_Catalog "}, clear=False)
def test_benchmark_source_normalization():
    """Whitespace and case in PLANNER_BENCHMARK_SOURCE are normalized."""
    assert deps._get_benchmark_source_type() == "model_catalog"


@pytest.mark.unit
@patch.dict("os.environ", {"PLANNER_BENCHMARK_SOURCE": "invalid_source"}, clear=False)
def test_unknown_benchmark_source_defaults_to_database():
    """Unknown PLANNER_BENCHMARK_SOURCE values default to database."""
    assert deps._get_benchmark_source_type() == "database"


def _make_mock_app():
    """Create a mock FastAPI app with a state namespace."""
    app = MagicMock()
    app.state = SimpleNamespace()
    return app


@pytest.mark.unit
@patch.dict("os.environ", {"PLANNER_BENCHMARK_SOURCE": "model_catalog"}, clear=False)
def test_model_catalog_mode_creates_client_and_syncs():
    """When source is model_catalog, init_app_state() creates client and starts sync."""
    app = _make_mock_app()
    mock_engine = MagicMock()
    mock_weights = {"chatbot": {"categories": {"overall": 5}}}
    with (
        patch("planner.knowledge_base.model_catalog_client.ModelCatalogClient") as mock_client_cls,
        patch("planner.api.dependencies._sync_model_catalog_async") as mock_sync,
        patch("planner.api.dependencies.RecommendationWorkflow") as mock_wf_cls,
        patch("planner.api.dependencies.ModelCatalog") as mock_mc,
        patch("planner.api.dependencies.UseCaseRepository"),
        patch("planner.api.dependencies.DeploymentGenerator"),
        patch("planner.api.dependencies.YAMLValidator"),
        patch(
            "planner.recommendation.quality.scoring.build_scoring_engine",
            return_value=(mock_engine, {}),
        ),
        patch(
            "planner.recommendation.quality.scoring.load_quality_weights", return_value=mock_weights
        ),
        patch("planner.recommendation.quality.scoring.validate_quality_weights"),
        patch("planner.recommendation.config_finder.ConfigFinder") as mock_cf_cls,
        patch("planner.knowledge_base.benchmarks.BenchmarkRepository") as mock_repo_cls,
    ):
        deps.init_app_state(app)

        mock_client_cls.assert_called_once()

        mock_cf_cls.assert_called_once_with(
            benchmark_repo=mock_repo_cls.return_value,
            catalog=mock_mc.return_value,
            engine=mock_engine,
            quality_weights=mock_weights,
        )
        mock_wf_cls.assert_called_once_with(config_finder=mock_cf_cls.return_value)

        mock_sync.assert_called_once()
        assert app.state.model_catalog_sync_thread == mock_sync.return_value
        assert app.state.model_catalog == mock_mc.return_value
        assert app.state.model_catalog_client == mock_client_cls.return_value


@pytest.mark.unit
@patch.dict("os.environ", {"PLANNER_BENCHMARK_SOURCE": "database"}, clear=False)
def test_database_workflow_uses_defaults():
    """When source is database, init_app_state() creates RecommendationWorkflow with shared catalog."""
    app = _make_mock_app()
    mock_engine = MagicMock()
    mock_weights = {"chatbot": {"categories": {"overall": 5}}}
    with (
        patch("planner.api.dependencies.RecommendationWorkflow") as mock_wf_cls,
        patch("planner.api.dependencies.ModelCatalog") as mock_mc,
        patch("planner.api.dependencies.UseCaseRepository"),
        patch("planner.api.dependencies.DeploymentGenerator"),
        patch("planner.api.dependencies.YAMLValidator"),
        patch(
            "planner.recommendation.quality.scoring.build_scoring_engine",
            return_value=(mock_engine, {}),
        ),
        patch(
            "planner.recommendation.quality.scoring.load_quality_weights", return_value=mock_weights
        ),
        patch("planner.recommendation.quality.scoring.validate_quality_weights"),
        patch("planner.recommendation.config_finder.ConfigFinder") as mock_cf_cls,
        patch("planner.knowledge_base.benchmarks.BenchmarkRepository") as mock_repo_cls,
    ):
        deps.init_app_state(app)
        mock_cf_cls.assert_called_once_with(
            benchmark_repo=mock_repo_cls.return_value,
            catalog=mock_mc.return_value,
            engine=mock_engine,
            quality_weights=mock_weights,
        )
        mock_wf_cls.assert_called_once_with(config_finder=mock_cf_cls.return_value)
        assert app.state.workflow == mock_wf_cls.return_value
        assert app.state.model_catalog_client is None


@pytest.mark.unit
@patch.dict("os.environ", {"PLANNER_BENCHMARK_SOURCE": "database"}, clear=False)
def test_init_app_state_sets_all_singletons():
    """init_app_state() populates all expected attributes on app.state."""
    app = _make_mock_app()
    with (
        patch("planner.api.dependencies.RecommendationWorkflow"),
        patch("planner.api.dependencies.ModelCatalog") as mock_mc,
        patch("planner.api.dependencies.UseCaseRepository") as mock_ucr,
        patch("planner.api.dependencies.DeploymentGenerator") as mock_dg,
        patch("planner.api.dependencies.YAMLValidator") as mock_yv,
        patch(
            "planner.recommendation.quality.scoring.build_scoring_engine",
            return_value=(MagicMock(), {}),
        ),
        patch("planner.recommendation.quality.scoring.load_quality_weights", return_value={}),
        patch("planner.recommendation.quality.scoring.validate_quality_weights"),
        patch("planner.recommendation.config_finder.ConfigFinder"),
        patch("planner.knowledge_base.benchmarks.BenchmarkRepository"),
    ):
        deps.init_app_state(app)

        assert app.state.model_catalog == mock_mc.return_value
        assert app.state.use_case_repo == mock_ucr.return_value
        assert app.state.deployment_generator == mock_dg.return_value
        assert app.state.yaml_validator == mock_yv.return_value
        assert app.state.cluster_managers == {}
