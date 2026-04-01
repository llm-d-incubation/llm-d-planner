"""Tests for model_catalog_sync ETL module."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest


def _perf_artifact(
    model_id="RedHatAI/test-model",
    hardware="H100",
    hw_count=4,
    ttft_p95=80.0,
    itl_p95=20.0,
    e2e_p95=5000.0,
    rps=7.0,
    prompt_tokens=512,
    output_tokens=256,
):
    """Build a fake performance-metrics artifact dict."""
    return {
        "artifactType": "metrics-artifact",
        "metricsType": "performance-metrics",
        "customProperties": {
            "model_id": {"string_value": model_id},
            "hardware_type": {"string_value": hardware},
            "hardware_count": {"int_value": hw_count},
            "ttft_mean": {"double_value": ttft_p95 * 0.7},
            "ttft_p90": {"double_value": ttft_p95 * 0.9},
            "ttft_p95": {"double_value": ttft_p95},
            "ttft_p99": {"double_value": ttft_p95 * 1.3},
            "itl_mean": {"double_value": itl_p95 * 0.7},
            "itl_p90": {"double_value": itl_p95 * 0.9},
            "itl_p95": {"double_value": itl_p95},
            "itl_p99": {"double_value": itl_p95 * 1.3},
            "e2e_mean": {"double_value": e2e_p95 * 0.7},
            "e2e_p90": {"double_value": e2e_p95 * 0.9},
            "e2e_p95": {"double_value": e2e_p95},
            "e2e_p99": {"double_value": e2e_p95 * 1.3},
            "tps_mean": {"double_value": 1500.0},
            "tps_p90": {"double_value": 1200.0},
            "tps_p95": {"double_value": 1000.0},
            "tps_p99": {"double_value": 800.0},
            "requests_per_second": {"double_value": rps},
            "mean_input_tokens": {"double_value": float(prompt_tokens)},
            "mean_output_tokens": {"double_value": float(output_tokens)},
            "framework_type": {"string_value": "vllm"},
            "framework_version": {"string_value": "v0.8.4"},
            "profiler_config": {
                "string_value": '{"args": {"prompt_tokens": '
                + str(prompt_tokens)
                + ', "output_tokens": '
                + str(output_tokens)
                + "}}",
            },
        },
    }


def _accuracy_artifact(overall_average=72.5):
    """Build a fake accuracy-metrics artifact dict."""
    return {
        "artifactType": "metrics-artifact",
        "metricsType": "accuracy-metrics",
        "customProperties": {
            "overall_average": {"double_value": overall_average},
        },
    }


def _model_dict(
    name="RedHatAI/granite-3.1-8b-instruct",
    provider="IBM",
    tasks=None,
    license_str="Apache 2.0",
    size="8B params",
    validated=True,
):
    """Build a fake model dict from the catalog API."""
    props = {
        "size": {"string_value": size},
    }
    if validated:
        props["validated_on"] = {"string_value": '["RHOAI 2.20"]'}
    return {
        "name": name,
        "provider": provider,
        "tasks": tasks or ["text-to-text"],
        "license": license_str,
        "source_id": "redhat_ai_validated_models",
        "customProperties": props,
    }


# ---------------------------------------------------------------------------
# _artifact_to_row tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestArtifactToRow:
    """Tests for _artifact_to_row mapping."""

    def test_maps_fields_correctly(self):
        from planner.knowledge_base.model_catalog_sync import _artifact_to_row

        artifact = _perf_artifact(
            model_id="RedHatAI/test-model",
            hardware="H100",
            hw_count=4,
            ttft_p95=80.0,
            prompt_tokens=512,
            output_tokens=256,
        )
        row = _artifact_to_row(artifact)

        assert row is not None
        assert row["model_hf_repo"] == "RedHatAI/test-model"
        assert row["hardware"] == "H100"
        assert row["hardware_count"] == 4
        assert row["ttft_p95"] == 80.0
        assert row["prompt_tokens"] == 512
        assert row["output_tokens"] == 256
        assert row["source"] == "model_catalog"

    def test_generates_id_and_config_id(self):
        from planner.knowledge_base.model_catalog_sync import _artifact_to_row

        row = _artifact_to_row(_perf_artifact())
        assert row is not None
        # id is a UUID string
        assert len(row["id"]) == 36
        assert "-" in row["id"]
        # config_id is an MD5 hex digest
        assert len(row["config_id"]) == 32

    def test_returns_none_for_missing_model_id(self):
        from planner.knowledge_base.model_catalog_sync import _artifact_to_row

        artifact = _perf_artifact()
        del artifact["customProperties"]["model_id"]
        assert _artifact_to_row(artifact) is None

    def test_returns_none_for_missing_hardware(self):
        from planner.knowledge_base.model_catalog_sync import _artifact_to_row

        artifact = _perf_artifact()
        del artifact["customProperties"]["hardware_type"]
        assert _artifact_to_row(artifact) is None

    def test_returns_none_for_zero_prompt_tokens(self):
        from planner.knowledge_base.model_catalog_sync import _artifact_to_row

        artifact = _perf_artifact(prompt_tokens=0, output_tokens=256)
        # profiler_config will have prompt_tokens=0
        assert _artifact_to_row(artifact) is None

    def test_returns_none_for_zero_rps(self):
        from planner.knowledge_base.model_catalog_sync import _artifact_to_row

        artifact = _perf_artifact(rps=0.0)
        assert _artifact_to_row(artifact) is None

    def test_returns_none_for_zero_hardware_count(self):
        from planner.knowledge_base.model_catalog_sync import _artifact_to_row

        artifact = _perf_artifact(hw_count=0)
        assert _artifact_to_row(artifact) is None

    def test_all_latency_fields_present(self):
        from planner.knowledge_base.model_catalog_sync import _artifact_to_row

        row = _artifact_to_row(_perf_artifact(ttft_p95=100.0, itl_p95=25.0, e2e_p95=6000.0))
        assert row is not None
        assert row["ttft_mean"] == pytest.approx(70.0)
        assert row["ttft_p90"] == pytest.approx(90.0)
        assert row["ttft_p95"] == pytest.approx(100.0)
        assert row["ttft_p99"] == pytest.approx(130.0)
        assert row["itl_mean"] == pytest.approx(17.5)
        assert row["itl_p95"] == pytest.approx(25.0)
        assert row["e2e_p95"] == pytest.approx(6000.0)

    def test_timestamps_and_defaults(self):
        from planner.knowledge_base.model_catalog_sync import _artifact_to_row

        row = _artifact_to_row(_perf_artifact())
        assert row is not None
        assert row["type"] == "model_catalog"
        assert row["created_at"] is not None
        assert row["updated_at"] is not None
        assert row["framework"] == "vllm"


# ---------------------------------------------------------------------------
# _catalog_model_to_model_info tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCatalogModelToModelInfo:
    """Tests for _catalog_model_to_model_info mapping."""

    def test_maps_correctly(self):
        from planner.knowledge_base.model_catalog_sync import (
            _catalog_model_to_model_info,
        )

        model = _model_dict()
        info = _catalog_model_to_model_info(model)

        assert info.model_id == "RedHatAI/granite-3.1-8b-instruct"
        assert info.family == "granite"
        assert info.approval_status == "approved"
        assert "chatbot_conversational" in info.supported_tasks

    def test_unvalidated_model_pending(self):
        from planner.knowledge_base.model_catalog_sync import (
            _catalog_model_to_model_info,
        )

        model = _model_dict(validated=False)
        info = _catalog_model_to_model_info(model)
        assert info.approval_status == "pending"

    def test_size_parsing(self):
        from planner.knowledge_base.model_catalog_sync import (
            _catalog_model_to_model_info,
        )

        model = _model_dict(size="70B params")
        info = _catalog_model_to_model_info(model)
        assert info.size_parameters == "70B"

    def test_license_type_permissive(self):
        from planner.knowledge_base.model_catalog_sync import (
            _catalog_model_to_model_info,
        )

        model = _model_dict(license_str="Apache 2.0")
        info = _catalog_model_to_model_info(model)
        assert info.license_type == "permissive"

    def test_license_type_restricted(self):
        from planner.knowledge_base.model_catalog_sync import (
            _catalog_model_to_model_info,
        )

        model = _model_dict(license_str="proprietary")
        info = _catalog_model_to_model_info(model)
        assert info.license_type == "restricted"


# ---------------------------------------------------------------------------
# sync_model_catalog tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSyncModelCatalog:
    """Tests for the sync_model_catalog ETL function."""

    def _build_mocks(self):
        """Build mock client, connection, model_catalog, and quality_scorer."""
        client = MagicMock()
        client.list_models.return_value = [
            _model_dict(name="RedHatAI/granite-3.1-8b-instruct"),
        ]
        client.get_model_artifacts.return_value = [
            _perf_artifact(model_id="RedHatAI/granite-3.1-8b-instruct"),
            _accuracy_artifact(overall_average=72.5),
        ]

        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        model_catalog = MagicMock()
        model_catalog.merge_external_models.return_value = 1

        quality_scorer = MagicMock()

        return client, conn, cursor, model_catalog, quality_scorer

    @patch("planner.knowledge_base.model_catalog_sync.execute_batch")
    def test_full_flow(self, mock_execute_batch):
        from planner.knowledge_base.model_catalog_sync import sync_model_catalog

        client, conn, cursor, model_catalog, quality_scorer = self._build_mocks()

        result = sync_model_catalog(client, conn, model_catalog, quality_scorer)

        # DELETE was called for model_catalog source
        delete_calls = [c for c in cursor.execute.call_args_list if "DELETE" in str(c)]
        assert len(delete_calls) == 1
        assert "model_catalog" in str(delete_calls[0])

        # Benchmark rows were inserted via execute_batch
        mock_execute_batch.assert_called_once()
        inserted_rows = mock_execute_batch.call_args[0][2]
        assert len(inserted_rows) == 1
        assert inserted_rows[0]["model_hf_repo"] == "RedHatAI/granite-3.1-8b-instruct"

        # merge_external_models was called with ModelInfo list
        model_catalog.merge_external_models.assert_called_once()
        merged_models = model_catalog.merge_external_models.call_args[0][0]
        assert len(merged_models) == 1
        assert merged_models[0].model_id == "RedHatAI/granite-3.1-8b-instruct"

        # set_catalog_fallback was called with scores dict
        quality_scorer.set_catalog_fallback.assert_called_once()
        scores = quality_scorer.set_catalog_fallback.call_args[0][0]
        assert "redhatai/granite-3.1-8b-instruct" in scores
        assert scores["redhatai/granite-3.1-8b-instruct"] == pytest.approx(72.5)

        # SyncResult fields
        assert result.benchmarks_inserted == 1
        assert result.models_merged == 1
        assert result.quality_scores_loaded == 1
        assert result.errors == []

    @patch("planner.knowledge_base.model_catalog_sync.execute_batch")
    def test_transaction_committed(self, mock_execute_batch):
        from planner.knowledge_base.model_catalog_sync import sync_model_catalog

        client, conn, cursor, model_catalog, quality_scorer = self._build_mocks()
        sync_model_catalog(client, conn, model_catalog, quality_scorer)

        conn.commit.assert_called()

    @patch("planner.knowledge_base.model_catalog_sync.execute_batch")
    def test_db_error_rolls_back(self, mock_execute_batch):
        from planner.knowledge_base.model_catalog_sync import sync_model_catalog

        client, conn, cursor, model_catalog, quality_scorer = self._build_mocks()
        mock_execute_batch.side_effect = Exception("DB error")

        result = sync_model_catalog(client, conn, model_catalog, quality_scorer)

        conn.rollback.assert_called_once()
        assert result.benchmarks_inserted == 0
        assert len(result.errors) > 0

    @patch("planner.knowledge_base.model_catalog_sync.execute_batch")
    def test_skips_malformed_artifacts(self, mock_execute_batch):
        from planner.knowledge_base.model_catalog_sync import sync_model_catalog

        client, conn, cursor, model_catalog, quality_scorer = self._build_mocks()

        # Add a malformed artifact (missing model_id)
        bad_artifact = _perf_artifact()
        del bad_artifact["customProperties"]["model_id"]

        client.get_model_artifacts.return_value = [
            bad_artifact,
            _perf_artifact(model_id="RedHatAI/granite-3.1-8b-instruct"),
            _accuracy_artifact(overall_average=72.5),
        ]

        result = sync_model_catalog(client, conn, model_catalog, quality_scorer)
        # Only the valid artifact should be inserted
        assert result.benchmarks_inserted == 1

    @patch("planner.knowledge_base.model_catalog_sync.execute_batch")
    def test_client_list_failure(self, mock_execute_batch):
        from planner.knowledge_base.model_catalog_sync import sync_model_catalog

        client, conn, cursor, model_catalog, quality_scorer = self._build_mocks()
        client.list_models.side_effect = Exception("Network error")

        result = sync_model_catalog(client, conn, model_catalog, quality_scorer)

        assert result.benchmarks_inserted == 0
        assert result.models_merged == 0
        assert len(result.errors) > 0
        # No DB writes should happen
        mock_execute_batch.assert_not_called()
