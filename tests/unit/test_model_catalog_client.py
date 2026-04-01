"""Tests for Model Catalog HTTP client."""

from unittest.mock import MagicMock, patch

import pytest

from planner.knowledge_base.model_catalog_client import ModelCatalogClient


@pytest.fixture
def client():
    return ModelCatalogClient(
        base_url="https://localhost:8443",
        token="test-token",
        source_id="redhat_ai_validated_models",
        verify_ssl=False,
    )


FAKE_MODELS_RESPONSE = {
    "items": [
        {
            "name": "RedHatAI/granite-3.1-8b-instruct",
            "provider": "IBM",
            "description": "Granite model",
            "tasks": ["text-to-text"],
            "customProperties": {
                "size": {
                    "string_value": "8B params",
                    "metadataType": "MetadataStringValue",
                },
                "validated": {
                    "string_value": "",
                    "metadataType": "MetadataStringValue",
                },
            },
        }
    ],
    "size": 1,
    "pageSize": 100,
    "nextPageToken": "",
}

FAKE_ARTIFACTS_RESPONSE = {
    "items": [
        {
            "artifactType": "metrics-artifact",
            "metricsType": "performance-metrics",
            "customProperties": {
                "hardware_type": {
                    "string_value": "H100",
                    "metadataType": "MetadataStringValue",
                },
                "hardware_count": {
                    "int_value": 4,
                    "metadataType": "MetadataIntValue",
                },
                "ttft_p95": {
                    "double_value": 98.15,
                    "metadataType": "MetadataDoubleValue",
                },
                "requests_per_second": {
                    "double_value": 7.0,
                    "metadataType": "MetadataDoubleValue",
                },
            },
        },
        {
            "artifactType": "metrics-artifact",
            "metricsType": "accuracy-metrics",
            "customProperties": {
                "overall_average": {
                    "double_value": 57.67,
                    "metadataType": "MetadataDoubleValue",
                },
                "mmlu": {
                    "double_value": 82.04,
                    "metadataType": "MetadataDoubleValue",
                },
            },
        },
    ],
    "size": 2,
    "pageSize": 200,
    "nextPageToken": "",
}


@pytest.mark.unit
@patch("planner.knowledge_base.model_catalog_client.httpx")
def test_list_models(mock_httpx, client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = FAKE_MODELS_RESPONSE
    mock_response.raise_for_status = MagicMock()

    mock_http = MagicMock()
    mock_http.get.return_value = mock_response
    mock_http.is_closed = False
    mock_httpx.Client.return_value = mock_http

    models = client.list_models()
    assert len(models) == 1
    assert models[0]["name"] == "RedHatAI/granite-3.1-8b-instruct"


@pytest.mark.unit
@patch("planner.knowledge_base.model_catalog_client.httpx")
def test_get_artifacts(mock_httpx, client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = FAKE_ARTIFACTS_RESPONSE
    mock_response.raise_for_status = MagicMock()

    mock_http = MagicMock()
    mock_http.get.return_value = mock_response
    mock_http.is_closed = False
    mock_httpx.Client.return_value = mock_http

    artifacts = client.get_model_artifacts("RedHatAI/granite-3.1-8b-instruct")
    perf = [a for a in artifacts if a.get("metricsType") == "performance-metrics"]
    acc = [a for a in artifacts if a.get("metricsType") == "accuracy-metrics"]
    assert len(perf) == 1
    assert len(acc) == 1
