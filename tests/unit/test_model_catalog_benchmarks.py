"""Tests for ModelCatalogBenchmarkSource."""

from unittest.mock import MagicMock

import pytest

from neuralnav.knowledge_base.model_catalog_benchmarks import ModelCatalogBenchmarkSource


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
            "model_id": {"string_value": model_id, "metadataType": "MetadataStringValue"},
            "hardware_type": {"string_value": hardware, "metadataType": "MetadataStringValue"},
            "hardware_configuration": {
                "string_value": f"{hardware} x {hw_count}",
                "metadataType": "MetadataStringValue",
            },
            "hardware_count": {"int_value": hw_count, "metadataType": "MetadataIntValue"},
            "ttft_mean": {"double_value": ttft_p95 * 0.7, "metadataType": "MetadataDoubleValue"},
            "ttft_p90": {"double_value": ttft_p95 * 0.9, "metadataType": "MetadataDoubleValue"},
            "ttft_p95": {"double_value": ttft_p95, "metadataType": "MetadataDoubleValue"},
            "ttft_p99": {"double_value": ttft_p95 * 1.3, "metadataType": "MetadataDoubleValue"},
            "itl_mean": {"double_value": itl_p95 * 0.7, "metadataType": "MetadataDoubleValue"},
            "itl_p90": {"double_value": itl_p95 * 0.9, "metadataType": "MetadataDoubleValue"},
            "itl_p95": {"double_value": itl_p95, "metadataType": "MetadataDoubleValue"},
            "itl_p99": {"double_value": itl_p95 * 1.3, "metadataType": "MetadataDoubleValue"},
            "e2e_mean": {"double_value": e2e_p95 * 0.7, "metadataType": "MetadataDoubleValue"},
            "e2e_p90": {"double_value": e2e_p95 * 0.9, "metadataType": "MetadataDoubleValue"},
            "e2e_p95": {"double_value": e2e_p95, "metadataType": "MetadataDoubleValue"},
            "e2e_p99": {"double_value": e2e_p95 * 1.3, "metadataType": "MetadataDoubleValue"},
            "tps_mean": {"double_value": 1500.0, "metadataType": "MetadataDoubleValue"},
            "tps_p90": {"double_value": 1200.0, "metadataType": "MetadataDoubleValue"},
            "tps_p95": {"double_value": 1000.0, "metadataType": "MetadataDoubleValue"},
            "tps_p99": {"double_value": 800.0, "metadataType": "MetadataDoubleValue"},
            "requests_per_second": {"double_value": rps, "metadataType": "MetadataDoubleValue"},
            "mean_input_tokens": {
                "double_value": float(prompt_tokens),
                "metadataType": "MetadataDoubleValue",
            },
            "mean_output_tokens": {
                "double_value": float(output_tokens),
                "metadataType": "MetadataDoubleValue",
            },
            "framework_type": {"string_value": "vllm", "metadataType": "MetadataStringValue"},
            "framework_version": {"string_value": "v0.8.4", "metadataType": "MetadataStringValue"},
            "use_case": {"string_value": "chatbot", "metadataType": "MetadataStringValue"},
            "profiler_config": {
                "string_value": '{"args": {"prompt_tokens": '
                + str(prompt_tokens)
                + ', "output_tokens": '
                + str(output_tokens)
                + "}}",
                "metadataType": "MetadataStringValue",
            },
        },
    }


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.list_models.return_value = [
        {"name": "RedHatAI/test-model", "source_id": "redhat_ai_validated_models"},
        {"name": "RedHatAI/other-model", "source_id": "redhat_ai_validated_models"},
    ]
    client.get_model_artifacts.side_effect = lambda name: {
        "RedHatAI/test-model": [
            _perf_artifact(
                model_id="RedHatAI/test-model",
                hardware="H100",
                hw_count=4,
                ttft_p95=80,
                itl_p95=20,
                e2e_p95=5000,
                rps=7,
            ),
            _perf_artifact(
                model_id="RedHatAI/test-model",
                hardware="L4",
                hw_count=1,
                ttft_p95=300,
                itl_p95=50,
                e2e_p95=15000,
                rps=2,
            ),
        ],
        "RedHatAI/other-model": [
            _perf_artifact(
                model_id="RedHatAI/other-model",
                hardware="H100",
                hw_count=2,
                ttft_p95=100,
                itl_p95=25,
                e2e_p95=6000,
                rps=5,
            ),
        ],
    }.get(name, [])
    return client


@pytest.mark.unit
def test_find_configs_meeting_slo(mock_client):
    source = ModelCatalogBenchmarkSource(mock_client)
    results = source.find_configurations_meeting_slo(
        prompt_tokens=512,
        output_tokens=256,
        ttft_p95_max_ms=200,
        itl_p95_max_ms=30,
        e2e_p95_max_ms=8000,
    )
    # H100x4 (ttft=80, itl=20, e2e=5000) and H100x2 (ttft=100, itl=25, e2e=6000) meet SLO
    # L4x1 (ttft=300) does NOT meet TTFT SLO of 200ms
    assert len(results) == 2
    hardware_configs = {(r.hardware, r.hardware_count) for r in results}
    assert ("H100", 4) in hardware_configs
    assert ("H100", 2) in hardware_configs


@pytest.mark.unit
def test_find_configs_gpu_filter(mock_client):
    source = ModelCatalogBenchmarkSource(mock_client)
    results = source.find_configurations_meeting_slo(
        prompt_tokens=512,
        output_tokens=256,
        ttft_p95_max_ms=500,
        itl_p95_max_ms=60,
        e2e_p95_max_ms=20000,
        gpu_types=["L4"],
    )
    assert len(results) == 1
    assert results[0].hardware == "L4"


@pytest.mark.unit
def test_find_configs_no_match(mock_client):
    source = ModelCatalogBenchmarkSource(mock_client)
    results = source.find_configurations_meeting_slo(
        prompt_tokens=512,
        output_tokens=256,
        ttft_p95_max_ms=10,
        itl_p95_max_ms=5,
        e2e_p95_max_ms=100,
    )
    assert results == []


@pytest.mark.unit
def test_benchmark_data_fields(mock_client):
    """Verify mapped BenchmarkData has all required fields."""
    source = ModelCatalogBenchmarkSource(mock_client)
    results = source.find_configurations_meeting_slo(
        prompt_tokens=512,
        output_tokens=256,
        ttft_p95_max_ms=200,
        itl_p95_max_ms=30,
        e2e_p95_max_ms=8000,
    )
    bench = results[0]
    assert bench.model_hf_repo is not None
    assert bench.hardware is not None
    assert bench.hardware_count > 0
    assert bench.ttft_p95 > 0
    assert bench.itl_p95 > 0
    assert bench.e2e_p95 > 0
    assert bench.requests_per_second > 0
    assert bench.tokens_per_second > 0
    assert bench.prompt_tokens == 512
    assert bench.output_tokens == 256
    assert bench.estimated is False
