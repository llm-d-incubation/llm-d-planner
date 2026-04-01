"""Test ModelCatalog.merge_external_models()."""

import json

import pytest

from planner.knowledge_base.model_catalog import ModelCatalog, ModelInfo


def _make_model_info(model_id: str, name: str = "test", provider: str = "TestOrg") -> ModelInfo:
    return ModelInfo(
        {
            "model_id": model_id,
            "name": name,
            "provider": provider,
            "family": "test",
            "size_parameters": "8B",
            "context_length": 128000,
            "supported_tasks": ["chatbot_conversational"],
            "domain_specialization": [],
            "license": "Apache 2.0",
            "license_type": "permissive",
            "min_gpu_memory_gb": 16,
            "recommended_for": ["chatbot_conversational"],
            "approval_status": "approved",
        }
    )


@pytest.fixture
def catalog(tmp_path):
    """Create a ModelCatalog with minimal JSON data."""
    data = {
        "models": [
            {
                "model_id": "local/model-a",
                "name": "Model A",
                "provider": "Local",
                "family": "test",
                "size_parameters": "8B",
                "context_length": 128000,
                "supported_tasks": ["chatbot_conversational"],
                "domain_specialization": [],
                "license": "Apache 2.0",
                "license_type": "permissive",
                "min_gpu_memory_gb": 16,
                "recommended_for": ["chatbot_conversational"],
                "approval_status": "approved",
            }
        ],
        "gpu_types": [],
    }
    p = tmp_path / "model_catalog.json"
    p.write_text(json.dumps(data))
    return ModelCatalog(data_path=p)


@pytest.mark.unit
def test_merge_adds_new_models(catalog):
    external = _make_model_info("external/model-b", name="Model B", provider="External")
    catalog.merge_external_models([external])
    assert catalog.get_model("external/model-b") is not None
    assert catalog.get_model("external/model-b").provider == "External"


@pytest.mark.unit
def test_merge_does_not_overwrite_local(catalog):
    """Local JSON models take precedence over external models with same ID."""
    external = _make_model_info("local/model-a", name="Overwritten", provider="External")
    catalog.merge_external_models([external])
    assert catalog.get_model("local/model-a").provider == "Local"


@pytest.mark.unit
def test_merge_reflected_in_get_all_models(catalog):
    external = _make_model_info("external/model-c")
    catalog.merge_external_models([external])
    all_ids = [m.model_id for m in catalog.get_all_models()]
    assert "external/model-c" in all_ids
    assert "local/model-a" in all_ids
