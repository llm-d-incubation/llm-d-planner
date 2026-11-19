"""Reference-data fetches recover without clearing Streamlit's cache."""

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

_spec = importlib.util.spec_from_file_location(
    "api_client", Path(__file__).resolve().parents[2] / "ui" / "api_client.py"
)
assert _spec and _spec.loader
api_client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(api_client)

pytestmark = pytest.mark.unit

FETCH_CASES: list[tuple[str, tuple, dict, Any, Any]] = [
    (
        "fetch_slo_defaults",
        ("chatbot_conversational",),
        {"success": True, "slo_defaults": {"ttft_ms": {"default": 300}}},
        {"ttft_ms": {"default": 300}},
        None,
    ),
    (
        "fetch_expected_rps",
        ("chatbot_conversational", 100),
        {"success": True, "expected_rps": 2.5},
        {"success": True, "expected_rps": 2.5},
        None,
    ),
    (
        "fetch_workload_profile",
        ("chatbot_conversational",),
        {"success": True, "workload_profile": {"prompt_tokens": 512}},
        {"prompt_tokens": 512},
        None,
    ),
    (
        "fetch_priority_weights",
        (),
        {"success": True, "priority_weights": {"quality": 0.45}},
        {"quality": 0.45},
        None,
    ),
    (
        "fetch_catalog_model_ids",
        (),
        {"models": [{"model_id": "model-b"}, {"model_id": "model-a"}]},
        ["model-a", "model-b"],
        [],
    ),
    (
        "fetch_gpu_types",
        (),
        {"gpu_types": [{"gpu_type": "H100", "memory_gb": 80}]},
        {"H100": {"gpu_type": "H100", "memory_gb": 80}},
        {},
    ),
]


@pytest.fixture(autouse=True)
def clear_reference_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.mark.parametrize("name,args,payload,expected,fallback", FETCH_CASES)
@pytest.mark.parametrize("failure", ["connection", "http", "json"])
def test_failed_fetch_retries_on_next_call(
    monkeypatch, name, args, payload, expected, fallback, failure
):
    success = Mock()
    success.json.return_value = payload
    if failure == "connection":
        failed = requests.exceptions.ConnectionError("backend unavailable")
    else:
        failed = Mock()
        if failure == "http":
            failed.raise_for_status.side_effect = requests.exceptions.HTTPError("503")
        else:
            failed.json.side_effect = requests.exceptions.JSONDecodeError("invalid JSON", "x", 0)
    get = Mock(side_effect=[failed, success])
    warning = Mock()
    monkeypatch.setattr(api_client.requests, "get", get)
    monkeypatch.setattr(api_client.st, "warning", warning)
    fetch = getattr(api_client, name)

    assert fetch(*args) == fallback
    assert fetch(*args) == expected
    assert fetch(*args) == expected
    assert get.call_count == 2
    assert warning.call_count == 1


@pytest.mark.parametrize("name,args,payload,expected,fallback", FETCH_CASES[:4])
def test_unsuccessful_api_result_is_not_cached(
    monkeypatch, name, args, payload, expected, fallback
):
    rejected = Mock()
    rejected.json.return_value = {"success": False}
    success = Mock()
    success.json.return_value = payload
    get = Mock(side_effect=[rejected, success])
    monkeypatch.setattr(api_client.requests, "get", get)
    monkeypatch.setattr(api_client.st, "warning", Mock())
    fetch = getattr(api_client, name)

    assert fetch(*args) == fallback
    assert fetch(*args) == expected
    assert fetch(*args) == expected
    assert get.call_count == 2


@pytest.mark.parametrize("name,args,payload,expected,fallback", FETCH_CASES)
def test_successful_fetch_is_cached(monkeypatch, name, args, payload, expected, fallback):
    response = Mock()
    response.json.return_value = payload
    get = Mock(return_value=response)
    warning = Mock()
    monkeypatch.setattr(api_client.requests, "get", get)
    monkeypatch.setattr(api_client.st, "warning", warning)

    fetch = getattr(api_client, name)
    assert fetch(*args) == expected
    assert fetch(*args) == expected
    get.assert_called_once()
    warning.assert_not_called()


def test_ui_rerun_recovers_after_backend_outage(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        available = False
        calls = 0

        def do_GET(self):
            type(self).calls += 1
            self.send_response(200 if self.available else 503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"gpu_types": [{"gpu_type": "H100"}]}).encode())

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setattr(api_client, "API_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setitem(sys.modules, "api_client", api_client)
    try:
        app = AppTest.from_string(
            "import streamlit as st\n"
            "from api_client import fetch_gpu_types\n"
            "st.json(fetch_gpu_types())\n"
            "st.button('Retry')\n"
        ).run()
        assert not app.exception
        assert len(app.warning) == 1
        assert json.loads(app.json[0].value) == {}

        Handler.available = True
        app.button[0].click().run()
        assert not app.exception
        assert not app.warning
        assert json.loads(app.json[0].value) == {"H100": {"gpu_type": "H100"}}

        app.button[0].click().run()
        assert not app.warning
        assert Handler.calls == 2
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
