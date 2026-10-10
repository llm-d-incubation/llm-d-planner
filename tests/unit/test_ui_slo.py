"""Exercise SLO percentile controls through Streamlit and the API client."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.unit

_SCRIPT = """
import streamlit as st
from api_client import fetch_ranked_recommendations
from components.slo import _render_slo_targets
from state import init_session_state

init_session_state()
_render_slo_targets({
    "ttft_ms": {"min": 100, "max": 500, "default": 200},
    "itl_ms": {"min": 10, "max": 50, "default": 25},
    "e2e_ms": {"min": 1000, "max": 10000, "default": 7000},
})
fetch_ranked_recommendations(
    use_case="chatbot_conversational", user_count=1000,
    prompt_tokens=512, output_tokens=256, expected_qps=1,
    ttft_target_ms=st.session_state.custom_ttft,
    itl_target_ms=st.session_state.custom_itl,
    e2e_target_ms=st.session_state.custom_e2e,
    percentile=st.session_state.slo_percentile,
)
"""


@pytest.fixture
def slo_app(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "ui"))
    post = Mock(return_value=Mock(json=lambda: {}))
    monkeypatch.setattr("requests.post", post)
    return AppTest.from_string(_SCRIPT), post


@pytest.mark.parametrize("percentile", ["p90", "p99"])
def test_selected_percentile_reaches_recommendation_request(slo_app, percentile):
    app, post = slo_app
    app.run()
    assert not app.exception
    app.selectbox[0].select_index(0 if percentile == "p90" else 2).run()
    assert not app.exception
    payload = post.call_args.kwargs["json"]
    assert payload["specification"]["slo_targets"]["percentile"] == percentile
    assert len(app.selectbox) == 1
    assert app.selectbox[0].options == ["P90", "P95", "P99"]


def test_specification_percentile_is_shown_in_selector(slo_app):
    app, _ = slo_app
    app.session_state.slo_percentile = "p99"
    app.run()
    assert not app.exception
    assert app.selectbox[0].value == "p99"


def test_percentile_change_invalidates_cached_recommendations(slo_app):
    app, _ = slo_app
    app.run()
    app.session_state.recommendation_result = {"balanced": []}
    app.selectbox[0].select_index(0).run()
    assert not app.exception
    assert app.session_state.recommendation_result is None


def test_unchanged_percentile_keeps_cached_recommendations(slo_app):
    app, _ = slo_app
    app.run()
    cached: dict[str, list] = {"balanced": []}
    app.session_state.recommendation_result = cached
    app.run()
    assert not app.exception
    assert app.session_state.recommendation_result == cached
