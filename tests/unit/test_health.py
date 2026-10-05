"""Health probes exercise the active benchmark database."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from planner.api.routes.health import router
from planner.knowledge_base.benchmarks import BenchmarkRepository
from planner.knowledge_base.loader import ensure_schema

pytestmark = pytest.mark.unit


@pytest.fixture
def health_app():
    app = FastAPI()
    app.include_router(router)
    with BenchmarkRepository(db_path=":memory:") as repo:
        app.state.benchmark_repo = repo
        yield app


def test_health_accepts_an_empty_database(health_app):
    with TestClient(health_app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "service": "planner"}
        assert health_app.state.benchmark_repo.get_stats()["total_benchmarks"] == 0


def test_health_rejects_a_closed_database(health_app):
    health_app.state.benchmark_repo.close()
    with TestClient(health_app) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json() == {"status": "unhealthy", "service": "planner"}


def test_health_recovers_after_database_schema_is_restored(health_app):
    conn = health_app.state.benchmark_repo._get_connection()
    conn.execute("DROP TABLE exported_summaries")
    with TestClient(health_app) as client:
        assert client.get("/health").status_code == 503
        ensure_schema(conn)
        assert client.get("/health").status_code == 200


def test_health_rejects_uninitialized_app_state():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json() == {"status": "unhealthy", "service": "planner"}
