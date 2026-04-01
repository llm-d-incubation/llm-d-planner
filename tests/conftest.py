"""Pytest configuration and shared fixtures.

Sets up a dedicated neuralnav_test database for database-marked tests,
loading a small static fixture dataset. The user's production neuralnav
database is never touched.
"""

import json
import logging
from pathlib import Path

import psycopg2
import pytest
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from planner.knowledge_base.loader import insert_benchmarks

logger = logging.getLogger(__name__)

# PostgreSQL connection for the server (not a specific DB)
PG_HOST = "localhost"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "neuralnav"

TEST_DB_NAME = "neuralnav_test"
TEST_DB_URL = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{TEST_DB_NAME}"

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SCHEMA_PATH = Path(__file__).parent.parent / "scripts" / "schema.sql"


def _create_test_database():
    """Create the neuralnav_test database."""
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, dbname="postgres"
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    # Drop if exists from a previous failed run
    cursor.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")
    cursor.execute(f"CREATE DATABASE {TEST_DB_NAME}")

    cursor.close()
    conn.close()


def _drop_test_database():
    """Drop the neuralnav_test database, terminating any active connections first."""
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, dbname="postgres"
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    # Terminate active connections to avoid DROP DATABASE failures
    cursor.execute(
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        f"WHERE datname = '{TEST_DB_NAME}' AND pid <> pg_backend_pid()"
    )
    cursor.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")

    cursor.close()
    conn.close()


def _apply_schema():
    """Apply the database schema to the test database."""
    conn = psycopg2.connect(TEST_DB_URL)
    cursor = conn.cursor()

    schema_sql = SCHEMA_PATH.read_text()
    cursor.execute(schema_sql)
    conn.commit()

    cursor.close()
    conn.close()


def _load_test_fixtures():
    """Load test benchmark data into the test database."""
    fixture_path = FIXTURES_DIR / "test_benchmarks.json"
    with open(fixture_path) as f:
        data = json.load(f)

    conn = psycopg2.connect(TEST_DB_URL)
    insert_benchmarks(conn, data["benchmarks"])
    conn.close()


@pytest.fixture(scope="session")
def test_db_url():
    """Create and populate the test database for the entire test session.

    Yields the database URL, then drops the test database on teardown.
    """
    _create_test_database()
    try:
        _apply_schema()
        _load_test_fixtures()
        yield TEST_DB_URL
    finally:
        _drop_test_database()
