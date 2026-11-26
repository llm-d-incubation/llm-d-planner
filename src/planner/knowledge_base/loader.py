"""Benchmark data loading utilities.

Provides functions for loading benchmark JSON data into the database.
Used by both the scripts/load_benchmarks.py CLI tool and the
/api/v1/db/* API endpoints.
"""

import hashlib
import logging
import uuid
from contextlib import closing
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_config_id(benchmark: dict) -> str:
    """Generate a deterministic config_id from benchmark configuration."""
    config_str = (
        f"{benchmark['model_hf_repo']}_{benchmark['hardware']}_"
        f"{benchmark['hardware_count']}_{benchmark['prompt_tokens']}_"
        f"{benchmark['output_tokens']}_{benchmark.get('requests_per_second', '')}"
    )
    return hashlib.sha256(config_str.encode()).hexdigest()[:32]


def normalize_benchmark_fields(benchmark: dict) -> dict:
    """Normalize field names from different JSON formats to match DB schema.

    Handles field mapping for:
    - benchmarks_BLIS.json (uses model_hf_repo, hardware)
    - benchmarks_estimated_performance.json (uses model_id, hardware)
    - benchmarks_interpolated_v2.json (uses model_id, hardware_type/gpu_type)
    """
    normalized = benchmark.copy()

    if "model_hf_repo" not in normalized and "model_id" in normalized:
        normalized["model_hf_repo"] = normalized["model_id"]

    if "hardware" not in normalized:
        if "hardware_type" in normalized:
            normalized["hardware"] = normalized["hardware_type"]
        elif "gpu_type" in normalized:
            normalized["hardware"] = normalized["gpu_type"]

    if "tokens_per_second" not in normalized and "tokens_per_second_mean" in normalized:
        normalized["tokens_per_second"] = normalized["tokens_per_second_mean"]

    if "mean_input_tokens" not in normalized and "prompt_tokens" in normalized:
        normalized["mean_input_tokens"] = normalized["prompt_tokens"]
    if "mean_output_tokens" not in normalized and "output_tokens" in normalized:
        normalized["mean_output_tokens"] = normalized["output_tokens"]

    return normalized


def prepare_benchmark_for_insert(
    benchmark: dict,
    source: str = "local",
    confidence_level: str = "estimated",
) -> dict:
    """Prepare a benchmark record for database insertion.

    Normalizes field names from various JSON formats, generates a UUID
    and config_id, sets timestamps, and applies source/confidence_level.

    Args:
        benchmark: Raw benchmark dict (from JSON file or estimation output).
        source: Data source identifier, e.g. 'blis', 'llm-optimizer'.
        confidence_level: Trust level -- 'benchmarked' or 'estimated'.

    Returns:
        Dict ready for insertion into exported_summaries.
    """
    prepared = normalize_benchmark_fields(benchmark)

    prepared["id"] = str(uuid.uuid4())
    prepared["config_id"] = generate_config_id(prepared)

    now = datetime.now().isoformat()
    prepared.setdefault("type", "local")
    prepared["provider"] = None
    prepared["jbenchmark_created_at"] = now
    prepared["created_at"] = now
    prepared["updated_at"] = now
    prepared["loaded_at"] = None

    prepared.setdefault("framework", None)
    prepared.setdefault("framework_version", None)
    prepared.setdefault("huggingface_prompt_dataset", None)
    prepared.setdefault("entrypoint", None)
    prepared.setdefault("docker_image", None)
    prepared.setdefault("responses_per_second", None)
    prepared.setdefault("tps_mean", None)
    prepared.setdefault("tps_p90", None)
    prepared.setdefault("tps_p95", None)
    prepared.setdefault("tps_p99", None)
    prepared.setdefault("prompt_tokens_stdev", None)
    prepared.setdefault("prompt_tokens_min", None)
    prepared.setdefault("prompt_tokens_max", None)
    prepared.setdefault("output_tokens_stdev", None)
    prepared.setdefault("output_tokens_min", None)
    prepared.setdefault("output_tokens_max", None)
    prepared.setdefault("profiler_type", None)
    prepared.setdefault("profiler_image", None)
    prepared.setdefault("profiler_tag", None)
    prepared.setdefault("model_uri", None)
    prepared["source"] = source
    prepared["confidence_level"] = confidence_level

    return prepared


_INSERT_COLUMNS = [
    "id",
    "config_id",
    "model_hf_repo",
    "provider",
    "type",
    "ttft_mean",
    "ttft_p90",
    "ttft_p95",
    "ttft_p99",
    "e2e_mean",
    "e2e_p90",
    "e2e_p95",
    "e2e_p99",
    "itl_mean",
    "itl_p90",
    "itl_p95",
    "itl_p99",
    "tps_mean",
    "tps_p90",
    "tps_p95",
    "tps_p99",
    "hardware",
    "hardware_count",
    "framework",
    "requests_per_second",
    "responses_per_second",
    "tokens_per_second",
    "mean_input_tokens",
    "mean_output_tokens",
    "huggingface_prompt_dataset",
    "jbenchmark_created_at",
    "entrypoint",
    "docker_image",
    "framework_version",
    "created_at",
    "updated_at",
    "loaded_at",
    "prompt_tokens",
    "prompt_tokens_stdev",
    "prompt_tokens_min",
    "prompt_tokens_max",
    "output_tokens",
    "output_tokens_min",
    "output_tokens_max",
    "output_tokens_stdev",
    "profiler_type",
    "profiler_image",
    "profiler_tag",
    "source",
    "confidence_level",
    "model_uri",
]

_INSERT_QUERY = (
    f"INSERT INTO exported_summaries ({', '.join(_INSERT_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in _INSERT_COLUMNS)}) "
    "ON CONFLICT (config_id) DO NOTHING"
)


def ensure_schema(conn) -> None:
    """Create the database schema if it doesn't already exist."""
    from planner.data._resolver import data_path

    schema_path = data_path("schema.sql")
    schema_sql = schema_path.read_text()
    cursor = conn.cursor()
    cursor.executescript(schema_sql)
    conn.commit()
    cursor.close()
    logger.info("Database schema ensured")


def extract_metadata(data: dict) -> dict:
    """Extract source and confidence_level from JSON file metadata.

    Looks for a top-level '_metadata' or 'metadata' dict and returns
    standardized source/confidence_level values.

    Returns:
        Dict with 'source' and 'confidence_level' keys (may be None).
    """
    meta = data.get("_metadata") or data.get("metadata") or {}
    return {
        "source": meta.get("source"),
        "confidence_level": meta.get("confidence_level"),
    }


def _insert_benchmarks(
    conn,
    benchmarks: list[dict],
    source: str = "local",
    confidence_level: str = "estimated",
) -> dict:
    """Insert benchmarks into the database (append mode).

    Duplicates (same config_id) are silently skipped.
    The caller is responsible for committing the transaction.

    Args:
        conn: sqlite3 connection
        benchmarks: List of benchmark dicts from JSON
        source: Data source identifier (default: "local")
        confidence_level: Confidence level for the data (default: "estimated")

    Returns:
        Dict with insertion stats: {inserted, total_in_db, stats}
    """
    prepared_benchmarks = [
        prepare_benchmark_for_insert(b, source=source, confidence_level=confidence_level)
        for b in benchmarks
    ]

    rows = [tuple(p[col] for col in _INSERT_COLUMNS) for p in prepared_benchmarks]

    logger.info(f"Inserting {len(rows)} benchmark records...")
    with closing(conn.cursor()) as cursor:
        cursor.executemany(_INSERT_QUERY, rows)

    logger.info(f"Successfully processed {len(benchmarks)} benchmarks")

    return get_db_stats(conn)


def get_db_stats(conn) -> dict:
    """Get current database statistics.

    Args:
        conn: sqlite3 connection

    Returns:
        Dict with total_benchmarks, num_models, num_hardware_types,
        num_traffic_profiles, traffic_distribution
    """
    with closing(conn.cursor()) as cursor:
        cursor.execute("""
            SELECT
                COUNT(DISTINCT model_hf_repo) as num_models,
                COUNT(DISTINCT hardware) as num_hardware_types,
                (SELECT COUNT(*) FROM (
                    SELECT DISTINCT prompt_tokens, output_tokens FROM exported_summaries
                )) as num_traffic_profiles,
                COUNT(*) as total_benchmarks
            FROM exported_summaries
        """)
        row = cursor.fetchone()

        stats = {
            "num_models": row["num_models"] if row else 0,
            "num_hardware_types": row["num_hardware_types"] if row else 0,
            "num_traffic_profiles": row["num_traffic_profiles"] if row else 0,
            "total_benchmarks": row["total_benchmarks"] if row else 0,
        }

        cursor.execute("""
            SELECT prompt_tokens, output_tokens, COUNT(*) as num_benchmarks
            FROM exported_summaries
            GROUP BY prompt_tokens, output_tokens
            ORDER BY prompt_tokens, output_tokens
        """)
        stats["traffic_distribution"] = [
            {
                "prompt_tokens": r["prompt_tokens"],
                "output_tokens": r["output_tokens"],
                "count": r["num_benchmarks"],
            }
            for r in cursor.fetchall()
        ]

        # Get benchmark source breakdown
        cursor.execute("""
            SELECT source, confidence_level, COUNT(*) as count
            FROM exported_summaries
            GROUP BY source, confidence_level
            ORDER BY count DESC;
        """)
        stats["benchmark_sources"] = [
            {"source": r["source"], "confidence_level": r["confidence_level"], "count": r["count"]}
            for r in cursor.fetchall()
        ]

    return stats


def reset_benchmarks(conn) -> None:
    """Reset the benchmark database by deleting all data.

    Deletes all rows from exported_summaries. The schema and indexes
    are preserved.

    Args:
        conn: sqlite3 connection
    """
    with conn, closing(conn.cursor()) as cursor:
        cursor.execute("DELETE FROM exported_summaries")
    logger.info("Benchmark database reset complete")
