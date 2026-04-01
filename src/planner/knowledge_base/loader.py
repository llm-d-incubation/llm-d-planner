"""
Benchmark data loading utilities.

Provides functions for loading benchmark JSON data into PostgreSQL.
Used by both the scripts/load_benchmarks.py CLI tool and the
/api/v1/db/* API endpoints.
"""

import hashlib
import logging
import uuid
from datetime import datetime

from psycopg2.extras import execute_batch

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

    # Map model_id -> model_hf_repo (estimated/interpolated files)
    if "model_hf_repo" not in normalized and "model_id" in normalized:
        normalized["model_hf_repo"] = normalized["model_id"]

    # Map hardware_type or gpu_type -> hardware (interpolated file)
    if "hardware" not in normalized:
        if "hardware_type" in normalized:
            normalized["hardware"] = normalized["hardware_type"]
        elif "gpu_type" in normalized:
            normalized["hardware"] = normalized["gpu_type"]

    # Map tokens_per_second_mean -> tokens_per_second (estimated files)
    if "tokens_per_second" not in normalized and "tokens_per_second_mean" in normalized:
        normalized["tokens_per_second"] = normalized["tokens_per_second_mean"]

    # Map mean_input_tokens/mean_output_tokens from prompt_tokens/output_tokens if missing
    if "mean_input_tokens" not in normalized and "prompt_tokens" in normalized:
        normalized["mean_input_tokens"] = normalized["prompt_tokens"]
    if "mean_output_tokens" not in normalized and "output_tokens" in normalized:
        normalized["mean_output_tokens"] = normalized["output_tokens"]

    return normalized


def prepare_benchmark_for_insert(benchmark: dict) -> dict:
    """Prepare a benchmark record for database insertion."""
    # First normalize field names from different JSON formats
    prepared = normalize_benchmark_fields(benchmark)

    # Generate UUID and config_id
    prepared["id"] = str(uuid.uuid4())
    prepared["config_id"] = generate_config_id(prepared)

    # Add required fields with defaults (matching real data schema)
    prepared["type"] = "local"  # benchmark type
    prepared["provider"] = None  # Optional field
    prepared["jbenchmark_created_at"] = datetime.now()
    prepared["created_at"] = datetime.now()
    prepared["updated_at"] = datetime.now()
    prepared["loaded_at"] = None  # Optional field

    # Optional fields that may not be in all JSON formats
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
    prepared["source"] = "local"

    return prepared


# The INSERT query used by insert_benchmarks()
_INSERT_QUERY = """
    INSERT INTO exported_summaries (
        id, config_id, model_hf_repo, provider, type,
        ttft_mean, ttft_p90, ttft_p95, ttft_p99,
        e2e_mean, e2e_p90, e2e_p95, e2e_p99,
        itl_mean, itl_p90, itl_p95, itl_p99,
        tps_mean, tps_p90, tps_p95, tps_p99,
        hardware, hardware_count, framework,
        requests_per_second, responses_per_second, tokens_per_second,
        mean_input_tokens, mean_output_tokens,
        huggingface_prompt_dataset, jbenchmark_created_at,
        entrypoint, docker_image, framework_version,
        created_at, updated_at, loaded_at,
        prompt_tokens, prompt_tokens_stdev, prompt_tokens_min, prompt_tokens_max,
        output_tokens, output_tokens_min, output_tokens_max, output_tokens_stdev,
        profiler_type, profiler_image, profiler_tag,
        source
    ) VALUES (
        %(id)s, %(config_id)s, %(model_hf_repo)s, %(provider)s, %(type)s,
        %(ttft_mean)s, %(ttft_p90)s, %(ttft_p95)s, %(ttft_p99)s,
        %(e2e_mean)s, %(e2e_p90)s, %(e2e_p95)s, %(e2e_p99)s,
        %(itl_mean)s, %(itl_p90)s, %(itl_p95)s, %(itl_p99)s,
        %(tps_mean)s, %(tps_p90)s, %(tps_p95)s, %(tps_p99)s,
        %(hardware)s, %(hardware_count)s, %(framework)s,
        %(requests_per_second)s, %(responses_per_second)s, %(tokens_per_second)s,
        %(mean_input_tokens)s, %(mean_output_tokens)s,
        %(huggingface_prompt_dataset)s, %(jbenchmark_created_at)s,
        %(entrypoint)s, %(docker_image)s, %(framework_version)s,
        %(created_at)s, %(updated_at)s, %(loaded_at)s,
        %(prompt_tokens)s, %(prompt_tokens_stdev)s, %(prompt_tokens_min)s, %(prompt_tokens_max)s,
        %(output_tokens)s, %(output_tokens_min)s, %(output_tokens_max)s, %(output_tokens_stdev)s,
        %(profiler_type)s, %(profiler_image)s, %(profiler_tag)s,
        %(source)s
    )
    ON CONFLICT (config_id) DO NOTHING;
"""


def insert_benchmarks(conn, benchmarks: list[dict]) -> dict:
    """Insert benchmarks into the database (append mode).

    Duplicates (same config_id) are silently skipped.

    Args:
        conn: psycopg2 connection
        benchmarks: List of benchmark dicts from JSON

    Returns:
        Dict with insertion stats: {inserted, total_in_db, stats}
    """
    cursor = conn.cursor()

    # Ensure unique constraint on config_id for duplicate detection
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_config_id_unique
        ON exported_summaries(config_id);
    """)
    conn.commit()

    # Prepare benchmarks with required fields
    prepared_benchmarks = [prepare_benchmark_for_insert(b) for b in benchmarks]

    logger.info(f"Inserting {len(prepared_benchmarks)} benchmark records...")
    execute_batch(cursor, _INSERT_QUERY, prepared_benchmarks, page_size=100)
    conn.commit()

    logger.info(f"Successfully processed {len(benchmarks)} benchmarks")

    # Get updated stats
    stats = get_db_stats(conn)
    cursor.close()
    return stats


def get_db_stats(conn) -> dict:
    """Get current database statistics.

    Args:
        conn: psycopg2 connection

    Returns:
        Dict with total_benchmarks, num_models, num_hardware_types,
        num_traffic_profiles, traffic_distribution
    """
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(DISTINCT model_hf_repo) as num_models,
            COUNT(DISTINCT hardware) as num_hardware_types,
            COUNT(DISTINCT (prompt_tokens, output_tokens)) as num_traffic_profiles,
            COUNT(*) as total_benchmarks
        FROM exported_summaries;
    """)
    row = cursor.fetchone()

    stats = {
        "num_models": row[0] if row else 0,
        "num_hardware_types": row[1] if row else 0,
        "num_traffic_profiles": row[2] if row else 0,
        "total_benchmarks": row[3] if row else 0,
    }

    # Get traffic profile distribution
    cursor.execute("""
        SELECT prompt_tokens, output_tokens, COUNT(*) as num_benchmarks
        FROM exported_summaries
        GROUP BY prompt_tokens, output_tokens
        ORDER BY prompt_tokens, output_tokens;
    """)
    stats["traffic_distribution"] = [
        {"prompt_tokens": r[0], "output_tokens": r[1], "count": r[2]} for r in cursor.fetchall()
    ]

    cursor.close()
    return stats


def reset_benchmarks(conn) -> None:
    """Reset the benchmark database by truncating all data.

    Truncates exported_summaries (cascading to related tables)
    and recreates the unique index on config_id.

    Args:
        conn: psycopg2 connection
    """
    cursor = conn.cursor()
    cursor.execute("TRUNCATE exported_summaries CASCADE;")
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_config_id_unique
        ON exported_summaries(config_id);
    """)
    conn.commit()
    cursor.close()
    logger.info("Benchmark database reset complete")
