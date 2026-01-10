#!/usr/bin/env python3
"""
Load benchmark data from benchmarks.json into PostgreSQL.

This script reads the synthetic benchmark data from data/benchmarks.json
and inserts it into the PostgreSQL exported_summaries table.
"""

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_batch


def get_db_connection():
    """Create a connection to the PostgreSQL database."""
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:neuralnav@localhost:5432/neuralnav"
    )

    try:
        conn = psycopg2.connect(db_url)
        return conn
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        print(f"Database URL: {db_url}")
        print("\nMake sure PostgreSQL is running:")
        print("  make db-start")
        sys.exit(1)


def load_benchmarks_json(json_file=None):
    """Load benchmarks from JSON file.

    Args:
        json_file: Optional path to JSON file relative to project root.
                  Defaults to "data/benchmarks_BLIS.json" if not specified.
    """
    if json_file:
        json_path = Path(__file__).parent.parent / json_file
    else:
        json_path = Path(__file__).parent.parent / "data" / "benchmarks_BLIS.json"

    if not json_path.exists():
        print(f"❌ Error: {json_path} not found")
        sys.exit(1)

    with open(json_path, 'r') as f:
        data = json.load(f)

    return data.get("benchmarks", [])


def generate_config_id(benchmark):
    """Generate a deterministic config_id from benchmark configuration."""
    # Create a hash from the configuration
    config_str = f"{benchmark['model_hf_repo']}_{benchmark['hardware']}_{benchmark['hardware_count']}_{benchmark['prompt_tokens']}_{benchmark['output_tokens']}"
    return hashlib.md5(config_str.encode()).hexdigest()


def normalize_benchmark_fields(benchmark):
    """Normalize field names from different JSON formats to match DB schema.

    Handles field mapping for:
    - benchmarks_BLIS.json (uses model_hf_repo, hardware)
    - benchmarks_estimated_performance.json (uses model_id, hardware)
    - benchmarks_interpolated_v2.json (uses model_id, hardware_type/gpu_type)
    """
    normalized = benchmark.copy()

    # Map model_id -> model_hf_repo (estimated/interpolated files)
    if 'model_hf_repo' not in normalized and 'model_id' in normalized:
        normalized['model_hf_repo'] = normalized['model_id']

    # Map hardware_type or gpu_type -> hardware (interpolated file)
    if 'hardware' not in normalized:
        if 'hardware_type' in normalized:
            normalized['hardware'] = normalized['hardware_type']
        elif 'gpu_type' in normalized:
            normalized['hardware'] = normalized['gpu_type']

    # Map tokens_per_second_mean -> tokens_per_second (estimated files)
    if 'tokens_per_second' not in normalized and 'tokens_per_second_mean' in normalized:
        normalized['tokens_per_second'] = normalized['tokens_per_second_mean']

    # Map mean_input_tokens/mean_output_tokens from prompt_tokens/output_tokens if missing
    if 'mean_input_tokens' not in normalized and 'prompt_tokens' in normalized:
        normalized['mean_input_tokens'] = normalized['prompt_tokens']
    if 'mean_output_tokens' not in normalized and 'output_tokens' in normalized:
        normalized['mean_output_tokens'] = normalized['output_tokens']

    return normalized


def prepare_benchmark_for_insert(benchmark):
    """Prepare a benchmark record for database insertion."""
    from datetime import datetime

    # First normalize field names from different JSON formats
    prepared = normalize_benchmark_fields(benchmark)

    # Generate UUID and config_id
    prepared['id'] = str(uuid.uuid4())
    prepared['config_id'] = generate_config_id(prepared)

    # Add required fields with defaults (matching real data schema)
    prepared['type'] = 'local'  # benchmark type
    prepared['provider'] = None  # Optional field
    prepared['jbenchmark_created_at'] = datetime.now()
    prepared['created_at'] = datetime.now()
    prepared['updated_at'] = datetime.now()
    prepared['loaded_at'] = None  # Optional field

    # Optional fields that may not be in all JSON formats
    prepared.setdefault('framework', None)
    prepared.setdefault('framework_version', None)
    prepared.setdefault('huggingface_prompt_dataset', None)
    prepared.setdefault('entrypoint', None)
    prepared.setdefault('docker_image', None)
    prepared.setdefault('responses_per_second', None)
    prepared.setdefault('tps_mean', None)
    prepared.setdefault('tps_p90', None)
    prepared.setdefault('tps_p95', None)
    prepared.setdefault('tps_p99', None)
    prepared.setdefault('prompt_tokens_stdev', None)
    prepared.setdefault('prompt_tokens_min', None)
    prepared.setdefault('prompt_tokens_max', None)
    prepared.setdefault('output_tokens_stdev', None)
    prepared.setdefault('output_tokens_min', None)
    prepared.setdefault('output_tokens_max', None)
    prepared.setdefault('profiler_type', None)
    prepared.setdefault('profiler_image', None)
    prepared.setdefault('profiler_tag', None)

    return prepared


def insert_benchmarks(conn, benchmarks):
    """Insert benchmarks into the database (append mode).

    Note: This function appends to existing data. Use `make db-reset` first
    if you want to start with a clean database. Duplicates (same config_id)
    are skipped.
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

    # Prepare insert query (all fields from real schema)
    insert_query = """
        INSERT INTO exported_summaries (
            id,
            config_id,
            model_hf_repo,
            provider,
            type,
            ttft_mean,
            ttft_p90,
            ttft_p95,
            ttft_p99,
            e2e_mean,
            e2e_p90,
            e2e_p95,
            e2e_p99,
            itl_mean,
            itl_p90,
            itl_p95,
            itl_p99,
            tps_mean,
            tps_p90,
            tps_p95,
            tps_p99,
            hardware,
            hardware_count,
            framework,
            requests_per_second,
            responses_per_second,
            tokens_per_second,
            mean_input_tokens,
            mean_output_tokens,
            huggingface_prompt_dataset,
            jbenchmark_created_at,
            entrypoint,
            docker_image,
            framework_version,
            created_at,
            updated_at,
            loaded_at,
            prompt_tokens,
            prompt_tokens_stdev,
            prompt_tokens_min,
            prompt_tokens_max,
            output_tokens,
            output_tokens_min,
            output_tokens_max,
            output_tokens_stdev,
            profiler_type,
            profiler_image,
            profiler_tag
        ) VALUES (
            %(id)s,
            %(config_id)s,
            %(model_hf_repo)s,
            %(provider)s,
            %(type)s,
            %(ttft_mean)s,
            %(ttft_p90)s,
            %(ttft_p95)s,
            %(ttft_p99)s,
            %(e2e_mean)s,
            %(e2e_p90)s,
            %(e2e_p95)s,
            %(e2e_p99)s,
            %(itl_mean)s,
            %(itl_p90)s,
            %(itl_p95)s,
            %(itl_p99)s,
            %(tps_mean)s,
            %(tps_p90)s,
            %(tps_p95)s,
            %(tps_p99)s,
            %(hardware)s,
            %(hardware_count)s,
            %(framework)s,
            %(requests_per_second)s,
            %(responses_per_second)s,
            %(tokens_per_second)s,
            %(mean_input_tokens)s,
            %(mean_output_tokens)s,
            %(huggingface_prompt_dataset)s,
            %(jbenchmark_created_at)s,
            %(entrypoint)s,
            %(docker_image)s,
            %(framework_version)s,
            %(created_at)s,
            %(updated_at)s,
            %(loaded_at)s,
            %(prompt_tokens)s,
            %(prompt_tokens_stdev)s,
            %(prompt_tokens_min)s,
            %(prompt_tokens_max)s,
            %(output_tokens)s,
            %(output_tokens_min)s,
            %(output_tokens_max)s,
            %(output_tokens_stdev)s,
            %(profiler_type)s,
            %(profiler_image)s,
            %(profiler_tag)s
        )
        ON CONFLICT (config_id) DO NOTHING;
    """

    print(f"Inserting {len(prepared_benchmarks)} benchmark records...")
    execute_batch(cursor, insert_query, prepared_benchmarks, page_size=100)

    conn.commit()
    print(f"✓ Successfully inserted {len(benchmarks)} benchmarks")

    # Show some statistics
    cursor.execute("""
        SELECT
            COUNT(DISTINCT model_hf_repo) as num_models,
            COUNT(DISTINCT hardware) as num_hardware_types,
            COUNT(DISTINCT (prompt_tokens, output_tokens)) as num_traffic_profiles,
            COUNT(*) as total_benchmarks
        FROM exported_summaries;
    """)
    stats = cursor.fetchone()

    print("\n📊 Database Statistics:")
    print(f"  Models: {stats[0]}")
    print(f"  Hardware types: {stats[1]}")
    print(f"  Traffic profiles: {stats[2]}")
    print(f"  Total benchmarks: {stats[3]}")

    # Show traffic profile distribution
    cursor.execute("""
        SELECT
            prompt_tokens,
            output_tokens,
            COUNT(*) as num_benchmarks
        FROM exported_summaries
        GROUP BY prompt_tokens, output_tokens
        ORDER BY prompt_tokens, output_tokens;
    """)

    print("\n🚦 Traffic Profile Distribution:")
    for row in cursor.fetchall():
        print(f"  ({row[0]}, {row[1]}): {row[2]} benchmarks")

    cursor.close()


def main():
    """Main function."""
    # Parse command-line arguments
    json_file = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 60)
    print("Loading Benchmark Data into PostgreSQL")
    print("=" * 60)
    print()

    # Load benchmarks from JSON
    benchmarks = load_benchmarks_json(json_file)
    print(f"✓ Loaded {len(benchmarks)} benchmarks from JSON")

    # Connect to database
    print("Connecting to PostgreSQL...")
    conn = get_db_connection()
    print("✓ Connected to database")
    print()

    try:
        # Insert benchmarks
        insert_benchmarks(conn, benchmarks)
    except Exception as e:
        print(f"\n❌ Error inserting benchmarks: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        conn.close()

    print("\n" + "=" * 60)
    print("✅ Migration complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  make db-query-traffic  # View traffic patterns")
    print("  make db-query-models   # View available models")
    print("  make db-shell          # Open PostgreSQL shell")


if __name__ == "__main__":
    main()
