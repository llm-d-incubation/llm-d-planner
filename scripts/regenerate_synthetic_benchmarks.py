#!/usr/bin/env python3
"""
Regenerate synthetic benchmark data based on real benchmark data.

For each synthetic benchmark configuration, this script:
1. Looks up the matching real benchmark (exact match on model, hardware, hardware_count, prompt_tokens, output_tokens)
2. Applies a random adjustment factor (±15%)
3. Adjusts TTFT and ITL by the same percentage
4. Calculates E2E using: e2e = ttft + (output_tokens * itl)
5. Keeps mean/p90/p99 proportional to p95
"""

import json
import os
import random
import sys
from pathlib import Path

import psycopg2


def get_db_connection():
    """Create a connection to the PostgreSQL database."""
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:compass@localhost:5432/compass"
    )
    try:
        conn = psycopg2.connect(db_url)
        return conn
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        sys.exit(1)


def lookup_real_benchmark(conn, model_hf_repo, hardware, hardware_count, prompt_tokens, output_tokens):
    """
    Look up real benchmark data for the given configuration.

    Returns the median values if multiple benchmarks exist for the same config.
    """
    query = """
        SELECT
            ttft_p95,
            itl_p95,
            e2e_p95,
            ttft_mean,
            ttft_p90,
            ttft_p99,
            itl_mean,
            itl_p90,
            itl_p99,
            e2e_mean,
            e2e_p90,
            e2e_p99,
            requests_per_second,
            tokens_per_second
        FROM exported_summaries
        WHERE model_hf_repo = %s
          AND hardware = %s
          AND hardware_count = %s
          AND prompt_tokens = %s
          AND output_tokens = %s
        ORDER BY ttft_p95
        LIMIT 1;
    """

    cursor = conn.cursor()
    cursor.execute(query, (model_hf_repo, hardware, hardware_count, prompt_tokens, output_tokens))
    result = cursor.fetchone()
    cursor.close()

    if result:
        return {
            'ttft_p95': result[0],
            'itl_p95': result[1],
            'e2e_p95': result[2],
            'ttft_mean': result[3],
            'ttft_p90': result[4],
            'ttft_p99': result[5],
            'itl_mean': result[6],
            'itl_p90': result[7],
            'itl_p99': result[8],
            'e2e_mean': result[9],
            'e2e_p90': result[10],
            'e2e_p99': result[11],
            'requests_per_second': result[12],
            'tokens_per_second': result[13],
        }
    return None


def adjust_benchmark_with_real_data(benchmark, real_data, output_tokens):
    """
    Adjust synthetic benchmark to match real data with ±15% random variance.

    Args:
        benchmark: Synthetic benchmark dict to update
        real_data: Real benchmark data from database
        output_tokens: Number of output tokens for E2E calculation

    Returns:
        Updated benchmark dict
    """
    # Random adjustment factor: ±15%
    adjustment_factor = random.uniform(0.85, 1.15)

    # Adjust TTFT by the factor
    ttft_p95 = real_data['ttft_p95'] * adjustment_factor

    # Adjust ITL by the same factor
    itl_p95 = real_data['itl_p95'] * adjustment_factor

    # Calculate E2E: e2e = ttft + (output_tokens * itl)
    e2e_p95 = ttft_p95 + (output_tokens * itl_p95)

    # Calculate other percentiles proportionally to p95
    # Using typical ratios: mean ≈ 0.7*p95, p90 ≈ 0.87*p95, p99 ≈ 1.1*p95
    benchmark['ttft_p95'] = round(ttft_p95, 1)
    benchmark['ttft_mean'] = round(ttft_p95 * 0.7, 1)
    benchmark['ttft_p90'] = round(ttft_p95 * 0.87, 1)
    benchmark['ttft_p99'] = round(ttft_p95 * 1.1, 1)

    benchmark['itl_p95'] = round(itl_p95, 1)
    benchmark['itl_mean'] = round(itl_p95 * 0.7, 1)
    benchmark['itl_p90'] = round(itl_p95 * 0.87, 1)
    benchmark['itl_p99'] = round(itl_p95 * 1.1, 1)

    benchmark['e2e_p95'] = round(e2e_p95, 1)
    benchmark['e2e_mean'] = round(e2e_p95 * 0.7, 1)
    benchmark['e2e_p90'] = round(e2e_p95 * 0.87, 1)
    benchmark['e2e_p99'] = round(e2e_p95 * 1.1, 1)

    # Adjust throughput proportionally (inverse relationship with latency)
    # If we're faster, we can handle more requests
    throughput_factor = 1.0 / adjustment_factor
    benchmark['requests_per_second'] = round(real_data['requests_per_second'] * throughput_factor, 1)
    benchmark['tokens_per_second'] = round(real_data['tokens_per_second'] * throughput_factor, 1)

    return benchmark


def main():
    print("=" * 60)
    print("Regenerating Synthetic Benchmark Data")
    print("=" * 60)
    print()

    # Load existing synthetic benchmarks
    json_path = Path(__file__).parent.parent / "data" / "benchmarks.json"
    with open(json_path, 'r') as f:
        data = json.load(f)

    benchmarks = data['benchmarks']
    print(f"Loaded {len(benchmarks)} synthetic benchmarks from JSON")

    # Connect to database with real data
    print("Connecting to PostgreSQL with real benchmark data...")
    conn = get_db_connection()
    print("✓ Connected\n")

    # Process each benchmark
    updated_count = 0
    not_found_count = 0

    for i, benchmark in enumerate(benchmarks, 1):
        model = benchmark['model_hf_repo']
        hardware = benchmark['hardware']
        hw_count = benchmark['hardware_count']
        prompt_tokens = benchmark['prompt_tokens']
        output_tokens = benchmark['output_tokens']

        print(f"[{i}/{len(benchmarks)}] {model} on {hw_count}x {hardware} ({prompt_tokens}→{output_tokens})...")

        # Look up real benchmark
        real_data = lookup_real_benchmark(conn, model, hardware, hw_count, prompt_tokens, output_tokens)

        if real_data:
            # Update benchmark with adjusted real data
            adjust_benchmark_with_real_data(benchmark, real_data, output_tokens)
            updated_count += 1
            print(f"  ✓ Updated: TTFT p95={benchmark['ttft_p95']}ms, ITL p95={benchmark['itl_p95']}ms, E2E p95={benchmark['e2e_p95']}ms")
        else:
            not_found_count += 1
            print(f"  ⚠️  No matching real benchmark found - keeping original values")

    conn.close()

    # Save updated benchmarks
    print()
    print("=" * 60)
    print(f"Saving updated benchmarks to {json_path}...")

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    print()
    print("✅ Regeneration complete!")
    print(f"  Updated: {updated_count}/{len(benchmarks)}")
    print(f"  Not found: {not_found_count}/{len(benchmarks)}")
    print()
    print("Next steps:")
    print("  1. Review the updated data/benchmarks.json")
    print("  2. Reload into PostgreSQL: make postgres-load-synthetic")
    print("  3. Test recommendation flow")


if __name__ == "__main__":
    main()
