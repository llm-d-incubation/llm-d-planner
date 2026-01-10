#!/usr/bin/env python3
"""
Convert PostgreSQL dump file to JSON format for benchmark loading.

This script extracts benchmark data from a PostgreSQL custom dump file
and converts it to the JSON format used by load_benchmarks.py.

Usage:
    python scripts/convert_pgdump_to_json.py data/integ-oct-29.sql
    # Creates: data/benchmarks_GuideLLM.json

    python scripts/convert_pgdump_to_json.py data/integ-oct-29.sql -o data/custom_output.json
    # Creates: data/custom_output.json

The output JSON file will have the same format as benchmarks_BLIS.json and can be
loaded using the standard JSON loader.

Requirements:
    - Docker running with neuralnav-postgres container (make db-start)
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def extract_data_via_docker(dump_file: Path) -> list[dict]:
    """Extract benchmark data using Docker PostgreSQL container.

    This approach:
    1. Copies the dump file into the running postgres container
    2. Creates a temporary database
    3. Restores the dump into it
    4. Queries and exports the data as JSON
    5. Cleans up
    """
    container_name = "neuralnav-postgres"

    # Check if container is running
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )
    if container_name not in result.stdout:
        print(f"Error: Docker container '{container_name}' is not running.")
        print("Please start it with: make db-start")
        sys.exit(1)

    print("Using Docker container for conversion...")

    try:
        # Copy dump file to container
        print("  Copying dump file to container...")
        subprocess.run(
            ["docker", "cp", str(dump_file), f"{container_name}:/tmp/dump.sql"],
            check=True,
        )

        # Create temp database
        print("  Creating temporary database...")
        subprocess.run(
            ["docker", "exec", container_name, "psql", "-U", "postgres",
             "-c", "DROP DATABASE IF EXISTS temp_import;"],
            capture_output=True,
        )
        subprocess.run(
            ["docker", "exec", container_name, "psql", "-U", "postgres",
             "-c", "CREATE DATABASE temp_import;"],
            check=True,
            capture_output=True,
        )

        # Restore dump into temp database (ignore errors about roles)
        print("  Restoring dump...")
        subprocess.run(
            ["docker", "exec", container_name, "pg_restore", "-U", "postgres",
             "-d", "temp_import", "--no-owner", "--no-privileges", "/tmp/dump.sql"],
            check=False,  # Ignore errors (e.g., role doesn't exist)
            capture_output=True,
        )

        # Export data as JSON (select all columns except id and timestamps)
        print("  Exporting data as JSON...")
        export_query = """
        SELECT json_agg(row_to_json(t))
        FROM (
            SELECT
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
                tokens_per_second,
                mean_input_tokens,
                mean_output_tokens,
                huggingface_prompt_dataset,
                entrypoint,
                docker_image,
                framework_version,
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
            FROM exported_summaries
        ) t;
        """

        result = subprocess.run(
            ["docker", "exec", container_name, "psql", "-U", "postgres",
             "-d", "temp_import", "-t", "-A", "-c", export_query],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse the JSON output
        json_data = result.stdout.strip()
        if not json_data or json_data == "null":
            return []

        benchmarks = json.loads(json_data)
        return benchmarks if benchmarks else []

    finally:
        # Cleanup: drop temp database and remove dump file
        print("  Cleaning up...")
        subprocess.run(
            ["docker", "exec", container_name, "psql", "-U", "postgres",
             "-c", "DROP DATABASE IF EXISTS temp_import;"],
            capture_output=True,
        )
        subprocess.run(
            ["docker", "exec", container_name, "rm", "-f", "/tmp/dump.sql"],
            capture_output=True,
        )


def convert_to_json_format(benchmarks: list[dict]) -> dict:
    """Convert benchmarks list to the standard JSON format."""
    return {
        "_metadata": {
            "description": "GuideLLM benchmark data converted from PostgreSQL dump",
            "version": "1.0",
            "source": "GuideLLM benchmarks via pg_dump conversion",
            "converted_at": datetime.now().isoformat(),
            "total_records": len(benchmarks),
        },
        "benchmarks": benchmarks,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert PostgreSQL dump file to JSON format for benchmark loading."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to PostgreSQL dump file (custom format from pg_dump)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Path to output JSON file (default: data/benchmarks_GuideLLM.json)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output (default: True)",
    )

    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    # Default output path
    if args.output is None:
        output_file = Path(__file__).parent.parent / "data" / "benchmarks_GuideLLM.json"
    else:
        output_file = args.output

    print(f"Converting {args.input_file} to JSON format...")

    # Extract data using Docker container
    benchmarks = extract_data_via_docker(args.input_file)

    if not benchmarks:
        print("Error: No benchmark data extracted from dump file.")
        print("Make sure the dump file contains data for the 'exported_summaries' table.")
        sys.exit(1)

    print(f"✓ Extracted {len(benchmarks)} benchmark records")

    # Convert to JSON format
    output_data = convert_to_json_format(benchmarks)

    # Write output
    with open(output_file, 'w') as f:
        if args.pretty:
            json.dump(output_data, f, indent=2, default=str)
        else:
            json.dump(output_data, f, default=str)

    print(f"✓ Written to {output_file}")

    # Show stats
    models = set(b.get('model_hf_repo') for b in benchmarks if b.get('model_hf_repo'))
    hardware = set(b.get('hardware') for b in benchmarks if b.get('hardware'))

    print(f"\n📊 Statistics:")
    print(f"  Total records: {len(benchmarks)}")
    print(f"  Unique models: {len(models)}")
    print(f"  Hardware types: {len(hardware)}")


if __name__ == "__main__":
    main()
