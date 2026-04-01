#!/usr/bin/env python3
"""
Load benchmark data from benchmarks.json into PostgreSQL.

This script reads benchmark data from a JSON file
and inserts it into the PostgreSQL exported_summaries table.

Core loading logic lives in planner.knowledge_base.loader and is
shared with the /api/v1/db/* API endpoints.
"""

import json
import os
import sys
from pathlib import Path

import psycopg2

from planner.knowledge_base.loader import insert_benchmarks


def get_db_connection():
    """Create a connection to the PostgreSQL database."""
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:neuralnav@localhost:5432/neuralnav")

    try:
        conn = psycopg2.connect(db_url)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print(f"Database URL: {db_url}")
        print("\nMake sure PostgreSQL is running:")
        print("  make db-start")
        sys.exit(1)


def load_benchmarks_json(json_file=None):
    """Load benchmarks from JSON file.

    Args:
        json_file: Optional path to JSON file relative to project root.
                  Defaults to "data/benchmarks/performance/benchmarks_BLIS.json" if not specified.
    """
    if json_file:
        json_path = Path(__file__).parent.parent / json_file
    else:
        json_path = (
            Path(__file__).parent.parent
            / "data"
            / "benchmarks"
            / "performance"
            / "benchmarks_BLIS.json"
        )

    if not json_path.exists():
        print(f"Error: {json_path} not found")
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)

    return data.get("benchmarks", [])


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
    print(f"Loaded {len(benchmarks)} benchmarks from JSON")

    # Connect to database
    print("Connecting to PostgreSQL...")
    conn = get_db_connection()
    print("Connected to database")
    print()

    try:
        # Insert benchmarks using shared loader
        stats = insert_benchmarks(conn, benchmarks)

        print("\nDatabase Statistics:")
        print(f"  Models: {stats['num_models']}")
        print(f"  Hardware types: {stats['num_hardware_types']}")
        print(f"  Traffic profiles: {stats['num_traffic_profiles']}")
        print(f"  Total benchmarks: {stats['total_benchmarks']}")

        print("\nTraffic Profile Distribution:")
        for tp in stats.get("traffic_distribution", []):
            print(f"  ({tp['prompt_tokens']}, {tp['output_tokens']}): {tp['count']} benchmarks")

    except Exception as e:
        print(f"\nError inserting benchmarks: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        conn.close()

    print("\n" + "=" * 60)
    print("Migration complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  make db-query-traffic  # View traffic patterns")
    print("  make db-query-models   # View available models")
    print("  make db-shell          # Open PostgreSQL shell")


if __name__ == "__main__":
    main()
