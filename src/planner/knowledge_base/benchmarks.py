"""
Data access layer for model benchmark data.

NOTE ON UNUSED METHODS:
This repository contains several query methods that are currently unused in production
but preserved for future use:

- get_benchmark() - Single benchmark lookup (unused - debugging/future API use)
- get_benchmarks_for_traffic_profile() - Filter by traffic profile (unused)
- get_benchmarks_for_model() - All benchmarks for a model (unused)
- get_benchmarks_for_hardware() - All benchmarks for GPU type (unused)
- get_available_models() - List distinct models (unused - may use for UI)
- get_available_hardware_types() - List distinct GPUs (unused - may use for UI)
- get_traffic_profiles() - List distinct traffic profiles (unused)
- get_all_benchmarks() - Full table dump (unused - debugging only)

PRODUCTION METHOD:
- find_configurations_meeting_slo() - Primary method used by ConfigFinder

These methods are kept for potential Phase 2 API endpoints, debugging, interactive
testing, or future UI features that may need to display available options.
"""

import json
import logging
from pathlib import Path

from planner.knowledge_base.db import create_connection
from planner.knowledge_base.loader import (
    _insert_benchmarks,
    extract_metadata,
    get_db_stats,
    reset_benchmarks,
)

logger = logging.getLogger(__name__)


class BenchmarkData:
    """Model performance benchmark entry."""

    def __init__(self, data: dict):
        """Initialize from a database row or dict with benchmark fields.

        Key fields beyond latency/throughput metrics:
            estimated: True for roofline-estimated benchmarks (default False).
            source: Benchmark tool/method, e.g. 'blis', 'llm-optimizer' (default 'other').
            confidence_level: 'benchmarked' or 'estimated' (default 'estimated').
        """
        self.model_hf_repo = data["model_hf_repo"]
        self.hardware = data["hardware"]
        self.hardware_count = data["hardware_count"]
        self.framework = data.get("framework", "vllm")
        self.framework_version = data.get("framework_version", "0.6.2")

        # Traffic profile
        self.prompt_tokens = data["prompt_tokens"]
        self.output_tokens = data["output_tokens"]
        self.mean_input_tokens = data["mean_input_tokens"]
        self.mean_output_tokens = data["mean_output_tokens"]

        # TTFT metrics (p95)
        self.ttft_mean = data["ttft_mean"]
        self.ttft_p90 = data["ttft_p90"]
        self.ttft_p95 = data["ttft_p95"]
        self.ttft_p99 = data["ttft_p99"]

        # ITL metrics (p95)
        self.itl_mean = data.get("itl_mean")
        self.itl_p90 = data.get("itl_p90")
        self.itl_p95 = data.get("itl_p95")
        self.itl_p99 = data.get("itl_p99")

        # E2E metrics (p95)
        self.e2e_mean = data["e2e_mean"]
        self.e2e_p90 = data["e2e_p90"]
        self.e2e_p95 = data["e2e_p95"]
        self.e2e_p99 = data["e2e_p99"]

        # TPS (tokens per second) metrics - all percentiles
        self.tps_mean = data.get("tps_mean")
        self.tps_p90 = data.get("tps_p90")
        self.tps_p95 = data.get("tps_p95")
        self.tps_p99 = data.get("tps_p99")

        # Throughput (legacy fields kept for backwards compatibility)
        self.tokens_per_second = data["tokens_per_second"]
        self.requests_per_second = data["requests_per_second"]

        # Estimated flag (True for interpolated benchmarks)
        self.estimated = data.get("estimated", False)

        # Model artifact URI (e.g., OCI registry URI)
        self.model_uri = data.get("model_uri")

        self.source = data.get("source", "other")
        self.confidence_level = data.get("confidence_level", "estimated")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_hf_repo": self.model_hf_repo,
            "hardware": self.hardware,
            "hardware_count": self.hardware_count,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "mean_input_tokens": self.mean_input_tokens,
            "mean_output_tokens": self.mean_output_tokens,
            "ttft_mean": self.ttft_mean,
            "ttft_p90": self.ttft_p90,
            "ttft_p95": self.ttft_p95,
            "ttft_p99": self.ttft_p99,
            "itl_mean": self.itl_mean,
            "itl_p90": self.itl_p90,
            "itl_p95": self.itl_p95,
            "itl_p99": self.itl_p99,
            "e2e_mean": self.e2e_mean,
            "e2e_p90": self.e2e_p90,
            "e2e_p95": self.e2e_p95,
            "e2e_p99": self.e2e_p99,
            "tps_mean": self.tps_mean,
            "tps_p90": self.tps_p90,
            "tps_p95": self.tps_p95,
            "tps_p99": self.tps_p99,
            "tokens_per_second": self.tokens_per_second,
            "requests_per_second": self.requests_per_second,
            "model_uri": self.model_uri,
            "estimated": self.estimated,
            "source": self.source,
            "confidence_level": self.confidence_level,
        }


class BenchmarkRepository:
    """Repository for querying model benchmark data."""

    def __init__(self, db_path: str | None = None):
        """
        Initialize benchmark repository.

        Args:
            db_path: Database file path (defaults to PLANNER_DB_PATH or data/planner.db)
        """
        self._conn = create_connection(db_path)
        logger.info("Initialized benchmark repository")

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __enter__(self) -> "BenchmarkRepository":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @classmethod
    def from_files(
        cls, *json_paths: str | Path, db_path: str | None = None
    ) -> "BenchmarkRepository":
        """Create a repository pre-loaded with benchmark data from JSON files.

        Args:
            *json_paths: Paths to benchmark JSON files (each with a "benchmarks" array)
            db_path: Database file path (defaults to PLANNER_DB_PATH or data/planner.db)

        Returns:
            A BenchmarkRepository with the data loaded.
        """
        repo = cls(db_path=db_path)
        for path in json_paths:
            with open(path) as f:
                data = json.load(f)
            benchmarks = data.get("benchmarks", [])
            if not benchmarks:
                continue
            meta = extract_metadata(data)
            source = meta["source"] or "local"
            confidence_level = meta["confidence_level"] or "estimated"
            _insert_benchmarks(
                repo._conn, benchmarks, source=source, confidence_level=confidence_level
            )
        repo._conn.commit()
        return repo

    def _get_connection(self):
        """Get the database connection."""
        return self._conn

    def get_stats(self) -> dict:
        """Get current database statistics."""
        return get_db_stats(self._conn)

    def check_health(self) -> None:
        """Verify database access and the benchmark schema without scanning all rows."""
        cursor = self._conn.execute("SELECT 1 FROM exported_summaries LIMIT 1")
        cursor.close()

    def reset(self) -> None:
        """Delete all benchmark data. Schema is preserved."""
        reset_benchmarks(self._conn)

    def load_benchmarks(
        self,
        benchmarks: list[dict],
        source: str = "local",
        confidence_level: str = "estimated",
    ) -> dict:
        """Insert benchmarks into the database (append mode).

        Duplicates (same config_id) are silently skipped.

        Args:
            benchmarks: List of benchmark dicts
            source: Data source identifier
            confidence_level: Trust level for the data

        Returns:
            Dict with database stats after the operation.
        """
        try:
            stats = _insert_benchmarks(
                self._conn, benchmarks, source=source, confidence_level=confidence_level
            )
            self._conn.commit()
            return stats
        except Exception:
            self._conn.rollback()
            raise

    def replace_benchmarks(
        self,
        source: str,
        benchmarks: list[dict],
        confidence_level: str = "benchmarked",
    ) -> dict:
        """Replace all benchmarks for a given source.

        Deletes existing rows matching the source, then inserts the new
        benchmarks. Used by Model Catalog sync to refresh data without
        affecting benchmarks from other sources.

        Args:
            source: Source identifier (e.g. "model_catalog")
            benchmarks: List of benchmark dicts to insert
            confidence_level: Trust level for the new data

        Returns:
            Dict with database stats after the operation.
        """
        conn = self._conn
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM exported_summaries WHERE source = ?", (source,))
            cursor.close()
            stats = _insert_benchmarks(
                conn, benchmarks, source=source, confidence_level=confidence_level
            )
            conn.commit()
            return stats
        except Exception:
            conn.rollback()
            raise

    def save_benchmarks(
        self,
        benchmarks: list["BenchmarkData"],
        source: str = "llm-optimizer",
        confidence_level: str = "estimated",
    ) -> None:
        """Persist benchmark data to the database.

        Args:
            benchmarks: BenchmarkData objects to insert.
            source: Data source identifier.
            confidence_level: Trust level ('benchmarked' or 'estimated').

        Raises:
            Exception: If the DB write fails.
        """
        # Prepare data before opening connection to fail fast and keep
        # the connection short-lived.
        benchmark_dicts = [b.to_dict() for b in benchmarks]
        for d in benchmark_dicts:
            d.setdefault("prompt_tokens", d.get("mean_input_tokens"))
            d.setdefault("output_tokens", d.get("mean_output_tokens"))

        conn = self._get_connection()
        try:
            _insert_benchmarks(
                conn, benchmark_dicts, source=source, confidence_level=confidence_level
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def get_benchmark(
        self,
        model_hf_repo: str,
        hardware: str,
        hardware_count: int,
        prompt_tokens: int,
        output_tokens: int,
    ) -> BenchmarkData | None:
        """
        Get benchmark for specific configuration and traffic profile.

        NOTE: This method is currently unused in production code. It's preserved
        for future API endpoints, debugging, or interactive testing.

        Production code uses find_configurations_meeting_slo() which queries
        multiple benchmarks with SLO filtering.

        Args:
            model_hf_repo: Model HuggingFace repository
            hardware: GPU type (e.g., NVIDIA-L4, NVIDIA-A100-80GB)
            hardware_count: Number of GPUs (tensor parallel size)
            prompt_tokens: Target prompt length (GuideLLM config)
            output_tokens: Target output length (GuideLLM config)

        Returns:
            BenchmarkData if found, None otherwise
        """
        query = """
            SELECT * FROM exported_summaries
            WHERE model_hf_repo = ?
              AND hardware = ?
              AND hardware_count = ?
              AND prompt_tokens = ?
              AND output_tokens = ?
            LIMIT 1
        """

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            query, (model_hf_repo, hardware, hardware_count, prompt_tokens, output_tokens)
        )
        row = cursor.fetchone()
        cursor.close()

        if row:
            return BenchmarkData(dict(row))
        return None

    def get_benchmarks_for_traffic_profile(
        self,
        model_hf_repo: str,
        hardware: str,
        hardware_count: int,
        prompt_tokens: int,
        output_tokens: int,
    ) -> list[BenchmarkData]:
        """
        Get all benchmarks matching model, hardware, and traffic profile.

        This is similar to get_benchmark but returns a list in case there are
        multiple benchmark runs for the same configuration.

        Args:
            model_hf_repo: Model HuggingFace repository
            hardware: GPU type
            hardware_count: Number of GPUs
            prompt_tokens: Target prompt length
            output_tokens: Target output length

        Returns:
            List of matching benchmarks
        """
        query = """
            SELECT * FROM exported_summaries
            WHERE model_hf_repo = ?
              AND hardware = ?
              AND hardware_count = ?
              AND prompt_tokens = ?
              AND output_tokens = ?
        """

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            query, (model_hf_repo, hardware, hardware_count, prompt_tokens, output_tokens)
        )
        rows = cursor.fetchall()
        cursor.close()

        return [BenchmarkData(dict(row)) for row in rows]

    def get_benchmarks_for_model(self, model_hf_repo: str) -> list[BenchmarkData]:
        """
        Get all benchmarks for a specific model.

        Args:
            model_hf_repo: Model HuggingFace repository

        Returns:
            List of benchmarks for this model
        """
        query = """
            SELECT * FROM exported_summaries
            WHERE model_hf_repo = ?
            ORDER BY hardware, hardware_count, prompt_tokens, output_tokens
        """

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query, (model_hf_repo,))
        rows = cursor.fetchall()
        cursor.close()

        return [BenchmarkData(dict(row)) for row in rows]

    def get_benchmarks_for_hardware(self, hardware: str) -> list[BenchmarkData]:
        """
        Get all benchmarks for a specific GPU type.

        Args:
            hardware: GPU type (e.g., NVIDIA-A100-80GB)

        Returns:
            List of benchmarks for this hardware
        """
        query = """
            SELECT * FROM exported_summaries
            WHERE hardware = ?
            ORDER BY model_hf_repo, hardware_count, prompt_tokens, output_tokens
        """

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query, (hardware,))
        rows = cursor.fetchall()
        cursor.close()

        return [BenchmarkData(dict(row)) for row in rows]

    def find_configurations_meeting_slo(
        self,
        prompt_tokens: int,
        output_tokens: int,
        ttft_p95_max_ms: int,
        itl_p95_max_ms: int,
        e2e_p95_max_ms: int,
        min_qps: float = 0,
        percentile: str = "p95",
        gpu_types: list[str] | None = None,
        exclude_estimated: bool = False,
    ) -> list[BenchmarkData]:
        """
        Find all configurations that meet SLO requirements for a traffic profile.

        For each unique system configuration (model_hf_repo, hardware, hardware_count,
        prompt_tokens, output_tokens), selects only the benchmark with the highest
        requests_per_second that still meets SLO requirements. This is critical because
        benchmarks are collected at multiple QPS rates, and we want the maximum throughput
        that doesn't violate SLO targets.

        Args:
            prompt_tokens: Target prompt length
            output_tokens: Target output length
            ttft_p95_max_ms: Maximum acceptable TTFT (ms) - parameter name kept for backwards compat
            itl_p95_max_ms: Maximum acceptable ITL (ms/token) - parameter name kept for backwards compat
            e2e_p95_max_ms: Maximum acceptable E2E (ms) - parameter name kept for backwards compat
            min_qps: Minimum required QPS
            percentile: Which percentile column to use (mean, p90, p95, p99)
            gpu_types: Optional list of GPU types to filter by (normalized canonical names)
            exclude_estimated: If True, exclude rows with confidence_level='estimated'

        Returns:
            List of benchmarks meeting all criteria (one per system configuration)
        """
        valid_percentiles = {"mean", "p90", "p95", "p99"}
        if percentile not in valid_percentiles:
            logger.warning(f"Invalid percentile '{percentile}', defaulting to p95")
            percentile = "p95"

        ttft_col = f"ttft_{percentile}"
        itl_col = f"itl_{percentile}"
        e2e_col = f"e2e_{percentile}"

        gpu_filter = ""
        if gpu_types:
            placeholders = ", ".join("?" for _ in gpu_types)
            gpu_filter = f"AND hardware IN ({placeholders})"
            logger.info(f"Filtering by GPU types: {gpu_types}")

        estimated_filter = ""
        if exclude_estimated:
            estimated_filter = "AND confidence_level != 'estimated'"

        logger.info(
            f"Querying benchmarks with percentile={percentile} "
            f"(columns: {ttft_col}, {itl_col}, {e2e_col})"
        )

        query = f"""
            WITH ranked_configs AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY model_hf_repo, hardware, hardware_count
                           ORDER BY requests_per_second DESC, {e2e_col} ASC
                       ) as rn
                FROM exported_summaries
                WHERE prompt_tokens = ?
                  AND output_tokens = ?
                  AND {ttft_col} <= ?
                  AND {itl_col} <= ?
                  AND {e2e_col} <= ?
                  AND requests_per_second >= ?
                  {gpu_filter}
                  {estimated_filter}
            )
            SELECT
                id, config_id, model_hf_repo, provider, type,
                ttft_mean, ttft_p90, ttft_p95, ttft_p99,
                e2e_mean, e2e_p90, e2e_p95, e2e_p99,
                itl_mean, itl_p90, itl_p95, itl_p99,
                tps_mean, tps_p90, tps_p95, tps_p99,
                hardware, hardware_count, framework, requests_per_second, tokens_per_second,
                mean_input_tokens, mean_output_tokens,
                prompt_tokens, output_tokens,
                model_uri, source, confidence_level
            FROM ranked_configs
            WHERE rn = 1
            ORDER BY model_hf_repo, hardware, hardware_count
        """

        params: list = [
            prompt_tokens,
            output_tokens,
            ttft_p95_max_ms,
            itl_p95_max_ms,
            e2e_p95_max_ms,
            min_qps,
        ]
        if gpu_types:
            params.extend(gpu_types)

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        cursor.close()

        results = [BenchmarkData(dict(row)) for row in rows]
        logger.info(f"Found {len(results)} benchmarks meeting SLO criteria")
        return results

    def get_available_models(self) -> list[str]:
        """Get list of all available models in the database."""
        query = "SELECT DISTINCT model_hf_repo FROM exported_summaries ORDER BY model_hf_repo"

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()

        return [row["model_hf_repo"] for row in rows]

    def get_available_hardware_types(self) -> list[str]:
        """Get list of all available hardware types in the database."""
        query = "SELECT DISTINCT hardware FROM exported_summaries ORDER BY hardware"

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()

        return [row["hardware"] for row in rows]

    def get_traffic_profiles(self) -> list[tuple[int, int]]:
        """Get list of all available traffic profiles (prompt_tokens, output_tokens)."""
        query = """
            SELECT DISTINCT prompt_tokens, output_tokens
            FROM exported_summaries
            ORDER BY prompt_tokens, output_tokens
        """

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()

        return [(row["prompt_tokens"], row["output_tokens"]) for row in rows]

    def get_all_benchmarks(self) -> list[BenchmarkData]:
        """
        Get all benchmarks.

        Warning: This may return a large dataset. Use with caution.
        """
        query = """
            SELECT * FROM exported_summaries
            ORDER BY model_hf_repo, hardware, hardware_count, prompt_tokens, output_tokens
        """

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()

        return [BenchmarkData(dict(row)) for row in rows]
