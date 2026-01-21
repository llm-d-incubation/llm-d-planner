"""
Data access layer for model benchmark data using PostgreSQL.

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
- find_configurations_meeting_slo() - Primary method used by CapacityPlanner

These methods are kept for potential Phase 2 API endpoints, debugging, interactive
testing, or future UI features that may need to display available options.
"""

import logging
import os
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class BenchmarkData:
    """Model performance benchmark entry."""

    def __init__(self, data: dict):
        """Initialize from database row dict."""
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
        }


class BenchmarkRepository:
    """Repository for querying model benchmark data from PostgreSQL."""

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize benchmark repository.

        Args:
            database_url: PostgreSQL connection string (defaults to DATABASE_URL env var)
        """
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:neuralnav@localhost:5432/neuralnav"
        )
        self._test_connection()

    def _test_connection(self):
        """Test database connection on initialization."""
        try:
            conn = self._get_connection()
            conn.close()
            logger.info("Successfully connected to PostgreSQL benchmark database")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            logger.error(f"Database URL: {self.database_url}")
            raise

    def _get_connection(self):
        """Get a database connection."""
        return psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)

    def get_benchmark(
        self,
        model_hf_repo: str,
        hardware: str,
        hardware_count: int,
        prompt_tokens: int,
        output_tokens: int
    ) -> Optional[BenchmarkData]:
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
            WHERE model_hf_repo = %s
              AND hardware = %s
              AND hardware_count = %s
              AND prompt_tokens = %s
              AND output_tokens = %s
            LIMIT 1
        """

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, (model_hf_repo, hardware, hardware_count, prompt_tokens, output_tokens))
            row = cursor.fetchone()
            cursor.close()

            if row:
                return BenchmarkData(dict(row))
            return None
        finally:
            conn.close()

    def get_benchmarks_for_traffic_profile(
        self,
        model_hf_repo: str,
        hardware: str,
        hardware_count: int,
        prompt_tokens: int,
        output_tokens: int
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
            WHERE model_hf_repo = %s
              AND hardware = %s
              AND hardware_count = %s
              AND prompt_tokens = %s
              AND output_tokens = %s
        """

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, (model_hf_repo, hardware, hardware_count, prompt_tokens, output_tokens))
            rows = cursor.fetchall()
            cursor.close()

            return [BenchmarkData(dict(row)) for row in rows]
        finally:
            conn.close()

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
            WHERE model_hf_repo = %s
            ORDER BY hardware, hardware_count, prompt_tokens, output_tokens
        """

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, (model_hf_repo,))
            rows = cursor.fetchall()
            cursor.close()

            return [BenchmarkData(dict(row)) for row in rows]
        finally:
            conn.close()

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
            WHERE hardware = %s
            ORDER BY model_hf_repo, hardware_count, prompt_tokens, output_tokens
        """

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, (hardware,))
            rows = cursor.fetchall()
            cursor.close()

            return [BenchmarkData(dict(row)) for row in rows]
        finally:
            conn.close()

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

        Returns:
            List of benchmarks meeting all criteria (one per system configuration)
        """
        # Map percentile to column suffix
        valid_percentiles = {"mean", "p90", "p95", "p99"}
        if percentile not in valid_percentiles:
            logger.warning(f"Invalid percentile '{percentile}', defaulting to p95")
            percentile = "p95"

        # Build column names based on percentile
        ttft_col = f"ttft_{percentile}"
        itl_col = f"itl_{percentile}"
        e2e_col = f"e2e_{percentile}"

        # Build optional GPU filter clause
        gpu_filter = ""
        if gpu_types:
            gpu_filter = "AND hardware = ANY(%s)"
            logger.info(f"Filtering by GPU types: {gpu_types}")

        logger.info(f"Querying benchmarks with percentile={percentile} (columns: {ttft_col}, {itl_col}, {e2e_col})")

        # Use window function to rank benchmarks by requests_per_second within each
        # system configuration, then select only the highest QPS that meets SLO.
        # When multiple benchmarks exist at the same QPS, prefer the one with lowest E2E latency.
        # NOTE: Using string formatting for column names is safe here since we validate percentile above
        query = f"""
            WITH ranked_configs AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY model_hf_repo, hardware, hardware_count
                           ORDER BY requests_per_second DESC, {e2e_col} ASC
                       ) as rn
                FROM exported_summaries
                WHERE prompt_tokens = %s
                  AND output_tokens = %s
                  AND {ttft_col} <= %s
                  AND {itl_col} <= %s
                  AND {e2e_col} <= %s
                  AND requests_per_second >= %s
                  {gpu_filter}
            )
            SELECT
                id, config_id, model_hf_repo, provider, type,
                ttft_mean, ttft_p90, ttft_p95, ttft_p99,
                e2e_mean, e2e_p90, e2e_p95, e2e_p99,
                itl_mean, itl_p90, itl_p95, itl_p99,
                tps_mean, tps_p90, tps_p95, tps_p99,
                hardware, hardware_count, framework, requests_per_second, tokens_per_second,
                mean_input_tokens, mean_output_tokens,
                prompt_tokens, output_tokens
            FROM ranked_configs
            WHERE rn = 1
            ORDER BY model_hf_repo, hardware, hardware_count
        """

        # Build query parameters
        params: list = [prompt_tokens, output_tokens, ttft_p95_max_ms, itl_p95_max_ms, e2e_p95_max_ms, min_qps]
        if gpu_types:
            params.append(gpu_types)

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            cursor.close()

            results = [BenchmarkData(dict(row)) for row in rows]
        finally:
            conn.close()

        logger.info(f"Found {len(results)} benchmarks meeting SLO criteria")
        return results

    def get_available_models(self) -> list[str]:
        """Get list of all available models in the database."""
        query = "SELECT DISTINCT model_hf_repo FROM exported_summaries ORDER BY model_hf_repo"

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()

            return [row["model_hf_repo"] for row in rows]
        finally:
            conn.close()

    def get_available_hardware_types(self) -> list[str]:
        """Get list of all available hardware types in the database."""
        query = "SELECT DISTINCT hardware FROM exported_summaries ORDER BY hardware"

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()

            return [row["hardware"] for row in rows]
        finally:
            conn.close()

    def get_traffic_profiles(self) -> list[tuple[int, int]]:
        """Get list of all available traffic profiles (prompt_tokens, output_tokens)."""
        query = """
            SELECT DISTINCT prompt_tokens, output_tokens
            FROM exported_summaries
            ORDER BY prompt_tokens, output_tokens
        """

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()

            return [(row["prompt_tokens"], row["output_tokens"]) for row in rows]
        finally:
            conn.close()

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
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()

            return [BenchmarkData(dict(row)) for row in rows]
        finally:
            conn.close()
