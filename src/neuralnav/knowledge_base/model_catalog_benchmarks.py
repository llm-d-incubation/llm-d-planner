"""RHOAI Model Catalog benchmark source.

Loads performance-metrics artifacts from the Model Catalog API,
maps them to BenchmarkData, and provides in-memory SLO filtering.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from neuralnav.knowledge_base.benchmarks import BenchmarkData

if TYPE_CHECKING:
    from neuralnav.knowledge_base.model_catalog_client import ModelCatalogClient

logger = logging.getLogger(__name__)

# Cache TTL in seconds (1 hour)
_CACHE_TTL = 3600


def _prop_str(props: dict, key: str, default: str = "") -> str:
    """Extract string value from customProperties."""
    entry = props.get(key)
    if entry is None:
        return default
    return entry.get("string_value", default)


def _prop_float(props: dict, key: str, default: float = 0.0) -> float:
    """Extract double/float value from customProperties."""
    entry = props.get(key)
    if entry is None:
        return default
    return float(entry.get("double_value", default))


def _prop_int(props: dict, key: str, default: int = 0) -> int:
    """Extract int value from customProperties."""
    entry = props.get(key)
    if entry is None:
        return default
    return int(entry.get("int_value", entry.get("double_value", default)))


def _parse_profiler_config(props: dict) -> tuple[int, int]:
    """Extract prompt_tokens and output_tokens from profiler_config JSON.

    Falls back to mean_input_tokens / mean_output_tokens if profiler_config
    is absent or unparseable.
    """
    raw = _prop_str(props, "profiler_config")
    if not raw:
        return (
            int(_prop_float(props, "mean_input_tokens")),
            int(_prop_float(props, "mean_output_tokens")),
        )
    try:
        config = json.loads(raw)
        args = config.get("args", {})
        return int(args.get("prompt_tokens", 0)), int(args.get("output_tokens", 0))
    except (json.JSONDecodeError, TypeError):
        return (
            int(_prop_float(props, "mean_input_tokens")),
            int(_prop_float(props, "mean_output_tokens")),
        )


def _artifact_to_benchmark_data(artifact: dict) -> BenchmarkData | None:
    """Map a performance-metrics artifact to a BenchmarkData instance."""
    props = artifact.get("customProperties", {})

    prompt_tokens, output_tokens = _parse_profiler_config(props)
    if prompt_tokens == 0 or output_tokens == 0:
        return None

    tps_mean = _prop_float(props, "tps_mean")
    model_id = _prop_str(props, "model_id")
    hardware = _prop_str(props, "hardware_type")

    if not model_id or not hardware:
        return None

    data = {
        "model_hf_repo": model_id,
        "hardware": hardware,
        "hardware_count": _prop_int(props, "hardware_count"),
        "framework": _prop_str(props, "framework_type", "vllm"),
        "framework_version": _prop_str(props, "framework_version"),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "mean_input_tokens": _prop_float(props, "mean_input_tokens"),
        "mean_output_tokens": _prop_float(props, "mean_output_tokens"),
        "ttft_mean": _prop_float(props, "ttft_mean"),
        "ttft_p90": _prop_float(props, "ttft_p90"),
        "ttft_p95": _prop_float(props, "ttft_p95"),
        "ttft_p99": _prop_float(props, "ttft_p99"),
        "itl_mean": _prop_float(props, "itl_mean"),
        "itl_p90": _prop_float(props, "itl_p90"),
        "itl_p95": _prop_float(props, "itl_p95"),
        "itl_p99": _prop_float(props, "itl_p99"),
        "e2e_mean": _prop_float(props, "e2e_mean"),
        "e2e_p90": _prop_float(props, "e2e_p90"),
        "e2e_p95": _prop_float(props, "e2e_p95"),
        "e2e_p99": _prop_float(props, "e2e_p99"),
        "tps_mean": tps_mean,
        "tps_p90": _prop_float(props, "tps_p90"),
        "tps_p95": _prop_float(props, "tps_p95"),
        "tps_p99": _prop_float(props, "tps_p99"),
        "tokens_per_second": tps_mean,
        "requests_per_second": _prop_float(props, "requests_per_second"),
        "estimated": False,
    }

    return BenchmarkData(data)


class ModelCatalogBenchmarkSource:
    """Load performance benchmarks from the RHOAI Model Catalog API.

    Satisfies the BenchmarkSource protocol via structural subtyping.
    Fetches all performance-metrics artifacts, caches them in memory
    with a configurable TTL, and provides SLO-based filtering.
    """

    def __init__(self, client: ModelCatalogClient) -> None:
        self._client = client
        self._benchmarks: list[BenchmarkData] = []
        self._loaded_at: float = 0

    def preload(self) -> None:
        """Eagerly load benchmark cache (call during app startup)."""
        self._load_all()

    def _ensure_loaded(self) -> None:
        """Load or refresh benchmark cache if stale."""
        if self._benchmarks and (time.time() - self._loaded_at) < _CACHE_TTL:
            return
        self._load_all()

    def _load_all(self) -> None:
        """Fetch all models and their performance-metrics artifacts."""
        benchmarks: list[BenchmarkData] = []
        models = self._client.list_models()
        for model in models:
            model_name = model.get("name", "")
            if not model_name:
                continue
            source_id = model.get("source_id")
            try:
                artifacts = self._client.get_model_artifacts(model_name, source_id=source_id)
            except Exception:
                logger.warning("Failed to fetch artifacts for %s, skipping", model_name)
                continue
            for artifact in artifacts:
                if (
                    artifact.get("artifactType") == "metrics-artifact"
                    and artifact.get("metricsType") == "performance-metrics"
                ):
                    bench = _artifact_to_benchmark_data(artifact)
                    if bench is not None:
                        benchmarks.append(bench)

        self._benchmarks = benchmarks
        self._loaded_at = time.time()
        logger.info("Loaded %d performance benchmarks from Model Catalog", len(benchmarks))

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
        """Find configurations meeting SLO from cached Model Catalog data.

        Replicates the semantics of BenchmarkRepository.find_configurations_meeting_slo():
        - Exact match on (prompt_tokens, output_tokens)
        - SLO compliance at the requested percentile
        - Optional GPU type filter (case-insensitive)
        - Deduplication: keep highest RPS per (model, hardware, hardware_count)
        """
        self._ensure_loaded()

        valid_percentiles = {"mean", "p90", "p95", "p99"}
        if percentile not in valid_percentiles:
            percentile = "p95"

        # Normalize GPU filter to uppercase for case-insensitive comparison
        gpu_filter = {g.upper() for g in gpu_types} if gpu_types else None

        results: list[BenchmarkData] = []
        for bench in self._benchmarks:
            # Match traffic profile
            if bench.prompt_tokens != prompt_tokens or bench.output_tokens != output_tokens:
                continue

            # GPU filter
            if gpu_filter and bench.hardware.upper() not in gpu_filter:
                continue

            # SLO check at requested percentile
            ttft = getattr(bench, f"ttft_{percentile}", 0) or 0
            itl = getattr(bench, f"itl_{percentile}", 0) or 0
            e2e = getattr(bench, f"e2e_{percentile}", 0) or 0

            if ttft > ttft_p95_max_ms or itl > itl_p95_max_ms or e2e > e2e_p95_max_ms:
                continue

            # Min QPS check
            if bench.requests_per_second < min_qps:
                continue

            results.append(bench)

        # Deduplicate: keep highest RPS per (model, hardware, hardware_count)
        best: dict[tuple, BenchmarkData] = {}
        for bench in results:
            key = (bench.model_hf_repo, bench.hardware, bench.hardware_count)
            existing = best.get(key)
            if existing is None or bench.requests_per_second > existing.requests_per_second:
                best[key] = bench

        final = sorted(
            best.values(),
            key=lambda b: (b.model_hf_repo, b.hardware, b.hardware_count),
        )
        logger.info("Found %d benchmarks meeting SLO criteria (Model Catalog)", len(final))
        return final
