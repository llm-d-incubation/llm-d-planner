"""ETL: sync RHOAI Model Catalog data into PostgreSQL and local catalogs."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from psycopg2.extensions import connection as pg_connection
from psycopg2.extras import execute_batch

if TYPE_CHECKING:
    from planner.knowledge_base.model_catalog import ModelCatalog, ModelInfo
    from planner.knowledge_base.model_catalog_client import ModelCatalogClient
    from planner.recommendation.quality.usecase_scorer import UseCaseQualityScorer

logger = logging.getLogger(__name__)


def _prop_str(props: dict, key: str, default: str = "") -> str:
    """Extract string value from customProperties."""
    entry = props.get(key)
    return entry.get("string_value", default) if isinstance(entry, dict) else default


def _prop_float(props: dict, key: str, default: float = 0.0) -> float:
    """Extract double/float value from customProperties."""
    entry = props.get(key)
    return float(entry.get("double_value", default)) if isinstance(entry, dict) else default


def _prop_int(props: dict, key: str, default: int = 0) -> int:
    """Extract int value from customProperties."""
    entry = props.get(key)
    if not isinstance(entry, dict):
        return default
    val = entry.get("int_value")
    if val is None:
        val = entry.get("double_value")
    if val is None:
        return default
    return int(val)


def _parse_profiler_config(props: dict) -> tuple[int, int]:
    """Extract prompt_tokens and output_tokens from profiler_config JSON."""
    raw = _prop_str(props, "profiler_config")
    fb = (
        int(_prop_float(props, "mean_input_tokens")),
        int(_prop_float(props, "mean_output_tokens")),
    )
    if not raw:
        return fb
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return fb
        args = parsed.get("args", {})
        return int(args.get("prompt_tokens", 0)), int(args.get("output_tokens", 0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return fb


_TASK_MAP: dict[str, list[str]] = {
    "text-to-text": [
        "chatbot_conversational",
        "summarization_short",
        "content_generation",
        "translation",
    ],  # fmt: skip
    "text-generation": ["chatbot_conversational", "code_completion", "content_generation"],
}
_SIZE_RE = re.compile(r"([\d.]+)\s*[Bb]\b")
_KNOWN_FAMILIES = ["granite", "llama", "mistral", "qwen", "gemma",
                    "phi", "deepseek", "mixtral", "kimi", "nemotron"]  # fmt: skip


def _parse_size(size_str: str) -> str:
    """Parse '8B params' into '8B'."""
    m = _SIZE_RE.search(size_str)
    return m.group(0).upper().replace(" ", "") if m else size_str


def _extract_family(name: str) -> str:
    """Extract model family from name."""
    base = name.split("/")[-1].lower()
    for fam in _KNOWN_FAMILIES:
        if fam in base:
            return fam
    return base.split("-")[0]


def _catalog_model_to_model_info(model: dict) -> ModelInfo:
    """Map a Model Catalog model dict to a ModelInfo instance."""
    from planner.knowledge_base.model_catalog import ModelInfo as _ModelInfo

    name = model.get("name", "")
    props = model.get("customProperties", {})
    size_str = props.get("size", {}).get("string_value", "")
    validated_on = props.get("validated_on", {}).get("string_value", "")
    validated_val = props.get("validated", {}).get("string_value", "").lower()
    has_validated = validated_val in ("true", "yes") or bool(validated_on)

    supported_tasks: list[str] = []
    for t in model.get("tasks", []):
        supported_tasks.extend(_TASK_MAP.get(t, [t]))

    size_parsed = _parse_size(size_str) if size_str else ""
    try:
        param_b = float(size_parsed.rstrip("Bb"))
    except ValueError:
        param_b = 7.0

    return _ModelInfo(
        {
            "model_id": name,
            "name": name.split("/")[-1],
            "provider": model.get("provider", "Unknown"),
            "family": _extract_family(name),
            "size_parameters": _parse_size(size_str) if size_str else f"{param_b}B",
            # Model Catalog API does not expose context_length (verified Mar 2026)
            "context_length": 128000,
            "supported_tasks": list(set(supported_tasks)),
            "domain_specialization": [],
            "license": model.get("license", "Unknown"),
            "license_type": "permissive"
            if "apache" in model.get("license", "").lower()
            else "restricted",
            "min_gpu_memory_gb": max(8, int(param_b * 1.5)),
            "recommended_for": supported_tasks,
            "approval_status": "approved" if has_validated else "pending",
        }
    )


_NULL_FIELDS = ["responses_per_second", "provider", "loaded_at", "huggingface_prompt_dataset",
                "entrypoint", "docker_image", "prompt_tokens_stdev", "prompt_tokens_min",
                "prompt_tokens_max", "output_tokens_stdev", "output_tokens_min",
                "output_tokens_max", "profiler_type", "profiler_image", "profiler_tag"]  # fmt: skip


def _artifact_to_row(artifact: dict, model_uri: str = "") -> dict | None:
    """Map a performance-metrics artifact to a DB row dict."""
    props = artifact.get("customProperties", {})
    prompt_tokens, output_tokens = _parse_profiler_config(props)
    if prompt_tokens == 0 or output_tokens == 0:
        return None

    model_id = _prop_str(props, "model_id")
    hardware = _prop_str(props, "hardware_type")
    if not model_id or not hardware:
        return None

    hw_count = _prop_int(props, "hardware_count")
    rps = _prop_float(props, "requests_per_second")
    if hw_count <= 0 or rps <= 0:
        return None

    now = datetime.now()
    row: dict = {
        "id": str(uuid.uuid4()),
        "model_hf_repo": model_id,
        "hardware": hardware,
        "hardware_count": hw_count,
        "framework": _prop_str(props, "framework_type", "vllm"),
        "framework_version": _prop_str(props, "framework_version"),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "mean_input_tokens": _prop_float(props, "mean_input_tokens"),
        "mean_output_tokens": _prop_float(props, "mean_output_tokens"),
        "requests_per_second": rps,
        "source": "model_catalog",
        "model_uri": model_uri or None,
        "type": "model_catalog",
        "created_at": now,
        "updated_at": now,
        "jbenchmark_created_at": now,
    }
    # Extract latency/throughput percentiles via loop
    for prefix in ("ttft", "itl", "e2e", "tps"):
        for suffix in ("mean", "p90", "p95", "p99"):
            row[f"{prefix}_{suffix}"] = _prop_float(props, f"{prefix}_{suffix}")
    row["tokens_per_second"] = row["tps_mean"]
    for k in _NULL_FIELDS:
        row[k] = None
    row["config_id"] = hashlib.sha256(
        f"{model_id}_{hardware}_{hw_count}_{prompt_tokens}_{output_tokens}_{rps}".encode()
    ).hexdigest()[:32]
    return row


_COLUMNS = (
    "id, config_id, model_hf_repo, provider, type, "
    "ttft_mean, ttft_p90, ttft_p95, ttft_p99, "
    "e2e_mean, e2e_p90, e2e_p95, e2e_p99, "
    "itl_mean, itl_p90, itl_p95, itl_p99, "
    "tps_mean, tps_p90, tps_p95, tps_p99, "
    "hardware, hardware_count, framework, "
    "requests_per_second, responses_per_second, tokens_per_second, "
    "mean_input_tokens, mean_output_tokens, "
    "huggingface_prompt_dataset, jbenchmark_created_at, "
    "entrypoint, docker_image, framework_version, "
    "created_at, updated_at, loaded_at, "
    "prompt_tokens, prompt_tokens_stdev, prompt_tokens_min, prompt_tokens_max, "
    "output_tokens, output_tokens_min, output_tokens_max, output_tokens_stdev, "
    "profiler_type, profiler_image, profiler_tag, source, model_uri"
)
_VALUES = ", ".join(f"%({c.strip()})s" for c in _COLUMNS.split(", "))
_INSERT_QUERY = (
    f"INSERT INTO exported_summaries ({_COLUMNS}) VALUES ({_VALUES}) "
    "ON CONFLICT (config_id) DO NOTHING;"
)


@dataclass
class SyncResult:
    """Outcome of a Model Catalog sync operation."""

    benchmarks_inserted: int = 0
    models_merged: int = 0
    quality_scores_loaded: int = 0
    errors: list[str] = field(default_factory=list)


def sync_model_catalog(
    client: ModelCatalogClient,
    conn: pg_connection,
    model_catalog: ModelCatalog,
    quality_scorer: UseCaseQualityScorer,
) -> SyncResult:
    """Sync Model Catalog data into PostgreSQL, ModelCatalog, and quality scorer."""
    result = SyncResult()

    try:
        models = client.list_models()
    except Exception as exc:
        result.errors.append(f"Failed to list models: {exc}")
        logger.warning("Model Catalog sync aborted: %s", exc)
        return result

    rows: list[dict] = []
    model_infos: list[ModelInfo] = []
    accuracy_scores: dict[str, float] = {}

    for model in models:
        model_name = model.get("name", "")
        if not model_name:
            continue

        try:
            info = _catalog_model_to_model_info(model)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            result.errors.append(f"Malformed model {model_name}: {exc}")
            continue
        if info.model_id:
            model_infos.append(info)

        source_id = model.get("source_id")
        try:
            artifacts = client.get_model_artifacts(model_name, source_id=source_id)
        except Exception as exc:
            result.errors.append(f"Failed to fetch artifacts for {model_name}: {exc}")
            continue

        model_uri = ""
        for artifact in artifacts:
            if artifact.get("artifactType") == "model-artifact":
                model_uri = artifact.get("uri", "")
                break

        for artifact in artifacts:
            a_type, m_type = artifact.get("artifactType"), artifact.get("metricsType")
            if a_type == "metrics-artifact" and m_type == "performance-metrics":
                try:
                    row = _artifact_to_row(artifact, model_uri=model_uri)
                except (KeyError, ValueError, TypeError):
                    continue
                if row is not None:
                    rows.append(row)
            elif a_type == "metrics-artifact" and m_type == "accuracy-metrics":
                props = artifact.get("customProperties", {})
                avg = props.get("overall_average") if isinstance(props, dict) else None
                if isinstance(avg, dict):
                    with contextlib.suppress(ValueError, TypeError):
                        accuracy_scores[model_name.lower()] = float(avg.get("double_value", 0))

    if rows:
        if result.errors:
            logger.warning(
                "Proceeding with DB insert despite %d crawl errors; " "%d valid rows collected",
                len(result.errors),
                len(rows),
            )
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM exported_summaries WHERE source = %s", ("model_catalog",))
            execute_batch(cursor, _INSERT_QUERY, rows, page_size=100)
            conn.commit()
            result.benchmarks_inserted = len(rows)
        except Exception as exc:
            conn.rollback()
            result.errors.append(f"DB error: {exc}")
            logger.error("Model Catalog sync DB error: %s", exc)
            return result

    if model_infos:
        result.models_merged = model_catalog.merge_external_models(model_infos)
    if accuracy_scores:
        quality_scorer.set_catalog_fallback(accuracy_scores)
        result.quality_scores_loaded = len(accuracy_scores)

    logger.info("Sync complete: %d benchmarks, %d models, %d scores",
                result.benchmarks_inserted, result.models_merged, result.quality_scores_loaded)  # fmt: skip
    return result
