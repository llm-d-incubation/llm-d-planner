"""Performance estimation via roofline model.

Generates synthetic BenchmarkData for (model, GPU) pairs that lack real
benchmark data, using the BentoML llm-optimizer roofline model.

Extracted from ConfigFinder to isolate estimation as a separate concern
from capacity planning and scoring logic.
"""

import contextlib
import io
import logging
import os
import time

from planner.capacity_planner import check_model_fits_gpu, get_model_config_from_hf
from planner.gpu_recommender import GPURecommender
from planner.knowledge_base.benchmarks import BenchmarkData, BenchmarkRepository
from planner.knowledge_base.model_catalog import ModelCatalog
from planner.shared.schemas import SLOTargets, TrafficProfile

logger = logging.getLogger(__name__)

# Mapping from gpu_catalog.json gpu_type to llm_optimizer GPU_SPECS keys.
# GPUs not in this map are not supported by the roofline model.
CATALOG_TO_ROOFLINE_GPU: dict[str, str] = {
    "H100": "H100",
    "H200": "H200",
    "A100-80": "A100",
    "A100-40": "A100-40GB",
    "L40": "L40",
    "L20": "L20",
    "B100": "B100",
    "B200": "B200",
}


@contextlib.contextmanager
def _suppress_noisy_output():
    """Suppress stdout/stderr from HuggingFace, safetensors tqdm, and llm_optimizer."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def convert_estimation_to_benchmark(
    model_id: str,
    gpu_type: str,
    gpu_count: int,
    prompt_tokens: int,
    output_tokens: int,
    ttft_ms: float,
    itl_ms: float,
    e2e_latency_ms: float,
    output_throughput_tps: float,
) -> BenchmarkData:
    """Convert GPU Recommender roofline output to BenchmarkData format.

    The roofline model produces single-point estimates (no percentile
    distribution), so the same value is used for mean/p90/p95/p99.
    """
    rps = output_throughput_tps / output_tokens if output_tokens > 0 else 0.0

    data = {
        "model_hf_repo": model_id,
        "hardware": gpu_type,
        "hardware_count": gpu_count,
        "framework": "vllm",
        "framework_version": "estimated",
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "mean_input_tokens": prompt_tokens,
        "mean_output_tokens": output_tokens,
        "ttft_mean": ttft_ms,
        "ttft_p90": ttft_ms,
        "ttft_p95": ttft_ms,
        "ttft_p99": ttft_ms,
        "itl_mean": itl_ms,
        "itl_p90": itl_ms,
        "itl_p95": itl_ms,
        "itl_p99": itl_ms,
        "e2e_mean": e2e_latency_ms,
        "e2e_p90": e2e_latency_ms,
        "e2e_p95": e2e_latency_ms,
        "e2e_p99": e2e_latency_ms,
        "tps_mean": output_throughput_tps,
        "tps_p90": output_throughput_tps,
        "tps_p95": output_throughput_tps,
        "tps_p99": output_throughput_tps,
        "tokens_per_second": output_throughput_tps,
        "requests_per_second": rps,
        "estimated": True,
        "source": "llm-optimizer",
        "confidence_level": "estimated",
        "model_uri": None,
    }
    return BenchmarkData(data)


def generate_estimated_configs(
    traffic_profile: TrafficProfile,
    slo_targets: SLOTargets,
    preferred_models: list[str],
    existing_benchmarks: list[BenchmarkData],
    gpu_types: list[str] | None,
    catalog: ModelCatalog,
    benchmark_repo: BenchmarkRepository,
    estimate_all_catalog: bool = False,
) -> tuple[list[BenchmarkData], list[str]]:
    """Generate estimated BenchmarkData for (model, GPU) pairs without benchmarks.

    Uses the capacity planner for memory feasibility and the BentoML roofline
    model for synthetic performance estimation. Results are written to the DB
    for future cache hits.

    Args:
        traffic_profile: Current traffic profile (prompt_tokens, output_tokens)
        slo_targets: SLO targets (TTFT, ITL, E2E)
        preferred_models: User-specified model IDs (HuggingFace format)
        existing_benchmarks: Benchmark results already found from DB
        gpu_types: GPU types to evaluate (None = all catalog GPUs)
        catalog: Model catalog for GPU info
        benchmark_repo: Repository for persisting estimates
        estimate_all_catalog: If True, also estimate for catalog models
                             without benchmarks (not just user-specified)

    Returns:
        Tuple of (list of new BenchmarkData, list of warning messages)
    """
    warnings: list[str] = []

    # 1. Build covered set from existing benchmarks (includes prior roofline estimates)
    # Key is (model, gpu, tp) so different TP values are estimated independently.
    covered: set[tuple[str, str, int]] = set()
    for bench in existing_benchmarks:
        covered.add((bench.model_hf_repo.lower(), bench.hardware.lower(), bench.hardware_count))

    # 2. Determine models to estimate
    models_to_estimate: list[str] = []
    for model_id in preferred_models:
        models_to_estimate.append(model_id)

    if estimate_all_catalog:
        for model_info in catalog.get_all_models():
            if model_info.model_id not in models_to_estimate:
                models_to_estimate.append(model_info.model_id)

    if not models_to_estimate:
        return [], warnings

    # 3. Determine GPUs to evaluate
    if gpu_types:
        catalog_gpus = [gt for gt in catalog.get_all_gpu_types() if gt.gpu_type in gpu_types]
    else:
        catalog_gpus = catalog.get_all_gpu_types()

    # Configurable limits
    max_models = int(os.getenv("PLANNER_ESTIMATED_MAX_MODELS", "5"))
    timeout_s = int(os.getenv("PLANNER_ESTIMATED_TIMEOUT_S", "60"))
    models_to_estimate = models_to_estimate[:max_models]

    hf_token = os.getenv("HF_TOKEN")
    new_benchmarks: list[BenchmarkData] = []
    start_time = time.monotonic()

    gpu_names = [g.gpu_type for g in catalog_gpus]
    logger.info(
        f"Estimation plan: {len(models_to_estimate)} models \u00d7 "
        f"{len(catalog_gpus)} GPUs {gpu_names}, "
        f"{len(covered)} (model, GPU, TP) combinations already covered"
    )
    for model_id in models_to_estimate:
        logger.info(f"  model: {model_id}")

    # 4. For each model, check feasibility and estimate performance
    for model_idx, model_id in enumerate(models_to_estimate, 1):
        elapsed = time.monotonic() - start_time
        if elapsed > timeout_s:
            remaining = len(models_to_estimate) - model_idx + 1
            msg = (
                f"Estimation timeout ({timeout_s}s) reached after {elapsed:.0f}s. "
                f"Skipping {remaining} remaining model(s)."
            )
            logger.warning(msg)
            warnings.append(msg)
            break
        logger.info(f"Estimating model {model_idx}/{len(models_to_estimate)}: {model_id}")
        # Fetch model config from HuggingFace
        try:
            with _suppress_noisy_output():
                model_config = get_model_config_from_hf(model_id, hf_token)
        except Exception as e:
            msg = f"Could not estimate performance for {model_id}: {e}"
            logger.warning(msg)
            warnings.append(msg)
            continue

        model_had_any_gpu = False
        model_checked_any_gpu = False
        model_had_error = False

        for gpu_info in catalog_gpus:
            # Map catalog GPU name to roofline model name
            roofline_gpu = CATALOG_TO_ROOFLINE_GPU.get(gpu_info.gpu_type)
            if not roofline_gpu:
                continue  # GPU not supported by roofline model

            model_checked_any_gpu = True

            # Check memory feasibility
            try:
                with _suppress_noisy_output():
                    valid_tps = check_model_fits_gpu(
                        model_id, model_config, gpu_info.memory_gb, hf_token=hf_token
                    )
            except Exception as e:
                msg = f"Could not check GPU fit for {model_id} on {gpu_info.gpu_type}: {e}"
                logger.warning(msg)
                warnings.append(msg)
                model_had_error = True
                logger.info(f"Skipping remaining GPUs for {model_id} due to model-level error")
                break
            if not valid_tps:
                logger.info(
                    f"  {gpu_info.gpu_type}: model does not fit ({gpu_info.memory_gb}GB) at any TP"
                )
                continue

            model_had_any_gpu = True

            # Estimate performance at each valid TP value
            for tp in valid_tps:
                # Skip if already covered (e.g., from prior DB benchmark or earlier estimate)
                if (model_id.lower(), gpu_info.gpu_type.lower(), tp) in covered:
                    continue

                # Run roofline estimation
                try:
                    with _suppress_noisy_output():
                        recommender = GPURecommender(
                            model_id=model_id,
                            input_len=traffic_profile.prompt_tokens,
                            output_len=traffic_profile.output_tokens,
                            max_gpus=tp,
                            gpu_list=[roofline_gpu],
                            max_ttft=slo_targets.ttft_p95_target_ms,
                            max_itl=slo_targets.itl_p95_target_ms,
                            max_latency=slo_targets.e2e_p95_target_ms / 1000,
                            catalog=catalog,
                        )
                        gpu_results, failed_gpus = recommender.get_gpu_results()
                except Exception as e:
                    msg = f"Roofline estimation failed for {model_id} on {gpu_info.gpu_type} TP={tp}: {e}"
                    logger.warning(msg)
                    warnings.append(msg)
                    continue

                if roofline_gpu in failed_gpus:
                    # Don't surface per-TP constraint failures as warnings —
                    # higher TP values may still succeed
                    continue

                if roofline_gpu not in gpu_results:
                    continue

                result = gpu_results[roofline_gpu]
                best_latency = (
                    result.best_configs.get("best_latency")
                    if isinstance(result.best_configs, dict)
                    else None
                )
                if not best_latency or not hasattr(best_latency, "ttft_ms"):
                    continue

                # Convert to BenchmarkData
                bench = convert_estimation_to_benchmark(
                    model_id=model_id,
                    gpu_type=gpu_info.gpu_type,  # Use catalog name for DB consistency
                    gpu_count=tp,
                    prompt_tokens=traffic_profile.prompt_tokens,
                    output_tokens=traffic_profile.output_tokens,
                    ttft_ms=best_latency.ttft_ms,
                    itl_ms=best_latency.itl_ms,
                    e2e_latency_ms=best_latency.e2e_latency_s * 1000,
                    output_throughput_tps=best_latency.output_throughput_tps,
                )
                new_benchmarks.append(bench)
                covered.add((model_id.lower(), gpu_info.gpu_type.lower(), tp))

        # Only warn "does not fit" when the model genuinely doesn't fit —
        # not when a network error prevented the check.
        if not model_had_any_gpu and model_checked_any_gpu and not model_had_error:
            msg = f"Model {model_id} does not fit on any available GPU"
            logger.warning(msg)
            warnings.append(msg)

    # 5. Write new estimates to DB for future cache hits
    if new_benchmarks:
        try:
            benchmark_repo.save_benchmarks(new_benchmarks)
            logger.info(f"Wrote {len(new_benchmarks)} roofline estimates to DB")
        except Exception as e:
            msg = f"Failed to persist roofline estimates to DB: {type(e).__name__}: {e}"
            logger.warning(msg)
            warnings.append(msg)

    return new_benchmarks, warnings
