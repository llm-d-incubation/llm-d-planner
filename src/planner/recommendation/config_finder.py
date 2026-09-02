"""Capacity planning and GPU configuration recommendation.

IMPORTANT: Database Implementation (Phase 1):
- Uses traffic profile-based exact matching on (prompt_tokens, output_tokens)
- Queries benchmarks by exact traffic profile (512→256, 1024→1024, 4096→512, 10240→1536)
- Filters by p95 SLO compliance (TTFT, ITL, E2E)
- Uses pre-calculated e2e_p95 from benchmarks (not dynamic calculation)

Benchmarks collected using GuideLLM with fixed traffic profiles:
- Batching: vLLM continuous batching (dynamic, auto-configured)
- KV cache: enabled (vLLM default)
- Request pattern: steady-state load

TODO (Phase 2+): Parametric Performance Models
- Train regression models: f(prompt_tokens, output_tokens) → (ttft_p95, itl_p95, e2e_p95)
- Support arbitrary traffic profiles beyond the 4 GuideLLM defaults
- Interpolate for in-range predictions with confidence intervals
"""

import logging
import math

from planner.knowledge_base.benchmarks import BenchmarkData, BenchmarkRepository
from planner.knowledge_base.model_catalog import ModelCatalog, ModelInfo
from planner.shared.schemas import (
    ConfigurationScores,
    DeploymentConfiguration,
    DeploymentIntent,
    DeploymentRecommendation,
    GPUConfig,
    SLOTargets,
    TrafficProfile,
)
from planner.shared.utils import extract_gpu_max_counts, normalize_gpu_types
from quality_scoring.engine import ScoringEngine

from .scorer import Scorer

logger = logging.getLogger(__name__)


class ConfigFinder:
    """Plan GPU capacity to meet SLO targets and traffic requirements."""

    def __init__(
        self,
        benchmark_repo: BenchmarkRepository | None = None,
        catalog: ModelCatalog | None = None,
        engine: ScoringEngine | None = None,
        quality_weights: dict | None = None,
    ):
        """
        Initialize capacity planner.

        Args:
            benchmark_repo: Benchmark repository for database queries.
            catalog: Model catalog
            engine: ScoringEngine for quality scoring (from quality_scoring package)
            quality_weights: Use-case category weights dict loaded from quality_weights.json
        """
        self.benchmark_repo = benchmark_repo or BenchmarkRepository()
        self.catalog = catalog or ModelCatalog()
        self._engine = engine
        self._quality_weights = quality_weights or {}

    def update_engine(self, engine: ScoringEngine, quality_weights: dict | None = None) -> None:
        """Replace the scoring engine and optionally the quality weights."""
        self._engine = engine
        if quality_weights is not None:
            self._quality_weights = quality_weights

    def _calculate_required_replicas(self, qps_per_replica: float, required_qps: float) -> int:
        """
        Calculate number of replicas needed for traffic.

        Args:
            qps_per_replica: QPS capacity per replica
            required_qps: Required QPS to handle

        Returns:
            Number of replicas (minimum 1)
        """
        if qps_per_replica <= 0:
            return 0  # Infeasible: cannot serve positive throughput

        # Add 20% headroom for safety
        headroom_factor = 1.2
        required_capacity = required_qps * headroom_factor

        replicas = math.ceil(required_capacity / qps_per_replica)
        return max(1, replicas)

    def _generate_reasoning_from_bench(
        self,
        bench: BenchmarkData,
        gpu_config: GPUConfig,
        intent: DeploymentIntent,
        model: ModelInfo | None = None,
    ) -> str:
        """Generate explanation for recommendation from benchmark data.

        Args:
            bench: Benchmark data
            gpu_config: GPU configuration
            intent: Deployment intent
            model: Model info (optional, may be None if not in catalog)

        Returns:
            Reasoning string
        """
        reasons = []

        # Model selection
        if model:
            reasons.append(
                f"Selected {model.name} ({model.size_parameters}) for {intent.use_case} use case"
            )
        else:
            reasons.append(f"Selected {bench.model_hf_repo} for {intent.use_case} use case")

        # GPU configuration
        if gpu_config.tensor_parallel > 1:
            reasons.append(
                f"Using {gpu_config.tensor_parallel}x tensor parallelism on {gpu_config.gpu_type} "
                f"for improved latency"
            )
        else:
            reasons.append(f"Deploying on {gpu_config.gpu_type} GPUs")

        # Scaling
        if gpu_config.replicas > 1:
            reasons.append(
                f"{gpu_config.replicas} independent replicas to handle {intent.user_count} users"
            )

        # Performance
        ttft_p95 = int(bench.ttft_p95) if bench.ttft_p95 else 0
        itl_p95 = int(bench.itl_p95) if bench.itl_p95 else 0
        reasons.append(f"Expected performance: TTFT={ttft_p95}ms (p95), ITL={itl_p95}ms (p95)")

        return ". ".join(reasons)

    def plan_all_capacities(
        self,
        traffic_profile: TrafficProfile,
        slo_targets: SLOTargets,
        intent: DeploymentIntent,
        include_near_miss: bool = False,  # Strict SLO filtering - no tolerance
        near_miss_tolerance: float = 0.0,  # No near-miss tolerance
        weights: dict[str, int] | None = None,  # Custom weights for balanced score
        cluster_gpu_types: list[str] | None = None,
        preferred_models: list[str] | None = None,
        enable_estimated: bool = True,
    ) -> tuple[list[DeploymentRecommendation], list[str]]:
        """
        Plan GPU capacity and return ALL viable configurations meeting SLO.

        Queries benchmarks for all (model, GPU) configurations meeting SLO targets,
        then scores each on quality, price, and latency.

        Args:
            traffic_profile: Traffic characteristics (prompt_tokens, output_tokens)
            slo_targets: p95 SLO targets
            intent: Original deployment intent
            include_near_miss: Whether to include configs within tolerance of SLO
            near_miss_tolerance: How much over SLO to allow (0.2 = 20%)
            weights: Custom weights for balanced score (0-10 scale)
                     Keys: quality, price, latency
            cluster_gpu_types: Detected GPU types from cluster (None = detection
                not attempted, [] = no GPUs detected, non-empty = hard filter
                intersected with user preferences)
            preferred_models: User-specified model IDs to include via estimated
                performance when no benchmark data exists
            enable_estimated: Whether to run roofline estimation for models/GPUs
                without benchmark data (default True)

        Returns:
            Tuple of (list of DeploymentRecommendations with scores, list of warning messages)
        """
        scorer = Scorer()
        all_configs: list[DeploymentRecommendation] = []

        # Determine SLO thresholds for query
        # If including near-miss, relax thresholds by tolerance
        if include_near_miss:
            query_ttft = int(slo_targets.ttft_target_ms * (1 + near_miss_tolerance))
            query_itl = int(slo_targets.itl_target_ms * (1 + near_miss_tolerance))
            query_e2e = int(slo_targets.e2e_target_ms * (1 + near_miss_tolerance))
        else:
            query_ttft = slo_targets.ttft_target_ms
            query_itl = slo_targets.itl_target_ms
            query_e2e = slo_targets.e2e_target_ms

        # Get percentile from SLO targets (default to p95 for backwards compatibility)
        percentile = getattr(slo_targets, "percentile", "p95")

        # Normalize user's preferred GPU types (handles both str and GpuPreference)
        normalized_user_gpus = normalize_gpu_types(intent.preferred_gpu_types, catalog=self.catalog)

        # Extract max_count limits from GpuPreference objects
        gpu_max_counts = extract_gpu_max_counts(intent.preferred_gpu_types, catalog=self.catalog)

        # Determine effective GPU filter by intersecting cluster and user preferences
        # cluster_gpu_types semantics:
        #   None or [] = no cluster detection / detection failed -> use user prefs only
        #   non-empty list = detected cluster GPUs -> intersect with user prefs
        if cluster_gpu_types:
            if normalized_user_gpus:
                effective_gpus = sorted(set(cluster_gpu_types) & set(normalized_user_gpus))
                logger.info(
                    f"Cluster GPUs: {cluster_gpu_types}. "
                    f"User preference: {normalized_user_gpus}. "
                    f"Effective filter: {effective_gpus}"
                )
                if not effective_gpus:
                    logger.warning(
                        "No overlap between cluster GPUs and user preference — "
                        "no configurations possible"
                    )
                    return [], []
            else:
                effective_gpus = sorted(cluster_gpu_types)
                logger.info(f"Using cluster GPUs as filter: {effective_gpus}")
        elif normalized_user_gpus:
            effective_gpus = normalized_user_gpus
            logger.info(f"Filtering by user preferred GPUs: {effective_gpus}")
        else:
            effective_gpus = []

        normalized_gpus = effective_gpus

        # Track whether the GPU filter came from cluster detection (vs user preference)
        # so we can fall back to all GPUs if cluster GPUs have no benchmark data.
        gpu_filter_from_cluster = bool(cluster_gpu_types) and not normalized_user_gpus

        # Query benchmark database for configurations meeting relaxed SLO targets
        matching_configs = self.benchmark_repo.find_configurations_meeting_slo(
            prompt_tokens=traffic_profile.prompt_tokens,
            output_tokens=traffic_profile.output_tokens,
            ttft_p95_max_ms=query_ttft,
            itl_p95_max_ms=query_itl,
            e2e_p95_max_ms=query_e2e,
            min_qps=0,
            percentile=percentile,
            gpu_types=normalized_gpus if normalized_gpus else None,
            exclude_estimated=not enable_estimated,
        )

        # Fallback: if the GPU filter produced no benchmark data, retry
        # without GPU filter so the user still gets recommendations.
        all_warnings: list[str] = []
        gpu_fallback = False
        if not matching_configs and normalized_gpus:
            if gpu_filter_from_cluster:
                msg = (
                    f"No benchmarks found for cluster GPUs "
                    f"({', '.join(normalized_gpus)}). "
                    f"Showing other available GPU configurations."
                )
            else:
                msg = (
                    f"No configurations found for preferred GPUs "
                    f"({', '.join(normalized_user_gpus)}). "
                    f"Showing other available GPU configurations."
                )
            logger.warning(msg)
            all_warnings.append(msg)
            gpu_fallback = True
            matching_configs = self.benchmark_repo.find_configurations_meeting_slo(
                prompt_tokens=traffic_profile.prompt_tokens,
                output_tokens=traffic_profile.output_tokens,
                ttft_p95_max_ms=query_ttft,
                itl_p95_max_ms=query_itl,
                e2e_p95_max_ms=query_e2e,
                min_qps=0,
                percentile=percentile,
                gpu_types=None,
                exclude_estimated=not enable_estimated,
            )

        # Estimated performance flow: generate roofline estimates for
        # preferred models (and optionally catalog models) that lack benchmark data.
        if enable_estimated and preferred_models:
            from .estimator import generate_estimated_configs

            estimated_configs, estimation_warnings = generate_estimated_configs(
                traffic_profile=traffic_profile,
                slo_targets=slo_targets,
                preferred_models=preferred_models,
                existing_benchmarks=matching_configs,
                gpu_types=normalized_gpus if normalized_gpus and not gpu_fallback else None,
                catalog=self.catalog,
                benchmark_repo=self.benchmark_repo,
            )
            all_warnings.extend(estimation_warnings)
            if estimated_configs:
                matching_configs.extend(estimated_configs)
                logger.info(
                    f"Added {len(estimated_configs)} estimated configurations from roofline model"
                )

        # When the user specified preferred models, filter results to only
        # those models.  Fall back to all configs if none of the preferred
        # models produced viable results.
        if preferred_models:
            preferred_set = {m.lower() for m in preferred_models}
            preferred_configs = [
                c for c in matching_configs if c.model_hf_repo.lower() in preferred_set
            ]
            if preferred_configs:
                logger.info(
                    f"Filtering to {len(preferred_configs)} configs for "
                    f"preferred models (from {len(matching_configs)} total)"
                )
                matching_configs = preferred_configs
            else:
                model_list = ", ".join(preferred_models)
                msg = (
                    f"No configurations found for preferred models "
                    f"({model_list}). Showing other available solutions."
                )
                logger.warning(msg)
                all_warnings.append(msg)

        if not matching_configs:
            logger.warning(
                f"No configurations found for traffic profile "
                f"({traffic_profile.prompt_tokens}→{traffic_profile.output_tokens})"
                + (f" with GPUs {normalized_gpus}" if normalized_gpus else "")
            )
            return [], all_warnings

        # Build model lookup from catalog for scoring
        # Models not in catalog will get quality score = 0
        all_models = self.catalog.get_all_models()
        model_lookup = {m.model_id.lower(): m for m in all_models}

        # Process each matching benchmark (no pre-filtering by model list)
        for bench in matching_configs:
            # Filter by max_count if specified
            if gpu_max_counts:
                gpu_type_upper = bench.hardware.upper()
                if (
                    gpu_type_upper in gpu_max_counts
                    and bench.hardware_count > gpu_max_counts[gpu_type_upper]
                ):
                    logger.debug(
                        f"Skipping {bench.model_hf_repo} on {bench.hardware_count}x{bench.hardware} "
                        f"(exceeds max_count={gpu_max_counts[gpu_type_upper]})"
                    )
                    continue

            # Look up model in catalog (may be None if not in catalog)
            model = model_lookup.get(bench.model_hf_repo.lower())

            # Calculate required replicas to handle traffic
            replicas = self._calculate_required_replicas(
                bench.requests_per_second, traffic_profile.expected_qps or 1.0
            )
            if replicas == 0:
                continue  # Zero-throughput benchmark — infeasible config

            # Create GPU config - gpu_count is PER REPLICA, not total
            gpu_config = GPUConfig(
                gpu_type=bench.hardware,
                gpu_count=bench.hardware_count,  # Per-replica GPU count
                tensor_parallel=bench.hardware_count,
                replicas=replicas,
            )

            # Calculate cost using TOTAL GPUs (per-replica * replicas)
            total_gpus = bench.hardware_count * replicas
            cost_per_hour = self.catalog.calculate_gpu_cost(
                bench.hardware, total_gpus, hours_per_month=1
            )

            if cost_per_hour is None:
                logger.warning(f"Could not calculate cost for {bench.hardware}")
                continue

            cost_per_month = cost_per_hour * 730  # ~30 days

            # Calculate latency score and SLO status
            predicted_ttft = int(bench.ttft_p95) if bench.ttft_p95 else 0
            predicted_itl = int(bench.itl_p95) if bench.itl_p95 else 0
            predicted_e2e = int(bench.e2e_p95) if bench.e2e_p95 else 0

            latency_score, slo_status = scorer.score_latency(
                predicted_ttft_ms=predicted_ttft,
                predicted_itl_ms=predicted_itl,
                predicted_e2e_ms=predicted_e2e,
                target_ttft_ms=slo_targets.ttft_target_ms,
                target_itl_ms=slo_targets.itl_target_ms,
                target_e2e_ms=slo_targets.e2e_target_ms,
                use_case=intent.use_case,
                near_miss_tolerance=near_miss_tolerance,
            )

            # Skip if exceeds SLO and we're not including near-miss
            if slo_status == "exceeds" and not include_near_miss:
                continue

            # Calculate quality score
            quality_score_raw = 0.0
            if self._engine is not None:
                from .quality import compute_quality_score

                scorecard = self._engine.get_scores(bench.model_hf_repo, fuzzy=True)
                if scorecard is not None:
                    use_case_key = intent.use_case.lower().replace(" ", "_").replace("-", "_")
                    cat_weights = self._quality_weights.get(use_case_key, {}).get("categories", {})
                    quality_score_raw = compute_quality_score(scorecard, cat_weights)

            quality_score = quality_score_raw
            is_estimated = getattr(bench, "confidence_level", None) == "estimated"
            if quality_score == 0.0 and is_estimated:
                model_size = model.size_parameters if model else bench.model_hf_repo
                quality_score = float(scorer.score_quality_by_size(model_size))

            # Determine model_id and model_name
            # Use catalog info if available, otherwise use benchmark model_hf_repo
            model_id = model.model_id if model else bench.model_hf_repo
            model_name = model.name if model else bench.model_hf_repo

            # Build benchmark_metrics with all percentile values for UI display
            benchmark_metrics = {
                "ttft_mean": int(bench.ttft_mean) if bench.ttft_mean else 0,
                "ttft_p90": int(bench.ttft_p90) if bench.ttft_p90 else 0,
                "ttft_p95": int(bench.ttft_p95) if bench.ttft_p95 else 0,
                "ttft_p99": int(bench.ttft_p99) if bench.ttft_p99 else 0,
                "itl_mean": int(bench.itl_mean) if bench.itl_mean else 0,
                "itl_p90": int(bench.itl_p90) if bench.itl_p90 else 0,
                "itl_p95": int(bench.itl_p95) if bench.itl_p95 else 0,
                "itl_p99": int(bench.itl_p99) if bench.itl_p99 else 0,
                "e2e_mean": int(bench.e2e_mean) if bench.e2e_mean else 0,
                "e2e_p90": int(bench.e2e_p90) if bench.e2e_p90 else 0,
                "e2e_p95": int(bench.e2e_p95) if bench.e2e_p95 else 0,
                "e2e_p99": int(bench.e2e_p99) if bench.e2e_p99 else 0,
                "tps_mean": float(bench.tps_mean) if bench.tps_mean else 0,
                "tps_p90": float(bench.tps_p90) if bench.tps_p90 else 0,
                "tps_p95": float(bench.tps_p95) if bench.tps_p95 else 0,
                "tps_p99": float(bench.tps_p99) if bench.tps_p99 else 0,
                # RPS per replica from benchmark (for card display)
                "requests_per_second": float(bench.requests_per_second)
                if bench.requests_per_second
                else 0,
                # Data validation flag: True = estimated/interpolated, False = real benchmark
                "estimated": getattr(bench, "estimated", False),
                # Classification fields for UI badges
                "source": getattr(bench, "source", "other"),
                "confidence_level": getattr(bench, "confidence_level", "benchmarked"),
            }

            # Build deployment configuration
            configuration = DeploymentConfiguration(
                model_id=model_id,
                model_name=model_name,
                model_uri=getattr(bench, "model_uri", None),
                gpu_config=gpu_config,
                use_case=intent.use_case,
                expected_qps=traffic_profile.expected_qps or 0.0,
                prompt_tokens=traffic_profile.prompt_tokens,
                output_tokens=traffic_profile.output_tokens,
                e2e_target_ms=slo_targets.e2e_target_ms,
            )

            # Build recommendation (price score calculated later after we know min/max)
            recommendation = DeploymentRecommendation(
                intent=intent,
                traffic_profile=traffic_profile,
                slo_targets=slo_targets,
                model_id=model_id,
                model_name=model_name,
                model_uri=getattr(bench, "model_uri", None),
                gpu_config=gpu_config,
                predicted_ttft_p95_ms=predicted_ttft,
                predicted_itl_p95_ms=predicted_itl,
                predicted_e2e_p95_ms=predicted_e2e,
                predicted_throughput_qps=bench.requests_per_second * replicas,
                cost_per_hour_usd=cost_per_hour,
                cost_per_month_usd=cost_per_month,
                meets_slo=(slo_status == "compliant"),
                reasoning=self._generate_reasoning_from_bench(bench, gpu_config, intent, model),
                benchmark_metrics=benchmark_metrics,  # All percentile data for UI
                # Temporary scores without price (will be updated below)
                scores=ConfigurationScores(
                    quality_score=quality_score,
                    price_score=0,  # Placeholder
                    latency_score=latency_score,
                    balanced_score=0.0,  # Placeholder
                    slo_status=slo_status,
                ),
                configuration=configuration,
            )

            all_configs.append(recommendation)

        if not all_configs:
            logger.warning("No viable configurations found for any model")
            return [], all_warnings

        # Now calculate price scores (need min/max across all configs)
        costs = [rec.cost_per_month_usd for rec in all_configs if rec.cost_per_month_usd]
        if costs:
            min_cost = min(costs)
            max_cost = max(costs)

            for rec in all_configs:
                if rec.scores and rec.cost_per_month_usd:
                    # Update price score
                    price_score = scorer.score_price(rec.cost_per_month_usd, min_cost, max_cost)
                    rec.scores.price_score = price_score

                    rec.scores.balanced_score = round(
                        scorer.score_balanced(
                            quality_score=rec.scores.quality_score,
                            price_score=price_score,
                            latency_score=rec.scores.latency_score,
                            weights=weights,
                        ),
                        1,
                    )

        # Count unique models in configurations
        unique_models = {rec.model_id for rec in all_configs}
        logger.info(
            f"Found {len(all_configs)} viable configurations across {len(unique_models)} models"
        )
        return all_configs, all_warnings
