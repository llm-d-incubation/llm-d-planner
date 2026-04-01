"""Cost management for GPU recommendations"""

import json
import os
from pathlib import Path
from typing import Any

from llm_optimizer.performance import (
    PerformanceEstimationParams,
    PerformanceEstimationResult,
    run_performance_estimation,
)
from llm_optimizer.predefined.gpus import GPU_SPECS

from planner.capacity_planner import (
    get_model_config_from_hf,
    get_model_info_from_hf,
    get_text_config,
)


class CostManager:
    """Manages GPU cost data with support for defaults and user overrides"""

    def __init__(self, custom_costs: dict[str, float] | None = None):
        """
        Initialize cost manager

        Args:
            custom_costs: Optional dict mapping GPU names to custom costs

        Raises:
            ValueError: If custom_costs contains invalid values
        """
        self.default_costs = self._load_default_costs()

        # Validate custom costs
        if custom_costs:
            for gpu_name, cost in custom_costs.items():
                if cost is not None and (not isinstance(cost, int | float) or cost < 0):
                    raise ValueError(
                        f"Invalid cost for {gpu_name}: {cost}. Cost must be a non-negative number."
                    )

        self.custom_costs = custom_costs or {}

        # Track if any custom costs were provided
        self.has_custom_costs = bool(
            custom_costs and any(v is not None for v in custom_costs.values())
        )

    def get_cost(self, gpu_name: str, num_gpus: int = 1) -> float | None:
        """
        Get cost for GPU configuration

        Args:
            gpu_name: Name of the GPU
            num_gpus: Number of GPUs

        Returns:
            Total cost or None if cost not available
        """
        # Check custom costs first
        if gpu_name in self.custom_costs:
            custom_cost = self.custom_costs[gpu_name]
            if custom_cost is not None:
                return custom_cost * num_gpus

        # Fall back to default costs
        if gpu_name in self.default_costs and "cost" in self.default_costs[gpu_name]:
            return float(self.default_costs[gpu_name]["cost"]) * num_gpus

        return None

    def get_all_costs(self) -> dict[str, float]:
        """
        Get all GPU costs (custom overrides defaults)

        Returns:
            Dict mapping GPU names to cost
        """
        costs = {}

        # Start with defaults (skip non-GPU entries like _disclaimer, _cost_description)
        for gpu_name, data in self.default_costs.items():
            if isinstance(data, dict) and "cost" in data:
                costs[gpu_name] = data["cost"]

        # Override with custom costs (filter out None values)
        for gpu_name, cost in self.custom_costs.items():
            if cost is not None:
                costs[gpu_name] = cost

        return costs

    def has_cost(self, gpu_name: str) -> bool:
        """
        Check if cost data is available for a GPU

        Args:
            gpu_name: Name of the GPU

        Returns:
            True if cost data is available, False otherwise
        """
        return gpu_name in self.custom_costs or gpu_name in self.default_costs

    def is_using_custom_costs(self) -> bool:
        """
        Check if any custom costs are being used

        Returns:
            True if custom costs were provided, False if using only defaults
        """
        return self.has_custom_costs

    def _load_default_costs(self) -> dict[str, Any]:
        """
        Load default costs from JSON file

        Returns:
            Dict mapping GPU names to cost data dictionaries
        """
        # Navigate from planner module to repo root, then to data/configuration
        cost_file = (
            Path(__file__).parent.parent.parent / "data" / "configuration" / "gpu_costs.json"
        )

        if cost_file.exists():
            with open(cost_file) as f:
                return dict(json.load(f))

        return {}


class GPURecommender:
    """Recommends optimal GPU for running LLM inference using BentoML's llm-optimizer roofline algorithm.

    Given a list of models and available GPUs, recommends the best GPU
    for each model based on synthetic performance estimates.
    """

    def __init__(
        self,
        model_id: str,
        input_len: int,
        output_len: int,
        max_gpus: int = 1,
        max_gpus_per_type: dict[str, int] | None = None,
        gpu_list: list | None = None,
        # Performance constraints
        max_ttft: float | None = None,
        max_itl: float | None = None,
        max_latency: float | None = None,
        # Cost parameters
        custom_gpu_costs: dict[str, float] | None = None,
    ):
        """
        Initialize GPU Recommender.

        Args:
            model_id: HuggingFace model ID
            input_len: Input sequence length
            output_len: Output sequence length
            max_gpus: Default maximum number of GPUs to use (applies to all GPU types unless overridden)
            max_gpus_per_type: Optional dict mapping GPU names to their specific max_gpus limit.
                              Example: {"H100": 8, "A100": 4, "L40": 2}
                              If a GPU is not in this dict, it will use the default max_gpus value.
            gpu_list: Optional list of GPU names to evaluate. If None, evaluates all GPUs in GPU_SPECS.
            max_ttft: Maximum time to first token constraint (ms)
            max_itl: Maximum inter-token latency constraint (ms)
            max_latency: Maximum end-to-end latency constraint (s)
            custom_gpu_costs: Optional dict mapping GPU names to custom costs
        """

        # Read HF Token from environment variable
        hf_token = os.getenv("HF_TOKEN", None)

        self.input_len = input_len
        self.output_len = output_len
        self.model_id = model_id
        self.model_info = get_model_info_from_hf(model_id, hf_token)
        self.model_config = get_model_config_from_hf(model_id, hf_token)
        self.text_config = get_text_config(self.model_config)

        self.max_gpus = max_gpus
        self.max_gpus_per_type = max_gpus_per_type or {}
        self.gpu_list = gpu_list if gpu_list else list(GPU_SPECS.keys())

        # Keep track of performance bounds
        self.max_ttft = max_ttft
        self.max_itl = max_itl
        self.max_latency = max_latency

        # Initialize cost manager
        self.cost_manager = CostManager(custom_costs=custom_gpu_costs)

        # Store results after recommendation
        self.gpu_results: dict[str, PerformanceEstimationResult] = {}
        self.failed_gpus: dict[str, str] = {}

    def get_gpu_results(self) -> tuple[dict[str, PerformanceEstimationResult], dict[str, str]]:
        """
        Runs bento's recommendation engine
        """

        gpu_results = {}
        failed_gpus = {}

        # Use the gpu_list from instance attribute
        for gpu_name in self.gpu_list:
            # Use GPU-specific max_gpus if configured, otherwise use default
            num_gpus = self.max_gpus_per_type.get(gpu_name, self.max_gpus)

            constraints = ""
            if self.max_ttft is not None:
                constraints += f"ttft:p95<={self.max_ttft}ms"
            if self.max_itl is not None:
                constraints += f"itl:p95<={self.max_itl}ms"
            if self.max_latency is not None:
                constraints += f"e2e_latency:p95<={self.max_latency}s"

            params = PerformanceEstimationParams(
                model=self.model_id,
                input_len=self.input_len,
                output_len=self.output_len,
                gpu=gpu_name,
                num_gpus=num_gpus,
                framework="vllm",
                target="throughput",
                constraints=constraints,
            )

            try:
                _, result = run_performance_estimation(params)

                # check that best_config exists (if not, it means estimation failed due to constraints)
                _ = (
                    result.best_configs[0]
                    if isinstance(result.best_configs, list)
                    else result.best_configs
                )
                gpu_results[gpu_name] = result
            except ValueError as e:
                msg = f"GPU {gpu_name} not suitable: {e}"
                failed_gpus[gpu_name] = msg
            except Exception as e:
                msg = f"Error estimating performance for GPU {gpu_name}: {e}"
                failed_gpus[gpu_name] = msg

        # Store results in instance variables
        self.gpu_results = gpu_results
        self.failed_gpus = failed_gpus

        return gpu_results, failed_gpus

    def _has_valid_best_latency(self, result: PerformanceEstimationResult) -> bool:
        """
        Check if a GPU result has valid best_latency configuration.

        Args:
            result: Performance estimation result to validate

        Returns:
            True if result has valid best_latency config, False otherwise
        """
        if not (hasattr(result, "best_configs") and result.best_configs):
            return False

        best_latency = (
            result.best_configs.get("best_latency")
            if isinstance(result.best_configs, dict)
            else None
        )
        if not best_latency:
            return False

        # Check if it has actual performance data
        has_data = (
            (
                hasattr(best_latency, "output_throughput_tps")
                and best_latency.output_throughput_tps is not None
            )
            or (hasattr(best_latency, "ttft_ms") and best_latency.ttft_ms is not None)
            or (hasattr(best_latency, "itl_ms") and best_latency.itl_ms is not None)
            or (hasattr(best_latency, "e2e_latency_s") and best_latency.e2e_latency_s is not None)
        )

        return has_data

    def get_gpu_with_highest_throughput(self) -> tuple[str, float] | None:
        """
        Get the GPU with the highest throughput from results.

        Returns:
            Tuple of (gpu_name, throughput_value) or None if no valid data
        """
        if not self.gpu_results:
            self.get_gpu_results()

        best_gpu = None
        best_throughput = -float("inf")

        for gpu_name, result in self.gpu_results.items():
            # Only consider GPUs with valid best_latency configuration
            if not self._has_valid_best_latency(result):
                continue

            best_latency_result = result.best_configs.get("best_latency")
            if (
                hasattr(best_latency_result, "output_throughput_tps")
                and best_latency_result.output_throughput_tps is not None
            ):
                throughput = best_latency_result.output_throughput_tps
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_gpu = gpu_name

        return (best_gpu, best_throughput) if best_gpu else None

    def get_gpu_with_lowest_ttft(self) -> tuple[str, float] | None:
        """
        Get the GPU with the lowest Time to First Token (TTFT) from results.

        Returns:
            Tuple of (gpu_name, ttft_value) or None if no valid data
        """
        if not self.gpu_results:
            self.get_gpu_results()

        best_gpu = None
        best_ttft = float("inf")

        for gpu_name, result in self.gpu_results.items():
            # Only consider GPUs with valid best_latency configuration
            if not self._has_valid_best_latency(result):
                continue

            best_latency_result = result.best_configs.get("best_latency")
            if hasattr(best_latency_result, "ttft_ms") and best_latency_result.ttft_ms is not None:
                ttft = best_latency_result.ttft_ms
                if ttft < best_ttft:
                    best_ttft = ttft
                    best_gpu = gpu_name

        return (best_gpu, best_ttft) if best_gpu else None

    def get_gpu_with_lowest_itl(self) -> tuple[str, float] | None:
        """
        Get the GPU with the lowest Inter-Token Latency (ITL) from results.

        Returns:
            Tuple of (gpu_name, itl_value) or None if no valid data
        """
        if not self.gpu_results:
            self.get_gpu_results()

        best_gpu = None
        best_itl = float("inf")

        for gpu_name, result in self.gpu_results.items():
            # Only consider GPUs with valid best_latency configuration
            if not self._has_valid_best_latency(result):
                continue

            best_latency_result = result.best_configs.get("best_latency")
            if hasattr(best_latency_result, "itl_ms") and best_latency_result.itl_ms is not None:
                itl = best_latency_result.itl_ms
                if itl < best_itl:
                    best_itl = itl
                    best_gpu = gpu_name

        return (best_gpu, best_itl) if best_gpu else None

    def get_gpu_with_lowest_e2e_latency(self) -> tuple[str, float] | None:
        """
        Get the GPU with the lowest End-to-End (E2E) latency from results.

        Returns:
            Tuple of (gpu_name, e2e_latency_value) or None if no valid data
        """
        if not self.gpu_results:
            self.get_gpu_results()

        best_gpu = None
        best_e2e = float("inf")

        for gpu_name, result in self.gpu_results.items():
            # Only consider GPUs with valid best_latency configuration
            if not self._has_valid_best_latency(result):
                continue

            best_latency_result = result.best_configs.get("best_latency")
            if (
                hasattr(best_latency_result, "e2e_latency_s")
                and best_latency_result.e2e_latency_s is not None
            ):
                e2e = best_latency_result.e2e_latency_s
                if e2e < best_e2e:
                    best_e2e = e2e
                    best_gpu = gpu_name

        return (best_gpu, best_e2e) if best_gpu else None

    def get_gpu_with_lowest_cost(self) -> tuple[str, float] | None:
        """
        Get the GPU with the lowest cost from results.

        Returns:
            Tuple of (gpu_name, cost) or None if no valid data
        """
        if not self.gpu_results:
            self.get_gpu_results()

        best_gpu = None
        best_cost = float("inf")

        for gpu_name, result in self.gpu_results.items():
            # Only consider GPUs with valid best_latency configuration
            if not self._has_valid_best_latency(result):
                continue

            # Get number of GPUs used for this configuration
            num_gpus = self.max_gpus_per_type.get(gpu_name, self.max_gpus)
            cost = self.cost_manager.get_cost(gpu_name, num_gpus)

            if cost is not None and cost < best_cost:
                best_cost = cost
                best_gpu = gpu_name

        return (best_gpu, best_cost) if best_gpu else None

    def get_results_sorted_by_cost(self) -> list[tuple[str, float, PerformanceEstimationResult]]:
        """
        Get GPU results sorted by cost (ascending). Only includes GPUs with valid performance data.

        Returns:
            List of tuples (gpu_name, cost, result) sorted by cost
        """
        if not self.gpu_results:
            self.get_gpu_results()

        results_with_cost = []

        for gpu_name, result in self.gpu_results.items():
            # Only consider GPUs with valid best_latency configuration
            if not self._has_valid_best_latency(result):
                continue

            # Get number of GPUs used for this configuration
            num_gpus = self.max_gpus_per_type.get(gpu_name, self.max_gpus)
            cost = self.cost_manager.get_cost(gpu_name, num_gpus)

            if cost is not None:
                results_with_cost.append((gpu_name, cost, result))

        # Sort by cost (ascending)
        results_with_cost.sort(key=lambda x: x[1])

        return results_with_cost

    def get_performance_summary(self, verbose: bool = False) -> dict:
        """
        Get a comprehensive performance summary for all GPUs.

        Args:
            verbose: If True, include concurrency analysis for each GPU

        Returns:
            Dictionary with structured performance data for all GPUs
        """
        if not self.gpu_results:
            self.get_gpu_results()

        summary: dict[str, Any] = {
            "estimated_best_performance": {},
            "gpu_results": {},
        }

        # Get best performance recommendations
        best_throughput = self.get_gpu_with_highest_throughput()
        if best_throughput:
            summary["estimated_best_performance"]["highest_throughput"] = {
                "gpu": best_throughput[0],
                "throughput_tps": round(best_throughput[1], 2),
            }

        best_ttft = self.get_gpu_with_lowest_ttft()
        if best_ttft:
            summary["estimated_best_performance"]["lowest_ttft"] = {
                "gpu": best_ttft[0],
                "ttft_ms": round(best_ttft[1], 2),
            }

        best_itl = self.get_gpu_with_lowest_itl()
        if best_itl:
            summary["estimated_best_performance"]["lowest_itl"] = {
                "gpu": best_itl[0],
                "itl_ms": round(best_itl[1], 2),
            }

        best_e2e = self.get_gpu_with_lowest_e2e_latency()
        if best_e2e:
            summary["estimated_best_performance"]["lowest_e2e_latency"] = {
                "gpu": best_e2e[0],
                "e2e_latency_s": round(best_e2e[1], 4),
            }

        # Get best cost recommendation
        best_cost = self.get_gpu_with_lowest_cost()
        if best_cost:
            summary["estimated_best_performance"]["lowest_cost"] = {
                "gpu": best_cost[0],
                "cost": round(best_cost[1], 2),
            }

        # Extract and format detailed results for each GPU from llm-optimizer output
        for gpu_name, gpu_result in self.gpu_results.items():
            # Only include GPUs that have valid performance data
            if not (hasattr(gpu_result, "best_configs") and gpu_result.best_configs):
                # Move to failed_gpus if not already there
                if gpu_name not in self.failed_gpus:
                    self.failed_gpus[gpu_name] = "No valid performance configuration found"
                continue

            gpu_data: dict[str, Any] = {}

            # Extract best_latency config (concurrency = 1)
            best_latency = (
                gpu_result.best_configs.get("best_latency")
                if isinstance(gpu_result.best_configs, dict)
                else None
            )
            if not best_latency:
                # No valid best_latency config, skip this GPU
                if gpu_name not in self.failed_gpus:
                    self.failed_gpus[gpu_name] = "No valid best_latency configuration found"
                continue

            gpu_data["best_latency"] = {
                "optimal_concurrency": 1,
                "throughput_tps": round(best_latency.output_throughput_tps, 2)
                if best_latency.output_throughput_tps
                else None,
                "ttft_ms": round(best_latency.ttft_ms, 2) if best_latency.ttft_ms else None,
                "itl_ms": round(best_latency.itl_ms, 2) if best_latency.itl_ms else None,
                "e2e_latency_s": round(best_latency.e2e_latency_s, 4)
                if best_latency.e2e_latency_s
                else None,
                "prefill_is_memory_bound": best_latency.prefill_is_memory_bound
                if hasattr(best_latency, "prefill_is_memory_bound")
                else None,
                "decode_is_memory_bound": best_latency.decode_is_memory_bound
                if hasattr(best_latency, "decode_is_memory_bound")
                else None,
            }

            # Extract best_throughput config (optimal concurrency)
            best_throughput_config = (
                gpu_result.best_configs.get("best_output_throughput")
                if isinstance(gpu_result.best_configs, dict)
                else None
            )
            if best_throughput_config:
                gpu_data["best_output_throughput"] = {
                    "optimal_concurrency": best_throughput_config.concurrency
                    if hasattr(best_throughput_config, "concurrency")
                    else None,
                    "throughput_tps": round(best_throughput_config.output_throughput_tps, 2)
                    if best_throughput_config.output_throughput_tps
                    else None,
                    "ttft_ms": round(best_throughput_config.ttft_ms, 2)
                    if best_throughput_config.ttft_ms
                    else None,
                    "itl_ms": round(best_throughput_config.itl_ms, 2)
                    if best_throughput_config.itl_ms
                    else None,
                    "e2e_latency_s": round(best_throughput_config.e2e_latency_s, 4)
                    if best_throughput_config.e2e_latency_s
                    else None,
                    "prefill_is_memory_bound": best_throughput_config.prefill_is_memory_bound
                    if hasattr(best_throughput_config, "prefill_is_memory_bound")
                    else None,
                    "decode_is_memory_bound": best_throughput_config.decode_is_memory_bound
                    if hasattr(best_throughput_config, "decode_is_memory_bound")
                    else None,
                }

            # Add concurrency analysis if verbose
            if (
                verbose
                and hasattr(gpu_result, "concurrency_analysis")
                and gpu_result.concurrency_analysis
            ):
                gpu_data["concurrency_analysis"] = []
                for conc_result in gpu_result.concurrency_analysis:
                    gpu_data["concurrency_analysis"].append(
                        {
                            "optimal_concurrency": conc_result.concurrency
                            if hasattr(conc_result, "concurrency")
                            else None,
                            "throughput_tps": round(conc_result.output_throughput_tps, 2)
                            if conc_result.output_throughput_tps
                            else None,
                            "ttft_ms": round(conc_result.ttft_ms, 2)
                            if conc_result.ttft_ms
                            else None,
                            "itl_ms": round(conc_result.itl_ms, 2) if conc_result.itl_ms else None,
                            "e2e_latency_s": round(conc_result.e2e_latency_s, 4)
                            if conc_result.e2e_latency_s
                            else None,
                        }
                    )

            # Add GPU memory info
            if hasattr(best_latency, "total_memory_gb"):
                gpu_data["total_memory_gb"] = best_latency.total_memory_gb
            if hasattr(best_latency, "model_memory_gb"):
                gpu_data["model_memory_gb"] = round(best_latency.model_memory_gb, 2)
            if hasattr(best_latency, "kv_cache_memory_gb"):
                gpu_data["kv_cache_memory_gb"] = round(best_latency.kv_cache_memory_gb, 2)

            # Add cost information
            num_gpus = self.max_gpus_per_type.get(gpu_name, self.max_gpus)
            cost = self.cost_manager.get_cost(gpu_name, num_gpus)
            if cost is not None:
                gpu_data["cost"] = round(cost, 2)
                gpu_data["num_gpus"] = num_gpus

            summary["gpu_results"][gpu_name] = gpu_data

        return summary
