"""Authenticate model loading before composing the optimizer's calculations."""

import json
from dataclasses import replace
from typing import cast

from huggingface_hub import HfApi, hf_hub_download
from llm_optimizer.common import (
    ModelConfig,
    calculate_model_parameters_from_config,
    infer_precision_from_config,
)
from llm_optimizer.performance import (
    PerformanceEstimationParams,
    PerformanceEstimationResult,
    calculate_concurrency_limits,
    estimate_performance_under_constraints,
    find_best_performance,
    find_optimal_concurrency_threshold,
    parse_slo_constraints,
)
from llm_optimizer.tuning import (
    generate_advanced_tuning_configs,
    generate_llm_optimizer_commands,
    generate_simple_tuning_configs,
)


def _infer_precision(config: dict, model_id: str, token: str | bool | None) -> str:
    """Preserve optimizer precision precedence, authenticating its size fallback."""
    # The dtype sentinel makes this call inspect quantization without falling
    # through to name/config-content hints or model-size inference.
    if (
        infer_precision_from_config(
            {"quantization_config": config.get("quantization_config"), "torch_dtype": "float16"}
        )
        == "fp8"
    ):
        return "fp8"
    model_name = model_id.lower()
    for hints, precision in (
        (("fp8",), "fp8"),
        (("bf16", "bfloat16"), "bf16"),
        (("fp16", "float16"), "fp16"),
    ):
        if any(hint in model_name for hint in hints):
            return precision
    # Omitting model_id prevents the dependency from making anonymous HF calls.
    precision = cast(str, infer_precision_from_config(config))
    if (
        precision != "fp16"
        or config.get("torch_dtype") in ("float16", "torch.float16")
        or any(hint in config.get("_name_or_path", "").lower() for hint in ("fp16", "float16"))
    ):
        return precision
    total_params = calculate_model_parameters_from_config(config)
    if not total_params:
        return precision
    try:
        info = HfApi(token=token).model_info(model_id, files_metadata=True)
        total_size = sum(
            sibling.size or 0
            for sibling in info.siblings or []
            if sibling.rfilename.endswith(".safetensors")
        )
    except Exception:
        return precision
    # Match the optimizer's bytes-per-parameter thresholds and fp16 fallback.
    if total_size:
        bytes_per_param = total_size / total_params
        if bytes_per_param <= 1.3:
            return "fp8"
        if bytes_per_param <= 2.5:
            return "bf16"
        if bytes_per_param <= 4.5:
            return "fp32"
    return precision


def get_optimizer_model_config(model_id: str, hf_token: str | None) -> ModelConfig:
    """Load raw config with the selected token, preserving optimizer input semantics."""
    token = False if hf_token == "" else hf_token
    try:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json", token=token)
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Could not download or read config.json for {model_id}: {e}") from e
    try:
        hidden_size = config["hidden_size"]
        num_layers = config["num_hidden_layers"]
        vocab_size = config["vocab_size"]
        num_heads = config.get("num_attention_heads", 0)
        return ModelConfig(
            num_params=calculate_model_parameters_from_config(config),
            num_layers=num_layers,
            hidden_dim=hidden_size,
            vocab_size=vocab_size,
            num_heads=num_heads,
            num_kv_heads=config.get("num_key_value_heads", num_heads),
            inferred_precision=_infer_precision(config, model_id, token),
        )
    except KeyError as e:
        raise KeyError(f"Could not find required key {e} in config.json for {model_id}") from e


def run_performance_estimation(
    params: PerformanceEstimationParams, model_config: ModelConfig
) -> tuple[PerformanceEstimationParams, PerformanceEstimationResult]:
    """Compose optimizer calculations for Planner's vLLM throughput estimates.

    The optimizer's top-level runner reloads config.json without accepting a token.
    Use its public calculation and tuning functions with our authenticated config.
    """
    params = replace(
        params,
        precision=model_config.inferred_precision if params.precision is None else params.precision,
    )
    calculation_args = {
        "num_gpus": params.num_gpus,
        "gpu_name": params.gpu,
        "model_config": model_config,
        "precision": params.precision,
        "input_length": params.input_len,
        "output_length": params.output_len,
    }
    constraints = parse_slo_constraints(params.constraints) if params.constraints else []
    best_configs = find_best_performance(**calculation_args)
    concurrency_limits = calculate_concurrency_limits(**calculation_args)
    optimal_concurrency = find_optimal_concurrency_threshold(**calculation_args)
    constrained_result = None
    if constraints:
        constrained_result = estimate_performance_under_constraints(
            **calculation_args, constraints=constraints
        )
        if constrained_result is None:
            raise ValueError("Cannot satisfy the given constraints with this configuration")

    reference_concurrency = optimal_concurrency
    if constrained_result is not None:
        reference_concurrency = constrained_result.concurrency
    elif best_configs["best_output_throughput"]:
        reference_concurrency = best_configs["best_output_throughput"].concurrency

    tuning_commands = {}
    for stage, generate_configs in (
        ("simple", generate_simple_tuning_configs),
        ("advanced", generate_advanced_tuning_configs),
    ):
        configs = generate_configs(
            framework=params.framework,
            num_gpus=params.num_gpus,
            gpu_name=params.gpu,
            model_config=model_config,
            optimal_concurrency=reference_concurrency,
            precision=params.precision,
            sequence_length=params.input_len,
        )
        commands = generate_llm_optimizer_commands(
            configs=configs,
            model_id=params.model,
            input_length=params.input_len,
            output_length=params.output_len,
            num_gpus=params.num_gpus,
            constraints=params.constraints,
            dataset=params.dataset,
        )
        tuning_commands[stage] = {params.framework: {"configs": configs, "commands": commands}}

    return params, PerformanceEstimationResult(
        model_config=model_config,
        best_configs=best_configs,
        concurrency_limits=concurrency_limits,
        optimal_concurrency=optimal_concurrency,
        constrained_result=constrained_result,
        tuning_commands=tuning_commands,
    )
