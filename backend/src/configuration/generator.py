"""YAML Generation Module for KServe/vLLM Deployments.

This module generates production-ready Kubernetes YAML configurations for
LLM inference deployments using Jinja2 templates.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from src.knowledge_base.model_catalog import ModelCatalog
from src.shared.schemas import DeploymentRecommendation

logger = logging.getLogger(__name__)


class DeploymentGenerator:
    """Generate deployment configurations from recommendations."""

    # vLLM version to use
    VLLM_VERSION = "v0.6.2"

    def __init__(self, output_dir: str | None = None, simulator_mode: bool = False):
        """
        Initialize the deployment generator.

        Args:
            output_dir: Directory to write generated YAML files (default: generated_configs/)
            simulator_mode: If True, use vLLM simulator instead of real vLLM (no GPU required)
        """
        # Set up template environment
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)), trim_blocks=True, lstrip_blocks=True
        )

        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Default to generated_configs/ in project root
            project_root = Path(__file__).parent.parent.parent.parent
            self.output_dir = project_root / "generated_configs"

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Simulator mode (for development/testing without GPUs)
        self.simulator_mode = simulator_mode

        # Model catalog for GPU pricing lookup
        self._catalog = ModelCatalog()

        logger.info(
            f"DeploymentGenerator initialized with output_dir: {self.output_dir}, simulator_mode: {simulator_mode}"
        )

    def generate_deployment_id(self, recommendation: DeploymentRecommendation) -> str:
        """
        Generate a unique deployment ID that meets Kubernetes naming requirements:
        - Must start with a letter
        - Only lowercase alphanumeric and hyphens
        - Max 44 characters (KServe adds "-predictor-default" suffix, total must be ≤63)

        Args:
            recommendation: Deployment recommendation

        Returns:
            Deployment ID (e.g., "chatbot-mistral-7b-20251003143022")
        """
        import re

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  # 14 chars: YYYYMMDDHHMMSS
        use_case = recommendation.intent.use_case.replace("_", "-")

        # Clean model name: remove special chars, keep alphanumeric and hyphens
        model_name = recommendation.model_id.split("/")[-1].lower()
        model_name = re.sub(r"[^a-z0-9-]", "-", model_name)
        # Remove consecutive hyphens
        model_name = re.sub(r"-+", "-", model_name).strip("-")

        # Build ID
        deployment_id = f"{use_case}-{model_name}-{timestamp}"

        # KServe creates names like "{deployment_id}-predictor-default" (adds 19 chars)
        # So deployment_id must be max 44 chars to stay under 63 char DNS limit
        max_deployment_id_len = 44

        if len(deployment_id) > max_deployment_id_len:
            # Truncate model name to fit
            max_model_len = (
                max_deployment_id_len - len(use_case) - len(timestamp) - 2
            )  # 2 for hyphens
            model_name = model_name[:max_model_len].rstrip("-")
            deployment_id = f"{use_case}-{model_name}-{timestamp}"

        return deployment_id

    def _prepare_template_context(
        self,
        recommendation: DeploymentRecommendation,
        deployment_id: str,
        namespace: str = "default",
    ) -> dict[str, Any]:
        """
        Prepare context dictionary for Jinja2 templates.

        Args:
            recommendation: Deployment recommendation
            deployment_id: Generated deployment ID
            namespace: Kubernetes namespace

        Returns:
            Context dictionary with all template variables
        """
        gpu_config = recommendation.gpu_config
        traffic = recommendation.traffic_profile
        slo = recommendation.slo_targets

        # Calculate GPU hourly rate from ModelCatalog
        gpu_info = self._catalog.get_gpu_type(gpu_config.gpu_type)
        gpu_hourly_rate = gpu_info.cost_per_hour_usd if gpu_info else 1.0

        # Determine resource requests based on GPU type
        gpu_type = gpu_config.gpu_type
        if "H100" in gpu_type or "H200" in gpu_type or "B200" in gpu_type or "A100" in gpu_type:
            # High-end GPUs: H100, H200, B200, A100-40, A100-80
            cpu_request = "24"
            cpu_limit = "48"
            memory_request = "128Gi"
            memory_limit = "256Gi"
        else:
            # Entry-level GPUs: L4, etc.
            cpu_request = "8"
            cpu_limit = "16"
            memory_request = "32Gi"
            memory_limit = "64Gi"

        # Calculate autoscaling parameters
        min_replicas = max(1, gpu_config.replicas // 2)
        max_replicas = gpu_config.replicas * 2

        # Determine max model length based on use case
        max_model_len_map = {
            "chatbot": 4096,
            "customer_service": 4096,
            "code_generation": 8192,
            "summarization": 8192,
            "content_creation": 8192,
            "qa_retrieval": 4096,
            "batch_analytics": 16384,
        }
        max_model_len = max_model_len_map.get(recommendation.intent.use_case, 4096)

        # Calculate max_num_seqs based on expected QPS and latency
        # Rule of thumb: concurrent requests = QPS × avg_latency_seconds
        avg_latency_sec = slo.e2e_p95_target_ms / 1000.0
        max_num_seqs = max(32, int(traffic.expected_qps * avg_latency_sec * 1.5))

        # Max batched tokens (vLLM parameter)
        max_num_batched_tokens = max_num_seqs * (traffic.prompt_tokens + traffic.output_tokens)

        context = {
            # Deployment metadata
            "deployment_id": deployment_id,
            "namespace": namespace,
            "model_id": recommendation.model_id,
            "model_name": recommendation.model_name,
            "use_case": recommendation.intent.use_case,
            "reasoning": recommendation.reasoning,
            "generated_at": datetime.now().isoformat(),
            # Simulator mode
            "simulator_mode": self.simulator_mode,
            # GPU configuration
            "gpu_type": gpu_config.gpu_type,
            "gpu_count": gpu_config.gpu_count,
            "tensor_parallel": gpu_config.tensor_parallel,
            "gpus_per_replica": gpu_config.tensor_parallel,  # GPUs per pod
            # vLLM configuration
            "vllm_version": self.VLLM_VERSION,
            "dtype": "auto",  # Let vLLM auto-detect (float16, bfloat16, etc.)
            "gpu_memory_utilization": 0.9,  # Use 90% of GPU memory
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_batch_size": max_num_seqs,
            "enable_prefix_caching": True,  # Enable KV cache optimization
            # Autoscaling
            "min_replicas": min_replicas,
            "max_replicas": max_replicas,
            "autoscaling_metric": "inference_requests_concurrency",
            "autoscaling_target": "10",  # Target 10 concurrent requests per pod
            "queue_depth_threshold": "20",
            # Resource requests
            "cpu_request": cpu_request,
            "cpu_limit": cpu_limit,
            "memory_request": memory_request,
            "memory_limit": memory_limit,
            # SLO targets (p95)
            "ttft_target": slo.ttft_p95_target_ms,
            "itl_target": slo.itl_p95_target_ms,
            "e2e_target": slo.e2e_p95_target_ms,
            "target_qps": traffic.expected_qps,
            # Traffic profile
            "expected_qps": traffic.expected_qps,
            "prompt_tokens": traffic.prompt_tokens,
            "output_tokens": traffic.output_tokens,
            # Cost estimation
            "cost_per_hour": recommendation.cost_per_hour_usd,
            "cost_per_month": recommendation.cost_per_month_usd,
            "gpu_hourly_rate": gpu_hourly_rate,
            # Intent metadata
            "user_count": recommendation.intent.user_count,
        }

        return context

    def generate_all(
        self, recommendation: DeploymentRecommendation, namespace: str = "default"
    ) -> dict[str, str]:
        """
        Generate all deployment YAML files.

        Args:
            recommendation: Deployment recommendation
            namespace: Kubernetes namespace

        Returns:
            Dictionary mapping config type to file path
        """
        deployment_id = self.generate_deployment_id(recommendation)
        context = self._prepare_template_context(recommendation, deployment_id, namespace)

        generated_files = {}

        # Generate each config file
        configs = [
            ("kserve-inferenceservice.yaml.j2", f"{deployment_id}-inferenceservice.yaml"),
            ("vllm-config.yaml.j2", f"{deployment_id}-vllm-config.yaml"),
            ("autoscaling.yaml.j2", f"{deployment_id}-autoscaling.yaml"),
            ("servicemonitor.yaml.j2", f"{deployment_id}-servicemonitor.yaml"),
        ]

        for template_name, output_filename in configs:
            try:
                template = self.env.get_template(template_name)
                rendered = template.render(**context)

                output_path = self.output_dir / output_filename
                with open(output_path, "w") as f:
                    f.write(rendered)

                # Extract config type from template name (remove .j2 suffix)
                config_type = template_name.replace(".yaml.j2", "").replace("kserve-", "")
                generated_files[config_type] = str(output_path)

                logger.info(f"Generated {config_type}: {output_path}")

            except Exception as e:
                logger.error(f"Failed to generate {template_name}: {e}")
                raise

        # Store deployment metadata
        metadata = {
            "deployment_id": deployment_id,
            "namespace": namespace,
            "generated_at": context["generated_at"],
            "files": generated_files,
            "recommendation": recommendation.model_dump(),
        }

        return {
            "deployment_id": deployment_id,
            "namespace": namespace,
            "files": generated_files,
            "metadata": metadata,
        }

    def generate_kserve_yaml(
        self,
        recommendation: DeploymentRecommendation,
        deployment_id: str | None = None,
        namespace: str = "default",
    ) -> str:
        """
        Generate only the KServe InferenceService YAML.

        Args:
            recommendation: Deployment recommendation
            deployment_id: Optional deployment ID (auto-generated if not provided)
            namespace: Kubernetes namespace

        Returns:
            Path to generated YAML file
        """
        if not deployment_id:
            deployment_id = self.generate_deployment_id(recommendation)

        context = self._prepare_template_context(recommendation, deployment_id, namespace)
        template = self.env.get_template("kserve-inferenceservice.yaml.j2")
        rendered = template.render(**context)

        output_path = self.output_dir / f"{deployment_id}-inferenceservice.yaml"
        with open(output_path, "w") as f:
            f.write(rendered)

        logger.info(f"Generated KServe YAML: {output_path}")
        return str(output_path)
