"""llm-d Deployment Generator.

Generates kustomize overlay + Helm values for the llm-d stack,
aligned with how llm-d recommends deployment:
- Model servers via kustomize (referencing llm-d base manifests)
- EPP + InferencePool via Helm (standalone chart)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from planner.configuration.utils import (
    generate_deployment_id,
    validate_model_id,
    validate_namespace,
)
from planner.shared.schemas import DeploymentRecommendation

logger = logging.getLogger(__name__)


class LlmdDeploymentGenerator:
    """Generate llm-d deployment manifests (kustomize overlay + helm values)."""

    def __init__(self, output_dir: str | None = None):
        template_dir = Path(__file__).parent / "templates" / "llmd"
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            project_root = Path(__file__).parent.parent.parent.parent
            self.output_dir = project_root / "generated_configs"

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_context(
        self,
        recommendation: DeploymentRecommendation,
        deployment_id: str,
        namespace: str,
    ) -> dict[str, Any]:
        """Prepare Jinja2 template context from recommendation."""
        gpu_config = recommendation.gpu_config

        model_id = recommendation.model_id or "unknown"
        validate_model_id(model_id)
        validate_namespace(namespace)

        tensor_parallel = gpu_config.tensor_parallel if gpu_config else 1

        return {
            "deployment_id": deployment_id,
            "namespace": namespace,
            "model_id": model_id,
            "tensor_parallel": tensor_parallel,
            "gpus_per_replica": tensor_parallel,
            "replicas": gpu_config.replicas if gpu_config else 1,
        }

    def generate_all(
        self,
        recommendation: DeploymentRecommendation,
        namespace: str = "default",
        gpus_per_node: int = 8,
    ) -> dict[str, Any]:
        """Generate all llm-d deployment files.

        Returns a dict with: deployment_id, namespace, files, contents.
        """
        deployment_id = generate_deployment_id(recommendation)
        context = self._prepare_context(recommendation, deployment_id, namespace)

        # Multi-node topology awareness
        gpu_config = recommendation.gpu_config
        tensor_parallel = gpu_config.tensor_parallel if gpu_config else 1
        multi_node = tensor_parallel > gpus_per_node
        context["multi_node"] = multi_node
        context["gpus_per_node"] = gpus_per_node

        configs: list[tuple[str, str, str]] = [
            ("kustomization.yaml.j2", "modelserver/kustomization.yaml", "kustomization"),
            ("patch-vllm.yaml.j2", "modelserver/patch-vllm.yaml", "patch_vllm"),
            ("values.yaml.j2", "scheduler/values.yaml", "helm_values"),
        ]

        deployment_dir = self.output_dir / deployment_id
        (deployment_dir / "modelserver").mkdir(parents=True, exist_ok=True)
        (deployment_dir / "scheduler").mkdir(parents=True, exist_ok=True)

        generated_files: dict[str, str] = {}
        generated_contents: dict[str, str] = {}

        for template_name, output_rel_path, config_type in configs:
            template = self.env.get_template(template_name)
            rendered = template.render(**context)

            output_path = deployment_dir / output_rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(rendered)

            generated_files[config_type] = str(output_path)
            generated_contents[config_type] = rendered

        return {
            "deployment_id": deployment_id,
            "namespace": namespace,
            "multi_node": multi_node,
            "files": generated_files,
            "contents": generated_contents,
        }
