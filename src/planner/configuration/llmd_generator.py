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
from planner.shared.schemas import DeploymentConfiguration

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
            # Default to generated_configs/ in current working directory
            self.output_dir = Path.cwd() / "generated_configs"

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_context(
        self,
        config: DeploymentConfiguration,
        deployment_id: str,
        namespace: str,
    ) -> dict[str, Any]:
        """Prepare Jinja2 template context from configuration."""
        model_id = config.model_id
        validate_model_id(model_id)
        validate_namespace(namespace)

        tensor_parallel = config.gpu_config.tensor_parallel

        return {
            "deployment_id": deployment_id,
            "namespace": namespace,
            "model_id": model_id,
            "tensor_parallel": tensor_parallel,
            "gpus_per_replica": tensor_parallel,
            "replicas": config.gpu_config.replicas,
        }

    def generate_all(
        self,
        config: DeploymentConfiguration,
        namespace: str = "default",
        pd_enabled: bool = False,
        prefill_replicas: int = 1,
        decode_replicas: int = 1,
    ) -> dict[str, Any]:
        """Generate all llm-d deployment files.

        Returns a dict with: deployment_id, namespace, files, contents.

        When *pd_enabled* is True, separate prefill and decode patches are
        generated instead of the single ``patch-vllm.yaml``.
        """
        if not 1 <= prefill_replicas <= 32:
            msg = f"prefill_replicas must be between 1 and 32, got {prefill_replicas}"
            raise ValueError(msg)
        if not 1 <= decode_replicas <= 32:
            msg = f"decode_replicas must be between 1 and 32, got {decode_replicas}"
            raise ValueError(msg)

        deployment_id = generate_deployment_id(config)
        context = self._prepare_context(config, deployment_id, namespace)

        context["pd_enabled"] = pd_enabled

        # Build the list of (template, output path, key, extra_context) tuples.
        # All model-server patches use the same unified template with different
        # deployment_name, replica_count, and extra_args.
        patch_template = "patch-modelserver.yaml.j2"

        configs: list[tuple[str, str, str, dict[str, Any]]] = [
            ("kustomization.yaml.j2", "modelserver/kustomization.yaml", "kustomization", {}),
        ]

        if pd_enabled:
            configs.append(
                (
                    patch_template,
                    "modelserver/patch-prefill.yaml",
                    "patch_prefill",
                    {
                        "deployment_name": "prefill",
                        "replica_count": prefill_replicas,
                        "extra_args": [
                            "--kv-connector=nixlv2",
                            "--kv-role=kv_producer",
                            "--enable-chunked-prefill",
                        ],
                    },
                )
            )
            configs.append(
                (
                    patch_template,
                    "modelserver/patch-decode.yaml",
                    "patch_decode",
                    {
                        "deployment_name": "decode",
                        "replica_count": decode_replicas,
                        "extra_args": ["--kv-connector=nixlv2", "--kv-role=kv_consumer"],
                    },
                )
            )
        else:
            configs.append(
                (
                    patch_template,
                    "modelserver/patch-vllm.yaml",
                    "patch_vllm",
                    {
                        "deployment_name": "decode",
                        "replica_count": context["replicas"],
                        "extra_args": [],
                    },
                )
            )

        configs.append(("values.yaml.j2", "scheduler/values.yaml", "helm_values", {}))

        deployment_dir = self.output_dir / deployment_id
        (deployment_dir / "modelserver").mkdir(parents=True, exist_ok=True)
        (deployment_dir / "scheduler").mkdir(parents=True, exist_ok=True)

        generated_files: dict[str, str] = {}
        generated_contents: dict[str, str] = {}

        for template_name, output_rel_path, config_type, extra_ctx in configs:
            template = self.env.get_template(template_name)
            rendered = template.render(**context, **extra_ctx)

            output_path = deployment_dir / output_rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(rendered)

            generated_files[config_type] = str(output_path)
            generated_contents[config_type] = rendered

        return {
            "deployment_id": deployment_id,
            "namespace": namespace,
            "files": generated_files,
            "contents": generated_contents,
        }
