"""llm-d Deployment Generator.

Generates kustomize overlay + Helm values for the llm-d stack,
aligned with how llm-d recommends deployment:
- Model servers via kustomize (referencing llm-d base manifests)
- EPP + InferencePool via Helm (standalone chart)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypedDict

from jinja2 import Environment, FileSystemLoader

from planner.configuration.utils import (
    generate_deployment_id,
    validate_model_id,
    validate_namespace,
)
from planner.shared.schemas import DeploymentConfiguration

logger = logging.getLogger(__name__)


class _PluginDef(TypedDict):
    type: str


class _SchedulingPluginDefBase(TypedDict):
    pluginRef: str  # noqa: N815  (matches upstream EPP schema)


class _SchedulingPluginDef(_SchedulingPluginDefBase, total=False):
    weight: int


class RoutingProfile(TypedDict):
    plugins: list[_PluginDef]
    scheduling_plugins: list[_SchedulingPluginDef]


# ---------------------------------------------------------------------------
# Routing profiles – each defines the EPP plugins and scheduling config
# injected into the Helm values template.
# ---------------------------------------------------------------------------
ROUTING_PROFILES: dict[str, RoutingProfile] = {
    "default": {
        "plugins": [
            {"type": "prefix-cache-scorer"},
            {"type": "decode-filter"},
            {"type": "max-score-picker"},
            {"type": "single-profile-handler"},
        ],
        "scheduling_plugins": [
            {"pluginRef": "decode-filter"},
            {"pluginRef": "max-score-picker"},
            {"pluginRef": "prefix-cache-scorer", "weight": 2},
        ],
    },
    "session-affinity": {
        "plugins": [
            {"type": "prefix-cache-scorer"},
            {"type": "decode-filter"},
            {"type": "max-score-picker"},
            {"type": "single-profile-handler"},
            {"type": "session-affinity-scorer"},
        ],
        "scheduling_plugins": [
            {"pluginRef": "decode-filter"},
            {"pluginRef": "max-score-picker"},
            {"pluginRef": "prefix-cache-scorer", "weight": 2},
            {"pluginRef": "session-affinity-scorer", "weight": 3},
        ],
    },
    "throughput-optimized": {
        "plugins": [
            {"type": "load-aware-scorer"},
            {"type": "decode-filter"},
            {"type": "max-score-picker"},
            {"type": "single-profile-handler"},
        ],
        "scheduling_plugins": [
            {"pluginRef": "decode-filter"},
            {"pluginRef": "max-score-picker"},
            {"pluginRef": "load-aware-scorer", "weight": 3},
        ],
    },
}


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
        routing_profile: str = "default",
    ) -> dict[str, Any]:
        """Generate all llm-d deployment files.

        Returns a dict with: deployment_id, namespace, files, contents.
        """
        if routing_profile not in ROUTING_PROFILES:
            raise ValueError(
                f"Unknown routing profile '{routing_profile}'. "
                f"Valid profiles: {', '.join(sorted(ROUTING_PROFILES))}"
            )

        deployment_id = generate_deployment_id(config)
        context = self._prepare_context(config, deployment_id, namespace)

        profile = ROUTING_PROFILES[routing_profile]
        context["plugins"] = profile["plugins"]
        context["scheduling_plugins"] = profile["scheduling_plugins"]

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
            "files": generated_files,
            "contents": generated_contents,
        }
