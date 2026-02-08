"""Data access layer for model catalog."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelInfo:
    """Model metadata."""

    def __init__(self, data: dict):
        self.model_id = data["model_id"]
        self.name = data["name"]
        self.provider = data["provider"]
        self.family = data["family"]
        self.size_parameters = data["size_parameters"]
        self.context_length = data["context_length"]
        self.supported_tasks = data["supported_tasks"]
        self.domain_specialization = data["domain_specialization"]
        self.license = data["license"]
        self.license_type = data["license_type"]
        self.min_gpu_memory_gb = data["min_gpu_memory_gb"]
        self.recommended_for = data["recommended_for"]
        self.approval_status = data["approval_status"]
        self.notes = data.get("notes", "")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "provider": self.provider,
            "family": self.family,
            "size_parameters": self.size_parameters,
            "context_length": self.context_length,
            "supported_tasks": self.supported_tasks,
            "domain_specialization": self.domain_specialization,
            "license": self.license,
            "license_type": self.license_type,
            "min_gpu_memory_gb": self.min_gpu_memory_gb,
            "recommended_for": self.recommended_for,
            "approval_status": self.approval_status,
            "notes": self.notes,
        }


class GPUType:
    """GPU type metadata with multi-provider pricing."""

    def __init__(self, data: dict):
        self.gpu_type = data["gpu_type"]
        self.aliases = data.get("aliases", [data["gpu_type"]])  # Default to primary name
        self.memory_gb = data["memory_gb"]
        self.compute_capability = data["compute_capability"]
        self.typical_use_cases = data["typical_use_cases"]
        self.cost_per_hour_usd = data["cost_per_hour_usd"]  # Base/minimum price
        # Cloud provider-specific pricing (optional)
        self.cost_per_hour_aws = data.get("cost_per_hour_aws")
        self.cost_per_hour_gcp = data.get("cost_per_hour_gcp")
        self.cost_per_hour_azure = data.get("cost_per_hour_azure")
        self.availability = data["availability"]
        self.notes = data.get("notes", "")

    def get_cost_for_provider(self, provider: str | None = None) -> float:
        """
        Get cost per hour for a specific cloud provider.
        
        Args:
            provider: Cloud provider ("aws", "gcp", "azure") or None for base price
            
        Returns:
            Cost per hour in USD
        """
        if provider == "aws" and self.cost_per_hour_aws:
            return self.cost_per_hour_aws
        elif provider == "gcp" and self.cost_per_hour_gcp:
            return self.cost_per_hour_gcp
        elif provider == "azure" and self.cost_per_hour_azure:
            return self.cost_per_hour_azure
        return self.cost_per_hour_usd

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "gpu_type": self.gpu_type,
            "aliases": self.aliases,
            "memory_gb": self.memory_gb,
            "compute_capability": self.compute_capability,
            "typical_use_cases": self.typical_use_cases,
            "cost_per_hour_usd": self.cost_per_hour_usd,
            "cost_per_hour_aws": self.cost_per_hour_aws,
            "cost_per_hour_gcp": self.cost_per_hour_gcp,
            "cost_per_hour_azure": self.cost_per_hour_azure,
            "availability": self.availability,
            "notes": self.notes,
        }


class ModelCatalog:
    """Repository for model and GPU metadata."""

    def __init__(self, data_path: Path | None = None):
        """
        Initialize model catalog.

        Args:
            data_path: Path to model_catalog.json
        """
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent.parent / "data" / "configuration" / "model_catalog.json"

        self.data_path = data_path
        self._models: dict[str, ModelInfo] = {}
        self._gpu_types: dict[str, GPUType] = {}
        self._gpu_alias_lookup: dict[str, GPUType] = {}  # Maps aliases to GPUType
        self._load_data()

    def _load_data(self):
        """Load catalog data from JSON file."""
        try:
            with open(self.data_path) as f:
                data = json.load(f)

                # Load models
                for model_data in data["models"]:
                    model = ModelInfo(model_data)
                    self._models[model.model_id] = model

                # Load GPU types and build alias lookup
                for gpu_data in data["gpu_types"]:
                    gpu = GPUType(gpu_data)
                    self._gpu_types[gpu.gpu_type] = gpu
                    # Add all aliases to lookup (case-insensitive)
                    for alias in gpu.aliases:
                        self._gpu_alias_lookup[alias.lower()] = gpu

                logger.info(
                    f"Loaded {len(self._models)} models and {len(self._gpu_types)} GPU types"
                )
        except Exception as e:
            logger.error(f"Failed to load model catalog from {self.data_path}: {e}")
            raise

    def get_model(self, model_id: str) -> ModelInfo | None:
        """Get model by ID."""
        return self._models.get(model_id)

    def get_gpu_type(self, gpu_type: str) -> GPUType | None:
        """Get GPU type metadata.

        Supports lookup by primary gpu_type name or any alias (case-insensitive).

        Args:
            gpu_type: GPU type identifier or alias

        Returns:
            GPUType if found, None otherwise
        """
        # First try exact match on primary name
        if gpu_type in self._gpu_types:
            return self._gpu_types[gpu_type]
        # Then try alias lookup (case-insensitive)
        return self._gpu_alias_lookup.get(gpu_type.lower())

    def find_models_for_use_case(self, use_case: str) -> list[ModelInfo]:
        """
        Find models recommended for a specific use case.

        Args:
            use_case: Use case identifier

        Returns:
            List of recommended models
        """
        return [
            model
            for model in self._models.values()
            if use_case in model.recommended_for and model.approval_status == "approved"
        ]

    def find_models_by_domain(self, domain: str) -> list[ModelInfo]:
        """
        Find models specialized for a domain.

        Args:
            domain: Domain (e.g., "code", "multilingual")

        Returns:
            List of models with domain specialization
        """
        return [
            model
            for model in self._models.values()
            if domain in model.domain_specialization and model.approval_status == "approved"
        ]

    def find_models_by_task(self, task: str) -> list[ModelInfo]:
        """
        Find models supporting a specific task.

        Args:
            task: Task type

        Returns:
            List of models supporting this task
        """
        return [
            model
            for model in self._models.values()
            if task in model.supported_tasks and model.approval_status == "approved"
        ]

    def get_all_models(self) -> list[ModelInfo]:
        """Get all approved models."""
        return [m for m in self._models.values() if m.approval_status == "approved"]

    def get_all_gpu_types(self) -> list[GPUType]:
        """Get all GPU types."""
        return list(self._gpu_types.values())

    def calculate_gpu_cost(
        self,
        gpu_type: str,
        gpu_count: int,
        hours_per_month: float = 730,  # ~30 days
        provider: str | None = None,
    ) -> float | None:
        """
        Calculate monthly GPU cost with proper scaling for multi-GPU setups.

        Cost Calculation Formula:
            Monthly_Cost = GPU_hourly_rate × GPU_count × hours_per_month
            
        For multi-GPU deployments:
            - TP=2 (2 GPUs): 2 × base_cost
            - TP=4 (4 GPUs): 4 × base_cost  
            - TP=8 (8 GPUs): 8 × base_cost
            
        Example costs (730 hours/month):
            - 1x A100-40: $1.50 × 1 × 730 = $1,095/mo
            - 2x A100-80: $2.00 × 2 × 730 = $2,920/mo
            - 4x H100:    $2.70 × 4 × 730 = $7,884/mo
            - 8x H100:    $2.70 × 8 × 730 = $15,768/mo
            - 4x H200:    $3.50 × 4 × 730 = $10,220/mo
            - 8x B200:    $5.50 × 8 × 730 = $32,120/mo

        Args:
            gpu_type: GPU type identifier
            gpu_count: Total number of GPUs (tensor_parallel × replicas)
            hours_per_month: Hours per month (default: 730 for 24/7 operation)
            provider: Optional cloud provider ("aws", "gcp", "azure") for 
                     provider-specific pricing, None for base/minimum price

        Returns:
            Monthly cost in USD, or None if GPU type not found
        """
        gpu = self.get_gpu_type(gpu_type)
        if not gpu:
            logger.warning(f"GPU type not found: {gpu_type}")
            return None

        hourly_rate = gpu.get_cost_for_provider(provider)
        monthly_cost = hourly_rate * gpu_count * hours_per_month
        
        logger.debug(
            f"Cost calculation: {gpu_count}x {gpu_type} @ ${hourly_rate:.2f}/hr "
            f"× {hours_per_month:.0f}hrs = ${monthly_cost:,.0f}/mo"
        )
        
        return monthly_cost
    
    def get_cost_breakdown(
        self,
        gpu_type: str,
        tensor_parallel: int,
        replicas: int,
    ) -> dict | None:
        """
        Get detailed cost breakdown for a deployment configuration.

        Args:
            gpu_type: GPU type identifier
            tensor_parallel: Number of GPUs per replica (TP degree)
            replicas: Number of independent replicas

        Returns:
            Dictionary with cost breakdown, or None if GPU not found
        """
        gpu = self.get_gpu_type(gpu_type)
        if not gpu:
            return None

        total_gpus = tensor_parallel * replicas
        hours_per_month = 730
        
        return {
            "gpu_type": gpu.gpu_type,
            "tensor_parallel": tensor_parallel,
            "replicas": replicas,
            "total_gpus": total_gpus,
            "hourly_rate_base": gpu.cost_per_hour_usd,
            "hourly_rate_aws": gpu.cost_per_hour_aws,
            "hourly_rate_gcp": gpu.cost_per_hour_gcp,
            "hourly_rate_azure": gpu.cost_per_hour_azure,
            "cost_per_hour_total": gpu.cost_per_hour_usd * total_gpus,
            "cost_per_month_base": gpu.cost_per_hour_usd * total_gpus * hours_per_month,
            "cost_per_month_aws": (gpu.cost_per_hour_aws or gpu.cost_per_hour_usd) * total_gpus * hours_per_month,
            "cost_per_month_gcp": (gpu.cost_per_hour_gcp or gpu.cost_per_hour_usd) * total_gpus * hours_per_month,
            "cost_per_month_azure": (gpu.cost_per_hour_azure or gpu.cost_per_hour_usd) * total_gpus * hours_per_month,
        }
