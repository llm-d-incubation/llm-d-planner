"""Planner facade — library API for external callers.

Zero-config usage:
    from planner import Planner
    p = Planner()
    p.load_bundled_benchmarks()
    intent = p.extract_intent("I need a chatbot for 1000 users")
    spec = p.generate_specification(intent)
    recs = p.generate_recommendations(spec)
    bundle = p.generate_deployment(recs.best_quality[0].configuration)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from planner.config import PlannerConfig
from planner.data._resolver import data_path
from planner.errors import PlannerError
from planner.knowledge_base.benchmarks import BenchmarkRepository
from planner.knowledge_base.model_catalog import ModelCatalog
from planner.orchestration.workflow import RecommendationWorkflow
from planner.recommendation.config_finder import ConfigFinder
from planner.recommendation.quality.scoring import build_scoring_engine, load_quality_weights
from planner.shared.schemas import (
    DeploymentBundle,
    DeploymentConfiguration,
    DeploymentIntent,
    DeploymentSpecification,
    RankedRecommendations,
)
from planner.specification.service import SpecificationService

if TYPE_CHECKING:
    from planner.intent_extraction import IntentExtractor
    from planner.llm.client import LLMClient

logger = logging.getLogger(__name__)


class Planner:
    """Zero-config library API for LLM deployment planning.

    Example:
        p = Planner()
        p.load_bundled_benchmarks()
        intent = DeploymentIntent(use_case="chatbot_conversational", user_count=100)
        spec = p.generate_specification(intent)
        recs = p.generate_recommendations(spec)
        bundle = p.generate_deployment(recs.best_quality[0].configuration)
    """

    def __init__(self, config: PlannerConfig | None = None, **kwargs):
        """Initialize Planner.

        Args:
            config: Configuration object. If None, uses defaults (bundled data,
                    no LLM, no quality auto-update).
            **kwargs: Shorthand — any PlannerConfig field can be passed as a
                      keyword argument instead of constructing a config object.
                      e.g. Planner(llm_provider="openai", llm_api_key="sk-...")
        """
        if config is None:
            config = PlannerConfig(**kwargs)
        elif kwargs:
            raise PlannerError("Pass either a PlannerConfig object or keyword arguments, not both.")
        self._config = config

        self._llm_client: LLMClient | None = None
        self._extractor: IntentExtractor | None = None

        data_dir = config.data_dir

        # Initialize in-memory SQLite database for benchmark data
        self._benchmark_repo = BenchmarkRepository(db_path=":memory:")

        # Resolve data paths
        catalog_path = data_path("configuration/model_catalog.json", data_dir)
        gpu_catalog_path = data_path("configuration/gpu_catalog.json", data_dir)
        quality_weights_path = data_path("configuration/quality_weights.json", data_dir)

        # Initialize components
        self._model_catalog = ModelCatalog(
            data_path=catalog_path,
            gpu_catalog_path=gpu_catalog_path,
        )
        self._spec_service = SpecificationService(data_dir=data_dir)

        # Build quality scoring engine
        self._scoring_engine, _ = build_scoring_engine(
            cache_dir=config.quality_cache_dir,
            auto_update=config.quality_auto_update,
            aa_api_key=config.aa_api_key,
        )
        self._quality_weights = load_quality_weights(quality_weights_path)

        # Initialize config finder with scoring engine
        self._config_finder = ConfigFinder(
            benchmark_repo=self._benchmark_repo,
            catalog=self._model_catalog,
            engine=self._scoring_engine,
            quality_weights=self._quality_weights,
        )

        # Initialize workflow orchestrator
        self._workflow = RecommendationWorkflow(
            config_finder=self._config_finder,
            spec_service=self._spec_service,
        )

        logger.info("Planner initialized with data_dir=%s", data_dir or "<bundled>")

    @staticmethod
    def enable_prompt_logging(directory: str | Path) -> None:
        """Enable saving LLM prompts to disk for debugging.

        Off by default. Call this before extract_intent() to save
        prompts for inspection.

        Args:
            directory: Directory where prompt files will be written.
                       Created if it doesn't exist.
        """
        from planner.intent_extraction.extractor import enable_prompt_logging

        enable_prompt_logging(directory)

    def load_bundled_benchmarks(self) -> None:
        """Load bundled BLIS benchmark data.

        This is the simplest way to get started. The bundled benchmarks
        cover the 4 GuideLLM traffic profiles for a curated set of models.
        """
        bundled = data_path("performance/benchmarks_BLIS.json")
        self._load_benchmark_file(bundled)
        logger.info("Loaded bundled BLIS benchmarks")

    def load_benchmarks(self, path: str | Path) -> None:
        """Load benchmark data from a custom JSON file.

        Args:
            path: Path to benchmark JSON file with {"benchmarks": [...]} structure
        """
        self._load_benchmark_file(Path(path))
        logger.info("Loaded benchmarks from %s", path)

    def _load_benchmark_file(self, path: Path) -> None:
        """Parse JSON benchmark file and load into the in-memory repo.

        Args:
            path: Path to benchmark JSON file

        Raises:
            FileNotFoundError: If benchmark file not found
            ValueError: If JSON is invalid or missing 'benchmarks' key
        """
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        benchmarks = data.get("benchmarks", [])
        if not benchmarks:
            logger.warning("No benchmarks found in %s", path)
            return

        # Extract metadata from file (source, confidence_level)
        from planner.knowledge_base.loader import extract_metadata

        meta = extract_metadata(data)
        source = meta["source"] or "local"
        confidence_level = meta["confidence_level"] or "estimated"

        self._benchmark_repo.load_benchmarks(
            benchmarks,
            source=source,
            confidence_level=confidence_level,
        )

    def generate_specification(
        self,
        intent: DeploymentIntent,
    ) -> DeploymentSpecification:
        """Generate a complete deployment specification from intent.

        Args:
            intent: Deployment intent with use_case, user_count, priorities

        Returns:
            Complete specification with SLO targets, workload profile, quality weights

        Raises:
            ValueError: If use_case is unknown or config data is missing
        """
        return self._spec_service.generate(intent)

    def generate_recommendations(
        self,
        spec: DeploymentSpecification,
        min_quality: float | None = None,
        max_cost: float | None = None,
        include_near_miss: bool = True,
        weights: dict[str, int] | None = None,
        enable_estimated: bool = True,
    ) -> RankedRecommendations:
        """Generate ranked recommendations from a specification.

        Args:
            spec: Deployment specification (from generate_specification)
            min_quality: Minimum quality score filter (0-100)
            max_cost: Maximum monthly cost filter (USD)
            include_near_miss: Whether to include near-SLO configurations
            weights: Custom weights for balanced score (0-10 scale)
            enable_estimated: Whether to include estimated performance data

        Returns:
            Ranked recommendations with 4 views (best_quality, lowest_cost, etc.)

        Raises:
            PlannerError: If no benchmarks loaded
        """
        stats = self._benchmark_repo.get_stats()
        if stats.get("total_benchmarks", 0) == 0:
            raise PlannerError(
                "No benchmarks loaded. Call load_bundled_benchmarks() or "
                "load_benchmarks(path) first."
            )

        return self._workflow.generate_recommendations(
            spec=spec,
            min_quality=min_quality,
            max_cost=max_cost,
            include_near_miss=include_near_miss,
            weights=weights,
            enable_estimated=enable_estimated,
        )

    def generate_deployment(
        self,
        config: DeploymentConfiguration,
        namespace: str = "default",
        stack: str = "vllm",
    ) -> DeploymentBundle:
        """Generate Kubernetes YAML deployment bundle from configuration.

        Args:
            config: Deployment configuration (from recommendations)
            namespace: Kubernetes namespace (default: "default")
            stack: Deployment stack ("vllm" or "llm-d")

        Returns:
            Deployment bundle with YAML files ready for kubectl apply

        Raises:
            ValueError: If stack is unknown
        """
        import tempfile

        from planner.configuration.generator import DeploymentGenerator
        from planner.configuration.llmd_generator import LlmdDeploymentGenerator

        # Use a temp dir for output_dir — the facade returns YAML strings,
        # not files, so we don't want to create dirs in the user's CWD.
        output_dir = tempfile.mkdtemp()

        if stack == "llm-d":
            llmd_gen = LlmdDeploymentGenerator(output_dir=output_dir)
            result = llmd_gen.generate_all(config=config, namespace=namespace)
        elif stack == "vllm":
            vllm_gen = DeploymentGenerator(output_dir=output_dir, simulator_mode=False)
            result = vllm_gen.generate_all(config=config, namespace=namespace)
        else:
            raise ValueError(f"Unknown deployment stack: {stack}. Valid options: 'vllm', 'llm-d'")

        return DeploymentBundle(
            deployment_id=result["deployment_id"],
            namespace=namespace,
            stack=stack,
            configuration=config,
            files=result["contents"],
        )

    def extract_intent(self, text: str) -> DeploymentIntent:
        """Extract deployment intent from natural language text.

        Requires LLM provider to be configured via llm_provider parameter
        in __init__ or LLM_PROVIDER environment variable.

        Args:
            text: Natural language description of deployment requirements

        Returns:
            Extracted and validated deployment intent

        Raises:
            PlannerError: If LLM provider not configured
            ImportError: If required LLM dependencies not installed
        """
        import os

        provider = self._config.llm_provider or os.environ.get("LLM_PROVIDER")
        if not provider:
            raise PlannerError(
                "No LLM provider configured. Pass llm_provider to Planner(), e.g.:\n"
                "  Planner(llm_provider='openai', llm_api_key='sk-...')\n"
                "Or set LLM_PROVIDER environment variable."
            )

        # Lazy-initialize LLM client and extractor
        if self._extractor is None:
            from planner.intent_extraction import IntentExtractor
            from planner.llm.factory import create_llm_client

            self._llm_client = create_llm_client(
                provider=provider,
                api_key=self._config.llm_api_key,
                base_url=self._config.llm_base_url,
                model=self._config.llm_model,
            )
            self._extractor = IntentExtractor(self._llm_client)

        intent = self._extractor.extract_intent(text)
        return self._extractor.infer_missing_fields(intent)

    def deploy_bundle_to_cluster(
        self,
        bundle: DeploymentBundle,
    ) -> dict:
        """Deploy a bundle to Kubernetes cluster.

        Requires kubectl in PATH and a configured kubeconfig.

        Args:
            bundle: Deployment bundle (from generate_deployment)

        Returns:
            Result dictionary with success status and applied files

        Raises:
            PlannerError: If deployment fails
        """
        from planner.cluster.manager import KubernetesClusterManager

        manager = KubernetesClusterManager(namespace=bundle.namespace)
        manager.create_namespace_if_not_exists()

        for name, yaml_content in bundle.files.items():
            result = manager.apply_yaml_content(yaml_content)
            if not result["success"]:
                raise PlannerError(f"Failed to apply {name}: {result.get('error')}")

        return {
            "success": True,
            "deployment_id": bundle.deployment_id,
            "files_applied": list(bundle.files.keys()),
        }

    def sync_model_catalog(
        self,
        url: str | None = None,
        token: str | None = None,
    ) -> dict:
        """Sync benchmark data from external Model Catalog API.

        Args:
            url: Model Catalog API URL. Falls back to config, then
                 MODEL_CATALOG_URL env var.
            token: Auth token. Falls back to config, then
                   MODEL_CATALOG_TOKEN env var.

        Returns:
            Sync result with counts and errors

        Raises:
            ImportError: If httpx not installed
            ValueError: If no URL provided
        """
        import os

        api_url = url or self._config.model_catalog_url or os.getenv("MODEL_CATALOG_URL")
        if not api_url:
            raise ValueError(
                "No Model Catalog URL. Pass url= to sync_model_catalog(), "
                "set model_catalog_url in PlannerConfig, or set MODEL_CATALOG_URL env var."
            )

        api_token = token or self._config.model_catalog_token or os.getenv("MODEL_CATALOG_TOKEN")

        try:
            from planner.knowledge_base.model_catalog_client import ModelCatalogClient
            from planner.knowledge_base.model_catalog_sync import sync_model_catalog
        except ImportError:
            raise ImportError(
                "Model Catalog sync requires httpx.\nInstall with: pip install llm-d-planner[quality-sync]"
            ) from None

        client = ModelCatalogClient(base_url=api_url, token=api_token)
        result = sync_model_catalog(
            client=client,
            benchmark_repo=self._benchmark_repo,
            model_catalog=self._model_catalog,
        )

        benchmarks_added = getattr(result, "benchmarks_added", 0)
        models_added = getattr(result, "models_added", 0)
        errors = getattr(result, "errors", [])

        return {
            "benchmarks_added": benchmarks_added,
            "models_added": models_added,
            "errors": errors,
        }
