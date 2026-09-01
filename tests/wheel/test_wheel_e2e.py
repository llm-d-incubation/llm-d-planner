"""Wheel E2E test: verify the installed package works end-to-end.

Run after `uv build && uv pip install dist/*.whl` in a clean venv.
Tests the Planner library API with canned data to catch packaging issues
(missing data files, broken imports, missing __init__.py).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestWheelE2E:
    """Verify the installed wheel works for the full pipeline."""

    def test_package_version_is_valid(self):
        """Verify hatch-vcs resolved a real version, not a fallback."""
        from importlib.metadata import version

        v = version("llm-d-planner")
        assert v != "0.0.0", "Version is fallback '0.0.0' — hatch-vcs may not have git history"
        assert "unknown" not in v.lower(), f"Version contains 'unknown': {v}"

    def test_import_and_data_resolution(self):
        """Verify core imports work and bundled data files are accessible."""
        from planner import Planner
        from planner.data._resolver import data_path

        # Verify bundled data files exist
        assert data_path("configuration/usecase_slo_workload.json").exists()
        assert data_path("configuration/model_catalog.json").exists()
        assert data_path("configuration/quality_weights.json").exists()

        # Verify Planner can be instantiated
        p = Planner()
        assert p is not None

    def test_bundled_benchmarks_load(self):
        """Verify bundled benchmark data can be loaded."""
        from planner import Planner

        p = Planner()
        p.load_bundled_benchmarks()

    def test_full_pipeline_from_wheel(self, monkeypatch):
        """Full pipeline using the installed wheel with canned data."""
        monkeypatch.setenv("PLANNER_DETECT_CLUSTER_GPUS", "false")

        from planner import Planner
        from planner.cluster.manager import KubernetesClusterManager
        from planner.shared.schemas import DeploymentIntent

        # Load canned quality data
        fixture_path = FIXTURES_DIR / "test_quality_scores.json"
        with open(fixture_path) as f:
            quality_data = json.load(f)

        def _mock_build_scoring_engine(cache_dir=None, auto_update=None, aa_api_key=None):
            from quality_scoring.engine import ScoringEngine

            engine = ScoringEngine(
                arena_rows=quality_data["arena_rows"],
                aa_models=quality_data["aa_models"],
            )
            metadata = {
                "arena_count": len(quality_data["arena_rows"]),
                "arena_fetched": "2026-08-14T00:00:00Z",
                "aa_count": len(quality_data["aa_models"]),
                "aa_fetched": "2026-08-14T00:00:00Z",
            }
            return engine, metadata

        # Load canned benchmarks
        benchmark_path = FIXTURES_DIR / "test_benchmarks.json"

        # Patch at planner.planner (import-time binding), not the source module
        with patch(
            "planner.planner.build_scoring_engine",
            side_effect=_mock_build_scoring_engine,
        ):
            p = Planner()
            p.load_benchmarks(str(benchmark_path))

        # Stage 2: Generate specification (skip extract_intent — no LLM in wheel test)
        intent = DeploymentIntent(
            use_case="chatbot_conversational",
            user_count=1000,
            quality_priority="high",
            cost_priority="low",
            latency_priority="medium",
            preferred_models=["meta-llama/llama-3.1-8b-instruct"],
        )

        spec = p.generate_specification(intent)
        assert spec.slo_targets.ttft_target_ms > 0

        # Stage 3: Generate recommendations
        recs = p.generate_recommendations(spec, enable_estimated=True)
        assert recs.total_configs_evaluated > 0

        if recs.balanced:
            # Stage 4: Generate deployment bundle
            top_config = recs.balanced[0].configuration
            assert top_config is not None
            bundle = p.generate_deployment(
                config=top_config,
                namespace="wheel-test-ns",
                stack="vllm",
            )
            assert "inferenceservice" in bundle.files

            # Stage 5: Deploy (mocked)
            mock_manager = MagicMock(spec=KubernetesClusterManager)
            mock_manager.create_namespace_if_not_exists.return_value = None
            mock_manager.apply_yaml_content.return_value = {"success": True}

            with patch(
                "planner.cluster.manager.KubernetesClusterManager",
                return_value=mock_manager,
            ):
                result = p.deploy_bundle_to_cluster(bundle)
                assert result["success"] is True
