"""Unit and integration tests for PostgreSQL migration.

Tests cover:
1. BenchmarkRepository - PostgreSQL connection and queries
2. Traffic profile exact matching
3. p95/ITL metric usage
4. SLO filtering and compliance checking
"""

import pytest
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from src.knowledge_base.benchmarks import BenchmarkRepository, BenchmarkData
from src.knowledge_base.slo_templates import SLOTemplateRepository
from src.context_intent.schema import SLOTargets


# Test database configuration
TEST_DB_URL = "postgresql://postgres:neuralnav@localhost:5432/neuralnav"


class TestBenchmarkRepository:
    """Tests for BenchmarkRepository with PostgreSQL backend."""

    @pytest.fixture
    def repo(self):
        """Create a BenchmarkRepository instance."""
        return BenchmarkRepository(database_url=TEST_DB_URL)

    def test_connection(self, repo):
        """Test that we can connect to PostgreSQL."""
        # Repository initializes successfully means connection works
        # The _test_connection() method is called in __init__
        assert repo is not None
        assert repo.database_url is not None

    def test_get_benchmark_exact_match(self, repo):
        """Test retrieving a benchmark with exact traffic profile match."""
        # Query for a known configuration
        # Using meta-llama/Llama-3.1-8B-Instruct, H100, 1 GPU, traffic (512, 256)
        benchmark = repo.get_benchmark(
            model_hf_repo="meta-llama/Llama-3.1-8B-Instruct",
            hardware="H100",
            hardware_count=1,
            prompt_tokens=512,
            output_tokens=256
        )

        assert benchmark is not None
        assert benchmark.model_hf_repo == "meta-llama/Llama-3.1-8B-Instruct"
        assert benchmark.hardware == "H100"
        assert benchmark.hardware_count == 1
        assert benchmark.prompt_tokens == 512
        assert benchmark.output_tokens == 256

    def test_get_benchmark_no_match(self, repo):
        """Test that non-existent configuration returns None."""
        benchmark = repo.get_benchmark(
            model_hf_repo="nonexistent/model",
            hardware="H100",
            hardware_count=1,
            prompt_tokens=512,
            output_tokens=256
        )

        assert benchmark is None

    def test_benchmark_has_p95_metrics(self, repo):
        """Test that benchmarks have p95 metrics (not p90)."""
        benchmark = repo.get_benchmark(
            model_hf_repo="meta-llama/Llama-3.1-8B-Instruct",
            hardware="H100",
            hardware_count=1,
            prompt_tokens=512,
            output_tokens=256
        )

        assert benchmark is not None
        # Check for p95 fields
        assert hasattr(benchmark, 'ttft_p95')
        assert hasattr(benchmark, 'itl_p95')  # ITL, not TPOT
        assert hasattr(benchmark, 'e2e_p95')

        # Values should be positive numbers
        assert benchmark.ttft_p95 > 0
        assert benchmark.itl_p95 > 0
        assert benchmark.e2e_p95 > 0

    def test_get_traffic_profiles(self, repo):
        """Test retrieving unique traffic profiles from database."""
        profiles = repo.get_traffic_profiles()

        assert len(profiles) > 0
        assert isinstance(profiles, list)

        # Should include the 4 GuideLLM profiles
        expected_profiles = [
            (512, 256),
            (1024, 1024),
            (4096, 512),
            (10240, 1536)
        ]

        for prompt, output in expected_profiles:
            assert (prompt, output) in profiles, f"Missing profile ({prompt}, {output})"

    def test_find_configurations_meeting_slo(self, repo):
        """Test finding configurations that meet SLO targets."""
        configs = repo.find_configurations_meeting_slo(
            prompt_tokens=512,
            output_tokens=256,
            ttft_p95_max_ms=200,
            itl_p95_max_ms=50,
            e2e_p95_max_ms=10000,
            min_qps=0
        )

        assert len(configs) > 0

        # All configs should meet SLO
        for config in configs:
            assert config.ttft_p95 <= 200
            assert config.itl_p95 <= 50
            assert config.e2e_p95 <= 10000

    def test_find_configurations_strict_slo(self, repo):
        """Test that strict SLO filters out slow configurations."""
        configs = repo.find_configurations_meeting_slo(
            prompt_tokens=512,
            output_tokens=256,
            ttft_p95_max_ms=10,  # Very tight
            itl_p95_max_ms=5,
            e2e_p95_max_ms=100,
            min_qps=0
        )

        # Should have no configurations meeting such strict SLO
        assert len(configs) == 0

    def test_get_available_models(self, repo):
        """Test retrieving list of available models."""
        models = repo.get_available_models()

        assert len(models) > 0
        assert isinstance(models, list)

        # Should include Llama-3.1-8B-Instruct
        assert "meta-llama/Llama-3.1-8B-Instruct" in models

    def test_benchmark_data_fields(self, repo):
        """Test that BenchmarkData has all required fields."""
        benchmark = repo.get_benchmark(
            model_hf_repo="meta-llama/Llama-3.1-8B-Instruct",
            hardware="H100",
            hardware_count=1,
            prompt_tokens=512,
            output_tokens=256
        )

        assert benchmark is not None

        # Check all required fields
        required_fields = [
            'model_hf_repo', 'hardware', 'hardware_count',
            'prompt_tokens', 'output_tokens',
            'ttft_p95', 'itl_p95', 'e2e_p95',
            'requests_per_second'  # Note: it's requests_per_second, not throughput_qps
        ]

        for field in required_fields:
            assert hasattr(benchmark, field), f"Missing field: {field}"
            assert getattr(benchmark, field) is not None, f"Field {field} is None"


class TestSLOTemplates:
    """Tests for SLO templates with p95/ITL migration."""

    @pytest.fixture
    def repo(self):
        """Create SLOTemplateRepository instance."""
        return SLOTemplateRepository()

    def test_load_templates(self, repo):
        """Test loading SLO templates from JSON."""
        templates = repo.get_all_templates()

        assert len(templates) > 0
        assert isinstance(templates, dict)

    def test_template_has_traffic_profile(self, repo):
        """Test that templates include traffic profile."""
        template = repo.get_template("chatbot_conversational")

        assert template is not None
        assert hasattr(template, 'prompt_tokens')
        assert hasattr(template, 'output_tokens')
        assert template.prompt_tokens > 0
        assert template.output_tokens > 0

    def test_template_has_experience_class(self, repo):
        """Test that templates include experience class."""
        template = repo.get_template("chatbot_conversational")

        assert template is not None
        assert hasattr(template, 'experience_class')

        # Experience class should be valid
        valid_classes = ['instant', 'conversational', 'interactive', 'deferred', 'batch']
        assert template.experience_class in valid_classes

    def test_template_has_p95_slo_targets(self, repo):
        """Test that SLO templates use p95 targets."""
        template = repo.get_template("chatbot_conversational")

        assert template is not None

        # Check for p95 fields (attributes on the object)
        assert hasattr(template, 'ttft_p95_target_ms')
        assert hasattr(template, 'itl_p95_target_ms')  # ITL, not tpot
        assert hasattr(template, 'e2e_p95_target_ms')

        # Values should be positive
        assert template.ttft_p95_target_ms > 0
        assert template.itl_p95_target_ms > 0
        assert template.e2e_p95_target_ms > 0

    def test_all_9_use_cases_present(self, repo):
        """Test that all 9 use cases from traffic_and_slos.md are present."""
        expected_use_cases = [
            'chatbot_conversational',
            'code_completion',
            'code_generation_detailed',
            'translation',
            'content_generation',  # Note: It's "content_generation" not "content_creation"
            'summarization_short',
            'document_analysis_rag',
            'long_document_summarization',
            'research_legal_analysis'
        ]

        templates = repo.get_all_templates()

        for use_case in expected_use_cases:
            assert use_case in templates, f"Missing use case: {use_case}"

    def test_traffic_profiles_match_guidelm(self, repo):
        """Test that traffic profiles match the 4 GuideLLM configurations."""
        expected_profiles = {
            (512, 256),
            (1024, 1024),
            (4096, 512),
            (10240, 1536)
        }

        templates = repo.get_all_templates()
        actual_profiles = set()

        for template in templates.values():
            actual_profiles.add((template.prompt_tokens, template.output_tokens))

        # All templates should use one of the 4 GuideLLM profiles
        for profile in actual_profiles:
            assert profile in expected_profiles, f"Unexpected profile: {profile}"


class TestTrafficProfileMatching:
    """Tests for traffic profile exact matching logic."""

    @pytest.fixture
    def repo(self):
        """Create a BenchmarkRepository instance."""
        return BenchmarkRepository(database_url=TEST_DB_URL)

    def test_exact_match_512_256(self, repo):
        """Test exact match for (512, 256) traffic profile."""
        benchmark = repo.get_benchmark(
            model_hf_repo="meta-llama/Llama-3.1-8B-Instruct",
            hardware="H100",
            hardware_count=1,
            prompt_tokens=512,
            output_tokens=256
        )

        assert benchmark is not None
        assert benchmark.prompt_tokens == 512
        assert benchmark.output_tokens == 256

    def test_exact_match_1024_1024(self, repo):
        """Test exact match for (1024, 1024) traffic profile."""
        benchmark = repo.get_benchmark(
            model_hf_repo="meta-llama/Llama-3.1-8B-Instruct",
            hardware="H100",
            hardware_count=1,
            prompt_tokens=1024,
            output_tokens=1024
        )

        # May or may not exist depending on data
        if benchmark is not None:
            assert benchmark.prompt_tokens == 1024
            assert benchmark.output_tokens == 1024

    def test_no_fuzzy_matching(self, repo):
        """Test that fuzzy matching is NOT used (exact match only)."""
        # Query for tokens that don't exactly match any profile
        benchmark = repo.get_benchmark(
            model_hf_repo="meta-llama/Llama-3.1-8B-Instruct",
            hardware="H100",
            hardware_count=1,
            prompt_tokens=500,  # Close to 512, but not exact
            output_tokens=250   # Close to 256, but not exact
        )

        # Should return None (no fuzzy matching)
        assert benchmark is None


class TestE2ELatencyCalculation:
    """Tests for E2E latency (pre-calculated vs dynamic)."""

    @pytest.fixture
    def repo(self):
        """Create a BenchmarkRepository instance."""
        return BenchmarkRepository(database_url=TEST_DB_URL)

    def test_e2e_precalculated_in_benchmarks(self, repo):
        """Test that E2E latency is pre-calculated in benchmark data."""
        benchmark = repo.get_benchmark(
            model_hf_repo="meta-llama/Llama-3.1-8B-Instruct",
            hardware="H100",
            hardware_count=1,
            prompt_tokens=512,
            output_tokens=256
        )

        assert benchmark is not None
        assert benchmark.e2e_p95 is not None
        assert benchmark.e2e_p95 > 0

        # E2E should be greater than TTFT (includes decode time)
        assert benchmark.e2e_p95 > benchmark.ttft_p95

    def test_e2e_vs_ttft_itl_relationship(self, repo):
        """Test that E2E is consistent with TTFT + (tokens × ITL)."""
        benchmark = repo.get_benchmark(
            model_hf_repo="meta-llama/Llama-3.1-8B-Instruct",
            hardware="H100",
            hardware_count=1,
            prompt_tokens=512,
            output_tokens=256
        )

        assert benchmark is not None

        # Rough check: E2E should be approximately TTFT + (output_tokens * ITL)
        # Allow for overhead/batching effects
        estimated_e2e = benchmark.ttft_p95 + (benchmark.output_tokens * benchmark.itl_p95)

        # E2E should be within reasonable range (allow 50% variance for batching effects)
        assert benchmark.e2e_p95 < estimated_e2e * 1.5
        assert benchmark.e2e_p95 > estimated_e2e * 0.5


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
