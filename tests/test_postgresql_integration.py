"""Tests for PostgreSQL benchmark repository and SLO templates.

Database tests use a dedicated planner_test database with static fixture
data (see conftest.py), so they work regardless of what production data
is loaded.

Tests cover:
1. BenchmarkRepository - PostgreSQL connection and queries
2. Traffic profile exact matching
3. p95/ITL metric usage
4. SLO filtering and compliance checking
5. SLO templates - JSON-based (no database needed)
"""

import pytest

from planner.knowledge_base.benchmarks import BenchmarkData, BenchmarkRepository
from planner.knowledge_base.slo_templates import SLOTemplateRepository


@pytest.mark.database
class TestBenchmarkRepository:
    """Tests for BenchmarkRepository with PostgreSQL backend."""

    @pytest.fixture
    def repo(self, test_db_url):
        """Create a BenchmarkRepository connected to the test database."""
        return BenchmarkRepository(database_url=test_db_url)

    def test_connection(self, repo):
        """Test that we can connect to PostgreSQL."""
        assert repo is not None
        assert repo.database_url is not None

    def test_get_benchmark_exact_match(self, repo):
        """Test retrieving a benchmark with exact traffic profile match."""
        benchmark = repo.get_benchmark(
            model_hf_repo="meta-llama/llama-3.1-8b-instruct",
            hardware="H100",
            hardware_count=1,
            prompt_tokens=512,
            output_tokens=256,
        )

        assert benchmark is not None
        assert benchmark.model_hf_repo == "meta-llama/llama-3.1-8b-instruct"
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
            output_tokens=256,
        )

        assert benchmark is None

    def test_benchmark_has_p95_metrics(self, repo):
        """Test that benchmarks have p95 metrics (not p90)."""
        benchmark = repo.get_benchmark(
            model_hf_repo="meta-llama/llama-3.1-8b-instruct",
            hardware="H100",
            hardware_count=1,
            prompt_tokens=512,
            output_tokens=256,
        )

        assert benchmark is not None
        assert hasattr(benchmark, "ttft_p95")
        assert hasattr(benchmark, "itl_p95")
        assert hasattr(benchmark, "e2e_p95")

        assert benchmark.ttft_p95 > 0
        assert benchmark.itl_p95 > 0
        assert benchmark.e2e_p95 > 0

    def test_get_traffic_profiles(self, repo):
        """Test retrieving unique traffic profiles from database."""
        profiles = repo.get_traffic_profiles()

        assert len(profiles) > 0
        assert isinstance(profiles, list)

        # Test fixture includes the 4 GuideLLM profiles
        expected_profiles = [(512, 256), (1024, 1024), (4096, 512), (10240, 1536)]

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
            min_qps=0,
        )

        assert len(configs) > 0

        for config in configs:
            assert config.ttft_p95 <= 200
            assert config.itl_p95 <= 50
            assert config.e2e_p95 <= 10000

    def test_find_configurations_strict_slo(self, repo):
        """Test that strict SLO filters out slow configurations."""
        configs = repo.find_configurations_meeting_slo(
            prompt_tokens=512,
            output_tokens=256,
            ttft_p95_max_ms=10,
            itl_p95_max_ms=5,
            e2e_p95_max_ms=100,
            min_qps=0,
        )

        assert len(configs) == 0

    def test_get_available_models(self, repo):
        """Test retrieving list of available models."""
        models = repo.get_available_models()

        assert len(models) > 0
        assert isinstance(models, list)

        assert "meta-llama/llama-3.1-8b-instruct" in models

    def test_benchmark_data_fields(self, repo):
        """Test that BenchmarkData has all required fields."""
        benchmark = repo.get_benchmark(
            model_hf_repo="meta-llama/llama-3.1-8b-instruct",
            hardware="H100",
            hardware_count=1,
            prompt_tokens=512,
            output_tokens=256,
        )

        assert benchmark is not None

        required_fields = [
            "model_hf_repo",
            "hardware",
            "hardware_count",
            "prompt_tokens",
            "output_tokens",
            "ttft_p95",
            "itl_p95",
            "e2e_p95",
            "requests_per_second",
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
        assert hasattr(template, "prompt_tokens")
        assert hasattr(template, "output_tokens")
        assert template.prompt_tokens > 0
        assert template.output_tokens > 0

    def test_template_has_experience_class(self, repo):
        """Test that templates include experience class."""
        template = repo.get_template("chatbot_conversational")

        assert template is not None
        assert hasattr(template, "experience_class")

        valid_classes = ["instant", "conversational", "interactive", "deferred", "batch"]
        assert template.experience_class in valid_classes

    def test_template_has_p95_slo_targets(self, repo):
        """Test that SLO templates use p95 targets."""
        template = repo.get_template("chatbot_conversational")

        assert template is not None

        assert hasattr(template, "ttft_p95_target_ms")
        assert hasattr(template, "itl_p95_target_ms")
        assert hasattr(template, "e2e_p95_target_ms")

        assert template.ttft_p95_target_ms > 0
        assert template.itl_p95_target_ms > 0
        assert template.e2e_p95_target_ms > 0

    def test_all_9_use_cases_present(self, repo):
        """Test that all 9 use cases from traffic_and_slos.md are present."""
        expected_use_cases = [
            "chatbot_conversational",
            "code_completion",
            "code_generation_detailed",
            "translation",
            "content_generation",
            "summarization_short",
            "document_analysis_rag",
            "long_document_summarization",
            "research_legal_analysis",
        ]

        templates = repo.get_all_templates()

        for use_case in expected_use_cases:
            assert use_case in templates, f"Missing use case: {use_case}"

    def test_traffic_profiles_match_guidelm(self, repo):
        """Test that traffic profiles match the 4 GuideLLM configurations."""
        expected_profiles = {(512, 256), (1024, 1024), (4096, 512), (10240, 1536)}

        templates = repo.get_all_templates()
        actual_profiles = set()

        for template in templates.values():
            actual_profiles.add((template.prompt_tokens, template.output_tokens))

        for profile in actual_profiles:
            assert profile in expected_profiles, f"Unexpected profile: {profile}"


@pytest.mark.database
class TestTrafficProfileMatching:
    """Tests for traffic profile exact matching logic."""

    @pytest.fixture
    def repo(self, test_db_url):
        """Create a BenchmarkRepository connected to the test database."""
        return BenchmarkRepository(database_url=test_db_url)

    def test_exact_match_512_256(self, repo):
        """Test exact match for (512, 256) traffic profile."""
        benchmark = repo.get_benchmark(
            model_hf_repo="meta-llama/llama-3.1-8b-instruct",
            hardware="H100",
            hardware_count=1,
            prompt_tokens=512,
            output_tokens=256,
        )

        assert benchmark is not None
        assert benchmark.prompt_tokens == 512
        assert benchmark.output_tokens == 256

    def test_exact_match_1024_1024(self, repo):
        """Test exact match for (1024, 1024) traffic profile."""
        benchmark = repo.get_benchmark(
            model_hf_repo="meta-llama/llama-3.1-8b-instruct",
            hardware="H100",
            hardware_count=1,
            prompt_tokens=1024,
            output_tokens=1024,
        )

        assert benchmark is not None
        assert benchmark.prompt_tokens == 1024
        assert benchmark.output_tokens == 1024

    def test_no_fuzzy_matching(self, repo):
        """Test that fuzzy matching is NOT used (exact match only)."""
        benchmark = repo.get_benchmark(
            model_hf_repo="meta-llama/llama-3.1-8b-instruct",
            hardware="H100",
            hardware_count=1,
            prompt_tokens=500,
            output_tokens=250,
        )

        assert benchmark is None


@pytest.mark.database
class TestE2ELatencyCalculation:
    """Tests for E2E latency (pre-calculated vs dynamic)."""

    @pytest.fixture
    def repo(self, test_db_url):
        """Create a BenchmarkRepository connected to the test database."""
        return BenchmarkRepository(database_url=test_db_url)

    def test_e2e_precalculated_in_benchmarks(self, repo):
        """Test that E2E latency is pre-calculated in benchmark data."""
        benchmark = repo.get_benchmark(
            model_hf_repo="meta-llama/llama-3.1-8b-instruct",
            hardware="H100",
            hardware_count=1,
            prompt_tokens=512,
            output_tokens=256,
        )

        assert benchmark is not None
        assert benchmark.e2e_p95 is not None
        assert benchmark.e2e_p95 > 0

        # E2E should be greater than TTFT (includes decode time)
        assert benchmark.e2e_p95 > benchmark.ttft_p95

    def test_e2e_vs_ttft_itl_relationship(self, repo):
        """Test that E2E is consistent with TTFT + (tokens x ITL)."""
        benchmark = repo.get_benchmark(
            model_hf_repo="meta-llama/llama-3.1-8b-instruct",
            hardware="H100",
            hardware_count=1,
            prompt_tokens=512,
            output_tokens=256,
        )

        assert benchmark is not None

        # Rough check: E2E should be approximately TTFT + (output_tokens * ITL)
        estimated_e2e = benchmark.ttft_p95 + (benchmark.output_tokens * benchmark.itl_p95)

        # E2E should be within reasonable range (allow 50% variance for batching effects)
        assert benchmark.e2e_p95 < estimated_e2e * 1.5
        assert benchmark.e2e_p95 > estimated_e2e * 0.5
