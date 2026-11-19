"""Tests for benchmark repository and use case definitions.

Uses a temporary SQLite database with static fixture data (see conftest.py).

Tests cover:
1. BenchmarkRepository - database connection and queries
2. Traffic profile exact matching
3. p95/ITL metric usage
4. SLO filtering and compliance checking
5. Use case definitions - JSON-based (no database needed)
"""

import pytest

from planner.knowledge_base.benchmarks import BenchmarkData, BenchmarkRepository
from planner.knowledge_base.loader import get_db_stats
from planner.knowledge_base.use_cases import UseCaseRepository


@pytest.mark.unit
class TestBenchmarkRepository:
    """Tests for BenchmarkRepository with database backend."""

    @pytest.fixture
    def repo(self, test_db_path):
        """Create a BenchmarkRepository connected to the test database."""
        return BenchmarkRepository(db_path=test_db_path)

    def test_connection(self, repo):
        """Test that we can connect to the database."""
        assert repo is not None
        assert repo._conn is not None

    def test_db_stats_include_benchmark_sources(self, repo):
        """Test that get_db_stats returns benchmark_sources from real data."""
        conn = repo._get_connection()
        try:
            stats = get_db_stats(conn)
        finally:
            conn.close()

        sources = stats["benchmark_sources"]
        assert len(sources) > 0
        for entry in sources:
            assert entry["source"]
            assert entry["confidence_level"]
            assert entry["count"] > 0
        assert sum(e["count"] for e in sources) == stats["total_benchmarks"]

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


@pytest.mark.unit
class TestUseCaseRepository:
    """Tests for use case definitions."""

    @pytest.fixture
    def repo(self):
        """Create UseCaseRepository instance."""
        return UseCaseRepository()

    def test_load_use_cases(self, repo):
        """Test loading use case definitions from JSON."""
        use_cases = repo.get_all_use_cases()

        assert len(use_cases) > 0
        assert isinstance(use_cases, dict)

    def test_use_case_has_traffic_profile(self, repo):
        """Test that use cases include traffic profile."""
        uc = repo.get_use_case("chatbot_conversational")

        assert uc is not None
        assert hasattr(uc, "prompt_tokens")
        assert hasattr(uc, "output_tokens")
        assert uc.prompt_tokens > 0
        assert uc.output_tokens > 0

    def test_use_case_has_slo_ranges(self, repo):
        """Test that use cases have SLO min/max ranges."""
        uc = repo.get_use_case("chatbot_conversational")

        assert uc is not None
        assert uc.ttft_range.min > 0
        assert uc.ttft_range.max > uc.ttft_range.min
        assert uc.itl_range.min > 0
        assert uc.itl_range.max > uc.itl_range.min
        assert uc.e2e_range.min > 0
        assert uc.e2e_range.max > uc.e2e_range.min

    def test_use_case_has_display_name(self, repo):
        """Test that use cases have display names."""
        uc = repo.get_use_case("chatbot_conversational")
        assert uc is not None
        assert uc.display_name == "Chatbot / Conversational AI"

    def test_all_9_use_cases_present(self, repo):
        """Test that all 9 use cases are present."""
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

        use_cases = repo.get_all_use_cases()

        for use_case in expected_use_cases:
            assert use_case in use_cases, f"Missing use case: {use_case}"

    def test_traffic_profiles_match_guidelm(self, repo):
        """Test that traffic profiles match the 4 GuideLLM configurations."""
        expected_profiles = {(512, 256), (1024, 1024), (4096, 512), (10240, 1536)}

        use_cases = repo.get_all_use_cases()
        actual_profiles = set()

        for uc in use_cases.values():
            actual_profiles.add((uc.prompt_tokens, uc.output_tokens))

        for profile in actual_profiles:
            assert profile in expected_profiles, f"Unexpected profile: {profile}"


@pytest.mark.unit
class TestTrafficProfileMatching:
    """Tests for traffic profile exact matching logic."""

    @pytest.fixture
    def repo(self, test_db_path):
        """Create a BenchmarkRepository connected to the test database."""
        return BenchmarkRepository(db_path=test_db_path)

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


@pytest.mark.unit
class TestE2ELatencyCalculation:
    """Tests for E2E latency (pre-calculated vs dynamic)."""

    @pytest.fixture
    def repo(self, test_db_path):
        """Create a BenchmarkRepository connected to the test database."""
        return BenchmarkRepository(db_path=test_db_path)

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
