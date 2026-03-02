"""Verify BenchmarkSource protocol is satisfied by existing and new implementations."""

import pytest

from neuralnav.knowledge_base.benchmark_source import BenchmarkSource


class FakeBenchmarkSource:
    """Minimal implementation to verify protocol shape."""

    def find_configurations_meeting_slo(
        self,
        prompt_tokens: int,
        output_tokens: int,
        ttft_p95_max_ms: int,
        itl_p95_max_ms: int,
        e2e_p95_max_ms: int,
        min_qps: float = 0,
        percentile: str = "p95",
        gpu_types: list[str] | None = None,
    ) -> list:
        return []


@pytest.mark.unit
def test_fake_satisfies_protocol():
    source: BenchmarkSource = FakeBenchmarkSource()
    result = source.find_configurations_meeting_slo(
        prompt_tokens=512,
        output_tokens=256,
        ttft_p95_max_ms=200,
        itl_p95_max_ms=30,
        e2e_p95_max_ms=8000,
    )
    assert result == []


@pytest.mark.unit
def test_fake_is_runtime_checkable_instance():
    source = FakeBenchmarkSource()
    assert isinstance(source, BenchmarkSource)


@pytest.mark.unit
def test_non_conforming_class_fails_isinstance():
    class NotASource:
        pass

    assert not isinstance(NotASource(), BenchmarkSource)
