"""Protocol definition for benchmark data sources."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from neuralnav.knowledge_base.benchmarks import BenchmarkData


@runtime_checkable
class BenchmarkSource(Protocol):
    """Protocol for benchmark data providers.

    Implementations:
    - BenchmarkRepository: PostgreSQL (standalone/upstream)
    - ModelCatalogBenchmarkSource: RHOAI Model Catalog API
    """

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
    ) -> list[BenchmarkData]: ...
