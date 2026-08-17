"""Small dependency-free runtime profiler for multi-rate pipeline modules."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class RuntimeStatistics:
    """Runtime distribution for one named module, in milliseconds."""

    sample_count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float


class RuntimeProfiler:
    """Collect mean/p50/p95/max timings without importing Isaac or pandas."""

    def __init__(self) -> None:
        self._samples_ms: dict[str, list[float]] = {}

    def record_seconds(self, module_name: str, elapsed_s: float) -> None:
        if not isinstance(module_name, str) or not module_name.strip():
            raise ValueError("[RuntimeProfiler] module_name must be non-empty")
        elapsed = float(elapsed_s)
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError(
                "[RuntimeProfiler] elapsed_s must be finite and non-negative; "
                f"value={elapsed_s!r}"
            )
        self._samples_ms.setdefault(module_name, []).append(elapsed * 1_000.0)

    @contextmanager
    def measure(self, module_name: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.record_seconds(module_name, perf_counter() - start)

    def summary(self) -> dict[str, RuntimeStatistics]:
        result: dict[str, RuntimeStatistics] = {}
        for name, samples in sorted(self._samples_ms.items()):
            values = np.asarray(samples, dtype=np.float64)
            result[name] = RuntimeStatistics(
                sample_count=int(values.size),
                mean_ms=float(np.mean(values)),
                p50_ms=float(np.percentile(values, 50.0)),
                p95_ms=float(np.percentile(values, 95.0)),
                max_ms=float(np.max(values)),
            )
        return result

    def reset(self) -> None:
        self._samples_ms.clear()
