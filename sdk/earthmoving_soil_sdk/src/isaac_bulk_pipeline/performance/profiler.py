"""Detailed runtime, resource and host/device-transfer accounting."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import os
import resource
from time import perf_counter, process_time
from typing import Iterator

import numpy as np


@dataclass
class TransferCounters:
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    h2d_transfers: int = 0
    d2h_transfers: int = 0
    gpu_synchronizations: int = 0

    def record_h2d(self, byte_count: int) -> None:
        self.h2d_bytes += int(byte_count)
        self.h2d_transfers += 1

    def record_d2h(self, byte_count: int) -> None:
        self.d2h_bytes += int(byte_count)
        self.d2h_transfers += 1

    def record_synchronization(self, count: int = 1) -> None:
        self.gpu_synchronizations += int(count)


@dataclass(frozen=True)
class PerformanceSummary:
    call_count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    maximum_ms: float
    total_wall_time_ms: float
    percentage_of_total: float


class PerformanceProfiler:
    """Low-overhead timing collector used by both V2 frontends."""

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = {}
        self._wall_start = perf_counter()
        self._cpu_start = process_time()
        self.transfers = TransferCounters()

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.record_ms(name, (perf_counter() - start) * 1_000.0)

    def record_ms(self, name: str, elapsed_ms: float) -> None:
        value = float(elapsed_ms)
        if not name or not np.isfinite(value) or value < 0.0:
            raise ValueError("[PerformanceProfiler] invalid timing sample")
        self._samples.setdefault(name, []).append(value)

    def summaries(self) -> dict[str, PerformanceSummary]:
        totals = {name: float(np.sum(values)) for name, values in self._samples.items()}
        total = max(sum(totals.values()), 1.0e-12)
        result: dict[str, PerformanceSummary] = {}
        for name, samples in sorted(self._samples.items()):
            values = np.asarray(samples, dtype=np.float64)
            result[name] = PerformanceSummary(
                call_count=int(values.size),
                mean_ms=float(np.mean(values)),
                median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95.0)),
                p99_ms=float(np.percentile(values, 99.0)),
                maximum_ms=float(np.max(values)),
                total_wall_time_ms=totals[name],
                percentage_of_total=100.0 * totals[name] / total,
            )
        return result

    def report(self) -> dict[str, object]:
        wall = max(perf_counter() - self._wall_start, 1.0e-12)
        cpu = max(process_time() - self._cpu_start, 0.0)
        summaries = {name: asdict(value) for name, value in self.summaries().items()}
        return {
            "modules": summaries,
            "process": {
                "wall_time_s": wall,
                "cpu_time_s": cpu,
                "cpu_utilization_process_percent_of_one_core": 100.0 * cpu / wall,
                "logical_cpu_count": os.cpu_count(),
                "peak_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            },
            "transfers": asdict(self.transfers),
        }

    def reset(self) -> None:
        self._samples.clear()
        self._wall_start = perf_counter()
        self._cpu_start = process_time()
        self.transfers = TransferCounters()
