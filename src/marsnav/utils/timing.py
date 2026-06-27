"""Small timing helpers for per-decision instrumentation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns


@dataclass
class BatchTimer:
    """Accumulate wall-clock timing for batched work.

    ``batches`` counts calls (for example one vectorized policy query) while
    ``items`` counts the work units inside those calls (for example one action
    decision per environment).
    """

    total_ns: int = 0
    batches: int = 0
    items: int = 0

    def add(self, elapsed_ns: int, *, items: int = 1) -> None:
        if elapsed_ns < 0:
            raise ValueError("elapsed_ns must be non-negative")
        if items <= 0:
            raise ValueError("items must be positive")
        self.total_ns += int(elapsed_ns)
        self.batches += 1
        self.items += int(items)

    def reset(self) -> None:
        self.total_ns = 0
        self.batches = 0
        self.items = 0

    @property
    def total_s(self) -> float:
        return self.total_ns / 1e9

    @property
    def mean_batch_ms(self) -> float:
        if self.batches == 0:
            return 0.0
        return self.total_ns / self.batches / 1e6

    @property
    def mean_item_ms(self) -> float:
        if self.items == 0:
            return 0.0
        return self.total_ns / self.items / 1e6


def now_ns() -> int:
    """Return a monotonic high-resolution timestamp in nanoseconds."""

    return perf_counter_ns()


def elapsed_ns(start_ns: int) -> int:
    """Nanoseconds elapsed since ``start_ns`` from :func:`now_ns`."""

    return perf_counter_ns() - start_ns
