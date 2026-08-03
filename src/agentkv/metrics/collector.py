"""Per-step metrics collection -> parquet (AGENTKV_SPEC.md §0.3).

Figures are generated from this committed parquet data and never hand-edited
(spec §7) — this module is the single writer of that data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generic, TypeVar

import pandas as pd


@dataclass(frozen=True)
class StepRecord:
    """One row per agent step. Fields per spec §0.3, plus temp/clock/power
    required by §0.2's rigor logging."""

    step_idx: int
    prompt_tokens: int
    cached_tokens: int
    prefill_tokens: int
    ttft_ms: float
    decode_ms: float
    output_tokens: int
    gpu_mem_bytes: int
    policy: str
    seed: int
    temp_c: int
    sm_clock_mhz: int
    power_w: float


T = TypeVar("T")


class RecordCollector(Generic[T]):
    """Buffers dataclass records in memory and flushes them to an append-only
    parquet file. Shared by `MetricsCollector` (per-step, spec §0.3) and
    `CompactionEventCollector` (per-compaction-event, spec §1.3) — same
    append-only, never-hand-edited contract (spec §7), different row shape."""

    def __init__(self) -> None:
        self._records: list[T] = []

    def __len__(self) -> int:
        return len(self._records)

    def record(self, record: T) -> None:
        self._records.append(record)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self._records])  # type: ignore[call-overload]

    def flush(self, path: Path) -> None:
        """Appends buffered records to `path` (creating it if absent) and clears the buffer."""
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = self.to_frame()
        if path.exists():
            frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
        frame.to_parquet(path, index=False)
        self._records.clear()


class MetricsCollector(RecordCollector[StepRecord]):
    """Buffers StepRecords in memory and flushes them to an append-only parquet file."""
