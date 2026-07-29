from __future__ import annotations

from pathlib import Path

from agentkv.metrics.collector import MetricsCollector, StepRecord


def make_record(step_idx: int, policy: str = "naive", seed: int = 0) -> StepRecord:
    return StepRecord(
        step_idx=step_idx,
        prompt_tokens=100 + step_idx,
        cached_tokens=50,
        prefill_tokens=50 + step_idx,
        ttft_ms=12.3,
        decode_ms=45.6,
        output_tokens=20,
        gpu_mem_bytes=1_000_000,
        policy=policy,
        seed=seed,
        temp_c=65,
        sm_clock_mhz=2100,
        power_w=45.0,
    )


def test_collector_buffers_records():
    collector = MetricsCollector()
    assert len(collector) == 0
    collector.record(make_record(0))
    collector.record(make_record(1))
    assert len(collector) == 2


def test_collector_to_frame_has_expected_columns():
    collector = MetricsCollector()
    collector.record(make_record(0))
    frame = collector.to_frame()
    assert list(frame.columns) == list(StepRecord.__dataclass_fields__.keys())
    assert frame.iloc[0]["step_idx"] == 0


def test_collector_flush_creates_and_appends(tmp_path: Path):
    path = tmp_path / "metrics.parquet"
    collector = MetricsCollector()
    collector.record(make_record(0))
    collector.flush(path)
    assert path.exists()
    assert len(collector) == 0

    collector.record(make_record(1))
    collector.flush(path)

    import pandas as pd

    frame = pd.read_parquet(path)
    assert len(frame) == 2
    assert sorted(frame["step_idx"].tolist()) == [0, 1]
