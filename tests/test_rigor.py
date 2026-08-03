from __future__ import annotations

import pytest

from agentkv.metrics.rigor import (
    ClockDriftExceeded,
    GpuClockLock,
    RigorConfig,
    interleave_schedule,
    median_iqr,
    warmup_filter,
)


def make_config(**overrides: object) -> RigorConfig:
    defaults: dict[str, object] = dict(
        warmup_trajectories=3,
        min_seeds=5,
        clock_drift_fail_pct=5.0,
        lock_gpu_clock_mhz=None,
    )
    defaults.update(overrides)
    return RigorConfig(**defaults)  # type: ignore[arg-type]


def test_rigor_config_from_yaml(tmp_path):
    path = tmp_path / "rigor.yaml"
    path.write_text(
        "warmup_trajectories: 3\n"
        "min_seeds: 5\n"
        "clock_drift_fail_pct: 5.0\n"
        "lock_gpu_clock_mhz: null\n"
    )
    config = RigorConfig.from_yaml(path)
    assert config.warmup_trajectories == 3
    assert config.min_seeds == 5
    assert config.clock_drift_fail_pct == 5.0
    assert config.lock_gpu_clock_mhz is None


def test_warmup_filter_drops_first_n():
    config = make_config(warmup_trajectories=3)
    trajectories = [f"traj-{i}" for i in range(10)]
    assert warmup_filter(trajectories, config) == [f"traj-{i}" for i in range(3, 10)]


def test_warmup_filter_shorter_than_n_returns_empty():
    config = make_config(warmup_trajectories=3)
    assert warmup_filter(["a", "b"], config) == []


def test_interleave_schedule_alternates_not_blocks():
    schedule = interleave_schedule(["naive", "append_only"], seeds=[0, 1, 2])
    assert schedule == [
        ("naive", 0),
        ("append_only", 0),
        ("naive", 1),
        ("append_only", 1),
        ("naive", 2),
        ("append_only", 2),
    ]


def test_interleave_schedule_requires_at_least_two_policies():
    with pytest.raises(ValueError):
        interleave_schedule(["naive"], seeds=[0, 1])


def test_median_iqr_requires_min_seeds():
    config = make_config(min_seeds=5)
    with pytest.raises(ValueError):
        median_iqr([1.0, 2.0, 3.0], config)


def test_median_iqr_computes_correctly():
    config = make_config(min_seeds=5)
    result = median_iqr([10.0, 20.0, 30.0, 40.0, 50.0], config)
    assert result.median == 30.0
    assert result.q1 == 20.0
    assert result.q3 == 40.0
    assert result.iqr == 20.0


def test_gpu_clock_lock_falls_back_to_drift_detection_without_target():
    config = make_config(lock_gpu_clock_mhz=None)
    lock = GpuClockLock(config)
    lock.acquire()
    assert not lock.is_hardware_locked
    lock.check_drift(lock._baseline_mhz)  # no drift -> no raise


def test_gpu_clock_lock_raises_on_excessive_drift():
    config = make_config(lock_gpu_clock_mhz=None, clock_drift_fail_pct=5.0)
    lock = GpuClockLock(config)
    lock.acquire()
    baseline = lock._baseline_mhz
    with pytest.raises(ClockDriftExceeded):
        lock.check_drift(int(baseline * 2))


def test_gpu_clock_lock_baseline_uses_median_of_samples():
    """A lone snapshot can catch a transient boost spike rather than the
    workload's steady clock — passing warmup samples should use their median
    instead of a fresh single read."""
    config = make_config(lock_gpu_clock_mhz=None)
    lock = GpuClockLock(config)
    lock.acquire([2340, 2360, 2535, 2350, 2345])  # one outlier spike among steady reads
    assert lock._baseline_mhz == 2350  # median, not the 2535 outlier
    lock.check_drift(2360)  # within 5% of the median -> no raise
