"""Benchmark rigor: GPU clock control, warmup, interleaving, robust statistics.

Implements AGENTKV_SPEC.md §0.2. On a thermally-throttled laptop, sloppy
measurement produces fake results — every knob here exists to defend against
a specific way that happens.
"""
from __future__ import annotations

import shutil
import statistics
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RigorConfig:
    warmup_trajectories: int
    min_seeds: int
    clock_drift_fail_pct: float
    lock_gpu_clock_mhz: int | None

    @classmethod
    def from_yaml(cls, path: Path) -> RigorConfig:
        import yaml

        data = yaml.safe_load(path.read_text())
        return cls(
            warmup_trajectories=data["warmup_trajectories"],
            min_seeds=data["min_seeds"],
            clock_drift_fail_pct=data["clock_drift_fail_pct"],
            lock_gpu_clock_mhz=data.get("lock_gpu_clock_mhz"),
        )


class ClockDriftExceeded(RuntimeError):
    """Raised when SM clock drift exceeds the configured threshold with no hardware lock."""


class GpuClockLock:
    """Best-effort `nvidia-smi -lgc` clock lock, with drift-detection fallback.

    Under WSL2, `-lgc` is typically unavailable (spec §9) — acquire() falls back
    to recording a baseline clock and check_drift() fails loudly instead of
    silently trusting an unlocked, possibly-throttling clock.
    """

    def __init__(self, config: RigorConfig) -> None:
        self._config = config
        self._locked = False
        self._baseline_mhz: int | None = None

    def acquire(self, baseline_samples: Sequence[int] | None = None) -> None:
        """`baseline_samples`: optional SM clock readings taken during a
        warmup period, used as the baseline instead of one fresh read.

        A single instantaneous read can land on a transient boost spike
        rather than the workload's actual steady-state clock — measured on
        the 4060 laptop: a lone post-warmup read caught ~2535MHz while the
        subsequent steady operating band was ~2340-2360MHz, producing a false
        ~7-8% "drift" that was really baseline noise, not throttling. Passing
        several samples collected across warmup and taking their median
        avoids that.
        """
        nvidia_smi = shutil.which("nvidia-smi")
        if self._config.lock_gpu_clock_mhz is None or nvidia_smi is None:
            self._baseline_mhz = self._resolve_baseline(baseline_samples)
            return
        target = self._config.lock_gpu_clock_mhz
        result = subprocess.run(
            [nvidia_smi, "-lgc", f"{target},{target}"],
            capture_output=True,
            text=True,
        )
        self._locked = result.returncode == 0
        if not self._locked:
            self._baseline_mhz = self._resolve_baseline(baseline_samples)

    def _resolve_baseline(self, baseline_samples: Sequence[int] | None) -> int:
        if baseline_samples:
            return int(statistics.median(baseline_samples))
        return self._read_sm_clock_mhz()

    def release(self) -> None:
        if self._locked:
            nvidia_smi = shutil.which("nvidia-smi")
            if nvidia_smi is not None:
                subprocess.run([nvidia_smi, "-rgc"], capture_output=True, text=True)
            self._locked = False

    @property
    def is_hardware_locked(self) -> bool:
        return self._locked

    def check_drift(self, current_mhz: int) -> None:
        if self._locked or self._baseline_mhz is None:
            return
        drift_pct = abs(current_mhz - self._baseline_mhz) / self._baseline_mhz * 100
        if drift_pct > self._config.clock_drift_fail_pct:
            raise ClockDriftExceeded(
                f"SM clock drifted {drift_pct:.1f}% from baseline {self._baseline_mhz} MHz "
                f"(now {current_mhz} MHz); threshold is {self._config.clock_drift_fail_pct}%. "
                "This run's timings are not trustworthy — discard and re-run."
            )

    @staticmethod
    def _read_sm_clock_mhz() -> int:
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            raise RuntimeError("nvidia-smi not found; cannot establish a clock baseline.")
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip().splitlines()[0])


def warmup_filter(trajectory_ids: Sequence[str], config: RigorConfig) -> list[str]:
    """Drops the first `warmup_trajectories` entries (spec §0.2 warmup requirement)."""
    return list(trajectory_ids[config.warmup_trajectories :])


def interleave_schedule(policies: Sequence[str], seeds: Sequence[int]) -> list[tuple[str, int]]:
    """Alternates policies per seed instead of blocking by policy.

    Spec §0.2: "never run all of policy A then all of policy B... This is a
    hard requirement." Order is (seed[0], policies...), (seed[1], policies...), ...
    so thermal drift across the run affects every policy equally.
    """
    if len(policies) < 2:
        raise ValueError("Interleaving requires at least two policies to alternate between.")
    return [(policy, seed) for seed in seeds for policy in policies]


@dataclass(frozen=True)
class MedianIQR:
    median: float
    q1: float
    q3: float

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1


def median_iqr(values: Sequence[float], config: RigorConfig) -> MedianIQR:
    """Median + IQR across seeds. Spec §0.2/§7: "report median + IQR ... never a bare mean."

    Refuses to compute a statistic that would misrepresent variance below the
    configured seed count, rather than silently reporting on too few seeds.
    """
    if len(values) < config.min_seeds:
        raise ValueError(
            f"Only {len(values)} seed(s) provided; spec requires >= {config.min_seeds}."
        )
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return MedianIQR(median=statistics.median(values), q1=q1, q3=q3)


@dataclass(frozen=True)
class GpuSample:
    temp_c: int
    sm_clock_mhz: int
    power_w: float


class GpuSampler:
    """Per-step GPU temp/clock/power via NVML (spec §0.2: "Record: GPU temp,
    clock, power, per-step, into the metrics parquet.").

    pynvml is imported lazily so this module stays importable on machines
    without an NVIDIA driver (e.g. running unit tests for the other functions
    in this file in isolation).
    """

    def __init__(self, device_index: int = 0) -> None:
        import pynvml

        self._pynvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    def sample(self) -> GpuSample:
        p = self._pynvml
        return GpuSample(
            temp_c=p.nvmlDeviceGetTemperature(self._handle, p.NVML_TEMPERATURE_GPU),
            sm_clock_mhz=p.nvmlDeviceGetClockInfo(self._handle, p.NVML_CLOCK_SM),
            power_w=p.nvmlDeviceGetPowerUsage(self._handle) / 1000.0,
        )

    def shutdown(self) -> None:
        self._pynvml.nvmlShutdown()
