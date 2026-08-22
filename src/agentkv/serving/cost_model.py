"""Analytical prefill cost model (AGENTKV_SPEC.md Phase 6, §6.1):

    "Your measurements are on a 1.7B model. Show the effect scales. Prefill
    FLOPs ≈ 2 × N_params × n_tokens plus attention terms; derive
    tokens-saved → FLOPs-saved → GPU-seconds-saved → dollars, parameterized
    by model size and hardware. Validate the model against your own
    measurements first, then extrapolate to 70B on an H100 and state the
    assumptions and the error bars. Do not present extrapolated numbers as
    measurements."

No GPU/vLLM needed anywhere in this module — every function here is pure
arithmetic over already-committed step data (`prompt_tokens`/
`prefill_tokens`/`ttft_ms`, `metrics.collector.StepRecord`'s own fields).

## The formula

Applied literally, per step:

    FLOPs(prefill_tokens, total_context_tokens) =
        2 * n_params * prefill_tokens                                       # linear term
      + 2 * n_layers * hidden_size * prefill_tokens * total_context_tokens  # attention term

The **linear term** (`2 * N * n_tokens`) is the standard forward-pass FLOPs
approximation for a dense transformer's matrix multiplies (Kaplan et al.
2020, "Scaling Laws for Neural Language Models," eq. 2.1), applied only to
`prefill_tokens` — the tokens actually *computed* this step, not the whole
context, since vLLM's prefix cache means cached tokens cost no new compute.
This is exactly what `prefill_tokens` vs. `cached_tokens` already
distinguishes in every committed parquet in this project.

The **attention term** is where compaction's savings mechanistically come
from. Even though only `prefill_tokens` tokens are newly computed, each one
attends over the *entire* resident context (`total_context_tokens` — this
project's `prompt_tokens`), so that cost scales with the *product* of the
two, not `prefill_tokens` alone (the context-dependent term in Kaplan et
al.'s eq. 2.2, applied per-step to prefill rather than a full sequence).
Shortening `total_context_tokens` — what every `CompactionPolicy` in this
project does — reduces this term directly. This is the reason compaction
saves compute, stated as arithmetic rather than argued from intuition.

Both terms are approximations (they ignore embedding/softmax/norm FLOPs,
and the attention term treats every new token as attending the full context
rather than its own causal-masked prefix — a deliberate simplification
matching spec's own "plus attention terms," not a claim of exactness). Real
achieved throughput on real hardware is always some fraction of the
"peak FLOPS" a formula like this implies — accounted for by `fit_efficiency`
below, not assumed away.

## Validate first, then extrapolate

`fit_efficiency` calibrates an `achieved_flops / hardware_peak_flops`
fraction against this project's own real, already-committed
`ttft_ms`/`prefill_tokens`/`prompt_tokens` measurements — this is the
"validate against your own measurements first" step, and it is a real
regression against real data, not an assumed constant.

`extrapolate_savings` then recomputes FLOPs for two policies' entire real,
already-measured per-step token sequences under a *different* model's
architecture (e.g. Llama-3-70B) — "extrapolate to 70B" means "what would
these exact measured trajectories cost in FLOPs if a 70B model were doing
the same work," not scaling one aggregate summary number by a param-count
ratio. The result is explicitly returned alongside a sensitivity range (the
fitted efficiency's IQR, plus an independent industry-typical-MFU
cross-check), never as a single bare number — spec's "state the assumptions
and the error bars," and "do not present extrapolated numbers as
measurements."
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


@dataclass(frozen=True)
class ModelSpec:
    name: str
    n_params: int
    n_layers: int
    hidden_size: int

    @classmethod
    def from_yaml(cls, path: Path, key: str) -> ModelSpec:
        data = yaml.safe_load(path.read_text())["models"][key]
        return cls(
            name=data["name"],
            n_params=int(data["n_params"]),
            n_layers=int(data["n_layers"]),
            hidden_size=int(data["hidden_size"]),
        )


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    peak_flops_per_sec: float
    usd_per_gpu_hour: float | None

    @classmethod
    def from_yaml(cls, path: Path, key: str) -> HardwareSpec:
        data = yaml.safe_load(path.read_text())["hardware"][key]
        rate = data.get("usd_per_gpu_hour")
        return cls(
            name=data["name"],
            peak_flops_per_sec=float(data["peak_flops_per_sec"]),
            usd_per_gpu_hour=None if rate is None else float(rate),
        )


def industry_typical_h100_mfu(path: Path) -> float:
    data = yaml.safe_load(path.read_text())
    return float(data["sensitivity"]["industry_typical_h100_mfu"])


def prefill_flops(model: ModelSpec, prefill_tokens: int, total_context_tokens: int) -> float:
    """FLOPs to prefill `prefill_tokens` new tokens against a
    `total_context_tokens`-long resident context. See module docstring for
    the derivation of both terms."""
    linear_term = 2.0 * model.n_params * prefill_tokens
    attention_term = (
        2.0 * model.n_layers * model.hidden_size * prefill_tokens * total_context_tokens
    )
    return linear_term + attention_term


def predicted_seconds(flops: float, hardware: HardwareSpec, efficiency: float) -> float:
    achieved_flops_per_sec = hardware.peak_flops_per_sec * efficiency
    return flops / achieved_flops_per_sec


def gpu_seconds_to_usd(seconds: float, hardware: HardwareSpec) -> float:
    if hardware.usd_per_gpu_hour is None:
        raise ValueError(f"{hardware.name} has no usd_per_gpu_hour — not a rentable target")
    return seconds / 3600.0 * hardware.usd_per_gpu_hour


@dataclass(frozen=True)
class EfficiencyFit:
    """`achieved_flops / hardware.peak_flops_per_sec`, fit per-step from real
    `ttft_ms`/`prefill_tokens`/`prompt_tokens` measurements — spec's
    "validate the model against your own measurements first," not an
    assumed constant."""

    model: str
    hardware: str
    n_steps: int
    median_efficiency: float
    iqr_low: float
    iqr_high: float


def fit_efficiency(
    steps: pd.DataFrame, model: ModelSpec, hardware: HardwareSpec, *, min_prefill_tokens: int = 0
) -> EfficiencyFit:
    """`min_prefill_tokens` excludes small steps from the fit. This matters:
    fit on every step unfiltered, achieved/peak efficiency rises
    monotonically with step size (measured on this project's own data:
    ~5% for <=50-token steps, ~21% for 50-200, ~51% for 200-1000, ~81% for
    >1000) — fixed per-request overhead (HTTP round-trip, vLLM scheduling,
    kernel launch) dominates tiny requests and amortizes away for large
    ones, so a single blended constant mostly reflects how many tiny steps
    happen to be in the sample, not a stable hardware property. Since what
    this module extrapolates is *compaction's* savings — dominated by large
    reprefill-after-compaction events, not tiny single-turn appends — the
    physically meaningful, transferable quantity is the compute-bound
    (large-step) efficiency, not the blend. See
    `experiments/cost_model_report.py`'s stratified breakdown for the
    evidence this default doesn't use unless the caller opts in."""
    ratios: list[float] = []
    for row in steps.itertuples():
        prefill = int(row.prefill_tokens)  # type: ignore[arg-type]
        ttft_s = float(row.ttft_ms) / 1000.0  # type: ignore[arg-type]
        if prefill <= min_prefill_tokens or ttft_s <= 0:
            continue
        flops = prefill_flops(model, prefill, int(row.prompt_tokens))  # type: ignore[arg-type]
        achieved_flops_per_sec = flops / ttft_s
        ratios.append(achieved_flops_per_sec / hardware.peak_flops_per_sec)

    if not ratios:
        raise ValueError("no usable steps (need prefill_tokens > 0 and ttft_ms > 0) to fit from")

    ratios.sort()
    median = statistics.median(ratios)
    if len(ratios) >= 4:
        q1, _, q3 = statistics.quantiles(ratios, n=4, method="inclusive")
    else:
        q1 = q3 = median
    return EfficiencyFit(
        model=model.name,
        hardware=hardware.name,
        n_steps=len(ratios),
        median_efficiency=median,
        iqr_low=q1,
        iqr_high=q3,
    )


@dataclass(frozen=True)
class ValidationResult:
    """How well `prefill_flops` + a fitted efficiency predicts real,
    held-in `ttft_ms` — not a proof of exactness (see module docstring's
    caveats on both formula terms), a measured error bar on the model
    itself."""

    n_steps: int
    median_abs_pct_error: float
    p90_abs_pct_error: float


def validate_against_measurements(
    steps: pd.DataFrame,
    model: ModelSpec,
    hardware: HardwareSpec,
    efficiency: float,
    *,
    min_prefill_tokens: int = 0,
) -> ValidationResult:
    """`min_prefill_tokens` — see `fit_efficiency`'s docstring; pass the same
    value used to fit `efficiency` to validate on the regime it was actually
    calibrated for."""
    errors: list[float] = []
    for row in steps.itertuples():
        prefill = int(row.prefill_tokens)  # type: ignore[arg-type]
        actual_s = float(row.ttft_ms) / 1000.0  # type: ignore[arg-type]
        if prefill <= min_prefill_tokens or actual_s <= 0:
            continue
        flops = prefill_flops(model, prefill, int(row.prompt_tokens))  # type: ignore[arg-type]
        predicted_s = predicted_seconds(flops, hardware, efficiency)
        errors.append(abs(predicted_s - actual_s) / actual_s)

    if not errors:
        raise ValueError("no usable steps to validate against")
    errors.sort()
    median = statistics.median(errors)
    p90_idx = min(len(errors) - 1, int(round(0.9 * (len(errors) - 1))))
    return ValidationResult(
        n_steps=len(errors), median_abs_pct_error=median, p90_abs_pct_error=errors[p90_idx]
    )


@dataclass(frozen=True)
class ExtrapolationResult:
    """spec: "derive tokens-saved → FLOPs-saved → GPU-seconds-saved →
    dollars" plus "state the assumptions and the error bars" — `efficiency`
    and every downstream field are explicit inputs/outputs, never hidden
    constants, so a caller can (and `experiments/cost_model_report.py`
    does) report this at multiple efficiency assumptions rather than one
    unlabeled point estimate."""

    baseline_policy: str
    comparison_policy: str
    model: str
    hardware: str
    efficiency: float
    baseline_flops: float
    comparison_flops: float
    flops_saved: float
    gpu_seconds_saved: float
    usd_saved: float | None


def _sum_prefill_flops(steps: pd.DataFrame, policy: str, model: ModelSpec) -> float:
    total = 0.0
    for row in steps[steps["policy"] == policy].itertuples():
        prefill = int(row.prefill_tokens)  # type: ignore[arg-type]
        prompt = int(row.prompt_tokens)  # type: ignore[arg-type]
        total += prefill_flops(model, prefill, prompt)
    return total


def extrapolate_savings(
    steps: pd.DataFrame,
    *,
    baseline_policy: str,
    comparison_policy: str,
    model: ModelSpec,
    hardware: HardwareSpec,
    efficiency: float,
) -> ExtrapolationResult:
    """Recomputes FLOPs for `baseline_policy`'s and `comparison_policy`'s
    entire real, already-measured per-step token sequences (every
    trajectory, every step — not one summary statistic) under `model`'s
    architecture. "Extrapolate to 70B" means replaying these exact measured
    shapes through a 70B-sized formula, not linearly scaling one aggregate
    number by a parameter-count ratio.

    `usd_saved` is `None` when `hardware.usd_per_gpu_hour` is unset (e.g.
    this project's own laptop GPU, which isn't a rentable target) —
    GPU-seconds-saved is still meaningful there, only the dollar conversion
    is inapplicable."""
    baseline_flops = _sum_prefill_flops(steps, baseline_policy, model)
    comparison_flops = _sum_prefill_flops(steps, comparison_policy, model)
    flops_saved = baseline_flops - comparison_flops
    seconds_saved = predicted_seconds(flops_saved, hardware, efficiency)
    usd_saved = None if hardware.usd_per_gpu_hour is None else gpu_seconds_to_usd(
        seconds_saved, hardware
    )
    return ExtrapolationResult(
        baseline_policy=baseline_policy,
        comparison_policy=comparison_policy,
        model=model.name,
        hardware=hardware.name,
        efficiency=efficiency,
        baseline_flops=baseline_flops,
        comparison_flops=comparison_flops,
        flops_saved=flops_saved,
        gpu_seconds_saved=seconds_saved,
        usd_saved=usd_saved,
    )
