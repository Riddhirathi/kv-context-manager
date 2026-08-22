from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from agentkv.serving.cost_model import (
    HardwareSpec,
    ModelSpec,
    extrapolate_savings,
    fit_efficiency,
    gpu_seconds_to_usd,
    industry_typical_h100_mfu,
    predicted_seconds,
    prefill_flops,
    validate_against_measurements,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COST_MODEL_YAML = REPO_ROOT / "configs" / "cost_model.yaml"

TINY_MODEL = ModelSpec(name="tiny", n_params=1000, n_layers=2, hidden_size=10)
TINY_HARDWARE = HardwareSpec(name="tiny-gpu", peak_flops_per_sec=1000.0, usd_per_gpu_hour=3.6)


def test_prefill_flops_matches_hand_computed_value():
    # linear = 2 * 1000 * 5 = 10,000; attention = 2 * 2 * 10 * 5 * 40 = 8,000
    flops = prefill_flops(TINY_MODEL, prefill_tokens=5, total_context_tokens=40)
    assert flops == 18_000.0


def test_prefill_flops_zero_prefill_is_zero():
    assert prefill_flops(TINY_MODEL, prefill_tokens=0, total_context_tokens=1000) == 0.0


def test_prefill_flops_scales_linearly_with_prefill_tokens_when_context_fixed():
    a = prefill_flops(TINY_MODEL, prefill_tokens=10, total_context_tokens=100)
    b = prefill_flops(TINY_MODEL, prefill_tokens=20, total_context_tokens=100)
    assert b == pytest.approx(2 * a)


def test_predicted_seconds_and_usd_conversion():
    # 1000 FLOPs at peak_flops_per_sec=1000, efficiency=0.5 -> achieved 500/s -> 2s.
    seconds = predicted_seconds(1000.0, TINY_HARDWARE, efficiency=0.5)
    assert seconds == pytest.approx(2.0)
    # $3.6/hr = $0.001/s -> 2s costs $0.002.
    assert gpu_seconds_to_usd(seconds, TINY_HARDWARE) == pytest.approx(0.002)


def test_gpu_seconds_to_usd_raises_when_hardware_not_rentable():
    laptop = HardwareSpec(name="laptop", peak_flops_per_sec=1000.0, usd_per_gpu_hour=None)
    with pytest.raises(ValueError, match="no usd_per_gpu_hour"):
        gpu_seconds_to_usd(10.0, laptop)


def _synthetic_steps(*, policy: str, true_efficiency: float, n: int = 20) -> pd.DataFrame:
    """Steps whose ttft_ms was generated from `prefill_flops` at a known,
    fixed efficiency — lets tests assert the fit recovers that exact value."""
    rows = []
    for i in range(1, n + 1):
        prefill = 10 * i
        prompt = 100 + 10 * i
        flops = prefill_flops(TINY_MODEL, prefill, prompt)
        ttft_s = flops / (TINY_HARDWARE.peak_flops_per_sec * true_efficiency)
        rows.append(
            {
                "policy": policy,
                "prefill_tokens": prefill,
                "prompt_tokens": prompt,
                "ttft_ms": ttft_s * 1000.0,
            }
        )
    return pd.DataFrame(rows)


def test_fit_efficiency_recovers_known_synthetic_efficiency():
    steps = _synthetic_steps(policy="naive", true_efficiency=0.2)
    fit = fit_efficiency(steps, TINY_MODEL, TINY_HARDWARE)
    assert fit.n_steps == 20
    assert fit.median_efficiency == pytest.approx(0.2, rel=1e-6)
    assert fit.iqr_low <= fit.median_efficiency <= fit.iqr_high


def test_fit_efficiency_skips_zero_prefill_and_zero_ttft_rows():
    steps = pd.concat(
        [
            _synthetic_steps(policy="naive", true_efficiency=0.3, n=5),
            pd.DataFrame(
                [
                    {"policy": "naive", "prefill_tokens": 0, "prompt_tokens": 50, "ttft_ms": 10.0},
                    {"policy": "naive", "prefill_tokens": 5, "prompt_tokens": 50, "ttft_ms": 0.0},
                ]
            ),
        ],
        ignore_index=True,
    )
    fit = fit_efficiency(steps, TINY_MODEL, TINY_HARDWARE)
    assert fit.n_steps == 5


def test_fit_efficiency_min_prefill_tokens_filters_small_steps():
    # 10 steps with prefill 10..100 (i=1..10) — with min_prefill_tokens=55,
    # only i=6..10 (prefill 60,70,80,90,100) survive.
    steps = _synthetic_steps(policy="naive", true_efficiency=0.25, n=10)
    fit = fit_efficiency(steps, TINY_MODEL, TINY_HARDWARE, min_prefill_tokens=55)
    assert fit.n_steps == 5
    assert fit.median_efficiency == pytest.approx(0.25, rel=1e-6)


def test_validate_against_measurements_min_prefill_tokens_matches_fit_regime():
    steps = _synthetic_steps(policy="naive", true_efficiency=0.25, n=10)
    fit = fit_efficiency(steps, TINY_MODEL, TINY_HARDWARE, min_prefill_tokens=55)
    result = validate_against_measurements(
        steps, TINY_MODEL, TINY_HARDWARE, efficiency=fit.median_efficiency, min_prefill_tokens=55
    )
    assert result.n_steps == 5
    assert result.median_abs_pct_error == pytest.approx(0.0, abs=1e-9)


def test_fit_efficiency_raises_on_no_usable_rows():
    steps = pd.DataFrame(
        [{"policy": "naive", "prefill_tokens": 0, "prompt_tokens": 50, "ttft_ms": 0.0}]
    )
    with pytest.raises(ValueError, match="no usable steps"):
        fit_efficiency(steps, TINY_MODEL, TINY_HARDWARE)


def test_validate_against_measurements_near_zero_error_on_self_consistent_data():
    steps = _synthetic_steps(policy="naive", true_efficiency=0.4)
    result = validate_against_measurements(steps, TINY_MODEL, TINY_HARDWARE, efficiency=0.4)
    assert result.n_steps == 20
    assert result.median_abs_pct_error == pytest.approx(0.0, abs=1e-9)
    assert result.p90_abs_pct_error == pytest.approx(0.0, abs=1e-9)


def test_validate_against_measurements_reports_error_when_efficiency_is_wrong():
    steps = _synthetic_steps(policy="naive", true_efficiency=0.4)
    # Fit assumes double the true efficiency -> predicted time is half actual -> ~50% error.
    result = validate_against_measurements(steps, TINY_MODEL, TINY_HARDWARE, efficiency=0.8)
    assert result.median_abs_pct_error == pytest.approx(0.5, rel=1e-6)


def test_extrapolate_savings_matches_hand_computed_difference():
    baseline = pd.DataFrame(
        [
            {"policy": "naive", "prefill_tokens": 10, "prompt_tokens": 100},
            {"policy": "naive", "prefill_tokens": 10, "prompt_tokens": 100},
        ]
    )
    comparison = pd.DataFrame(
        [{"policy": "hybrid", "prefill_tokens": 5, "prompt_tokens": 50}]
    )
    steps = pd.concat([baseline, comparison], ignore_index=True)

    result = extrapolate_savings(
        steps,
        baseline_policy="naive",
        comparison_policy="hybrid",
        model=TINY_MODEL,
        hardware=TINY_HARDWARE,
        efficiency=0.5,
    )
    expected_baseline_flops = 2 * prefill_flops(TINY_MODEL, 10, 100)
    expected_comparison_flops = prefill_flops(TINY_MODEL, 5, 50)
    assert result.baseline_flops == pytest.approx(expected_baseline_flops)
    assert result.comparison_flops == pytest.approx(expected_comparison_flops)
    assert result.flops_saved == pytest.approx(expected_baseline_flops - expected_comparison_flops)
    assert result.usd_saved == pytest.approx(
        gpu_seconds_to_usd(
            predicted_seconds(result.flops_saved, TINY_HARDWARE, 0.5), TINY_HARDWARE
        )
    )


def test_extrapolate_savings_negative_when_comparison_costs_more():
    steps = pd.DataFrame(
        [
            {"policy": "naive", "prefill_tokens": 5, "prompt_tokens": 50},
            {"policy": "kv_evict", "prefill_tokens": 50, "prompt_tokens": 500},
        ]
    )
    result = extrapolate_savings(
        steps,
        baseline_policy="naive",
        comparison_policy="kv_evict",
        model=TINY_MODEL,
        hardware=TINY_HARDWARE,
        efficiency=0.5,
    )
    assert result.flops_saved < 0
    assert result.usd_saved < 0


def test_model_spec_and_hardware_spec_load_from_shipped_config():
    qwen = ModelSpec.from_yaml(COST_MODEL_YAML, "qwen3_0_6b")
    assert qwen.name == "Qwen/Qwen3-0.6B"
    assert qwen.n_params > 0
    assert qwen.n_layers == 28

    llama = ModelSpec.from_yaml(COST_MODEL_YAML, "llama3_70b")
    assert llama.n_layers == 80
    assert llama.hidden_size == 8192

    laptop = HardwareSpec.from_yaml(COST_MODEL_YAML, "rtx_4060_laptop")
    assert laptop.usd_per_gpu_hour is None

    h100 = HardwareSpec.from_yaml(COST_MODEL_YAML, "h100_sxm")
    assert h100.usd_per_gpu_hour is not None
    assert h100.peak_flops_per_sec > laptop.peak_flops_per_sec


def test_industry_typical_h100_mfu_loads_from_shipped_config():
    mfu = industry_typical_h100_mfu(COST_MODEL_YAML)
    assert 0.0 < mfu < 1.0
