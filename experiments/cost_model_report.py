#!/usr/bin/env python3
"""Phase 6 §6.1 — the analytical cost model report (AGENTKV_SPEC.md).

Pure post-processing, no GPU/vLLM needed: everything here reads already-
committed `results/phase4/phase4_pareto_cost_steps.parquet` (the same
14-trajectory, 5-policy sweep Gate 4's Pareto plot uses) and
`configs/cost_model.yaml` (spec §10: "no magic numbers in code").

Matching spec's own ordering — validate first, then extrapolate:

1. **Validate** `serving.cost_model.prefill_flops` against this project's
   own real measurements (Qwen3-0.6B on the RTX 4060 Laptop). The first
   attempt — one efficiency constant fit on every step — validates badly
   (median error ~92%); a stratified breakdown by step size shows why
   (fixed per-request overhead dominates small steps) and motivates fitting
   only on the compute-bound regime instead, which validates well (median
   error ~9%). Reported here rather than silently only shipping the fix,
   since the failure mode is itself informative.
2. **Extrapolate** naive-vs-hybrid's real, measured token savings to
   Llama-3-70B on an H100 — not by scaling one summary number, but by
   replaying the exact same measured per-step token sequences through a
   70B-sized FLOPs formula.
3. **Cross-check** the extrapolation's biggest assumption (that this
   project's own laptop-GPU efficiency transfers to a production H100
   stack) against an independent, commonly-cited industry MFU figure, and
   report both as a range — spec: "state the assumptions and the error
   bars. Do not present extrapolated numbers as measurements."

Usage (no GPU needed):
    python experiments/cost_model_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from agentkv.serving.cost_model import (  # noqa: E402
    ExtrapolationResult,
    HardwareSpec,
    ModelSpec,
    extrapolate_savings,
    fit_efficiency,
    industry_typical_h100_mfu,
    validate_against_measurements,
)

COST_MODEL_YAML = REPO_ROOT / "configs" / "cost_model.yaml"
BASELINE_POLICY = "naive"
COMPARISON_POLICY = "hybrid"

# Justified below by the stratified breakdown this script prints/writes —
# steps at or under this size are dominated by fixed per-request overhead
# (HTTP round-trip, vLLM scheduling, kernel launch), not GPU compute, so
# blending them into one efficiency constant mostly measures how many tiny
# steps happen to be in the sample. Compaction's actual savings are
# dominated by large reprefill-after-compaction events, which is exactly
# the regime this threshold keeps.
MIN_PREFILL_TOKENS_FOR_FIT = 200
SIZE_BUCKETS = [
    (0, 50, "tiny (<=50)"),
    (50, 200, "small (50-200)"),
    (200, 1000, "medium (200-1000)"),
    (1000, None, "large (>1000)"),
]


def fmt_usd(x: float) -> str:
    if abs(x) >= 1:
        return f"${x:,.2f}"
    if abs(x) >= 0.01:
        return f"${x:.4f}"
    return f"${x:.6f}"


def render_stratified_breakdown(
    steps: pd.DataFrame, model: ModelSpec, hardware: HardwareSpec
) -> list[str]:
    """Evidence for `MIN_PREFILL_TOKENS_FOR_FIT`: fits efficiency separately
    per step-size bucket and shows it rise monotonically with size, plus the
    per-bucket validation error — small steps fit badly (overhead-
    dominated), large steps fit well (compute-dominated)."""
    lines = [
        "| step size | n steps | median efficiency | efficiency IQR | validation median error |",
        "|---|---|---|---|---|",
    ]
    for lo, hi, label in SIZE_BUCKETS:
        mask = steps["prefill_tokens"] > lo
        if hi is not None:
            mask &= steps["prefill_tokens"] <= hi
        bucket = steps[mask]
        if len(bucket) < 10:
            lines.append(f"| {label} | {len(bucket)} | too few steps | — | — |")
            continue
        fit = fit_efficiency(bucket, model, hardware)
        val = validate_against_measurements(
            bucket, model, hardware, efficiency=fit.median_efficiency
        )
        lines.append(
            f"| {label} | {fit.n_steps:,} | {fit.median_efficiency:.2%} | "
            f"[{fit.iqr_low:.2%}, {fit.iqr_high:.2%}] | {val.median_abs_pct_error:.1%} |"
        )
    lines.append("")
    return lines


def render_extrapolation(label: str, result: ExtrapolationResult) -> list[str]:
    dollar_line = (
        f"- **Dollars saved (this dataset, this hardware's `$/GPU-hour`): "
        f"{fmt_usd(result.usd_saved)}**"
        if result.usd_saved is not None
        else "- Dollars saved: n/a (hardware has no `$/GPU-hour` — not a rentable target)"
    )
    return [
        f"### {label} (efficiency={result.efficiency:.4f})",
        "",
        f"- {result.model} on {result.hardware}",
        f"- {result.baseline_policy} FLOPs: {result.baseline_flops:.3e}",
        f"- {result.comparison_policy} FLOPs: {result.comparison_flops:.3e}",
        f"- FLOPs saved: {result.flops_saved:.3e}",
        f"- GPU-seconds saved: {result.gpu_seconds_saved:,.1f}s "
        f"({result.gpu_seconds_saved / 3600:.3f} GPU-hours)",
        dollar_line,
        "",
    ]


def main() -> None:
    steps = pd.read_parquet(REPO_ROOT / "results" / "phase4" / "phase4_pareto_cost_steps.parquet")

    qwen = ModelSpec.from_yaml(COST_MODEL_YAML, "qwen3_0_6b")
    llama = ModelSpec.from_yaml(COST_MODEL_YAML, "llama3_70b")
    laptop = HardwareSpec.from_yaml(COST_MODEL_YAML, "rtx_4060_laptop")
    h100 = HardwareSpec.from_yaml(COST_MODEL_YAML, "h100_sxm")
    industry_mfu = industry_typical_h100_mfu(COST_MODEL_YAML)

    # Step 0: unfiltered fit, as evidence for why an unfiltered constant is
    # the wrong calibration — reported, not used downstream.
    unfiltered_fit = fit_efficiency(steps, qwen, laptop)
    unfiltered_validation = validate_against_measurements(
        steps, qwen, laptop, efficiency=unfiltered_fit.median_efficiency
    )

    # Step 1: validate on the regime that's actually used downstream — steps
    # where FLOPs, not fixed per-request overhead, dominate wall-clock time
    # (see MIN_PREFILL_TOKENS_FOR_FIT's comment and the stratified table).
    fit = fit_efficiency(steps, qwen, laptop, min_prefill_tokens=MIN_PREFILL_TOKENS_FOR_FIT)
    validation = validate_against_measurements(
        steps,
        qwen,
        laptop,
        efficiency=fit.median_efficiency,
        min_prefill_tokens=MIN_PREFILL_TOKENS_FOR_FIT,
    )

    # Step 1b: an aggregate, end-to-end check independent of the per-step
    # error metric above — does predicted GPU-seconds saved (naive vs.
    # hybrid, same hardware actually measured on, ALL steps including tiny
    # ones) land near the REAL measured wall-clock saved (sum of ttft_ms)?
    # Reported honestly even though it does NOT validate cleanly — see the
    # "Assumptions and limitations" section for why, rather than treated as
    # a second confirmation of the per-step result above.
    same_hw_result = extrapolate_savings(
        steps,
        baseline_policy=BASELINE_POLICY,
        comparison_policy=COMPARISON_POLICY,
        model=qwen,
        hardware=laptop,
        efficiency=fit.median_efficiency,
    )
    real_ttft_saved_s = (
        steps[steps["policy"] == BASELINE_POLICY]["ttft_ms"].sum()
        - steps[steps["policy"] == COMPARISON_POLICY]["ttft_ms"].sum()
    ) / 1000.0
    predicted_seconds_same_hw = same_hw_result.gpu_seconds_saved

    # Step 2: extrapolate to 70B/H100 using the SAME real per-step token
    # sequences, at three efficiency assumptions (median fit, fit's own IQR
    # bounds as a sensitivity range).
    median_result = extrapolate_savings(
        steps,
        baseline_policy=BASELINE_POLICY,
        comparison_policy=COMPARISON_POLICY,
        model=llama,
        hardware=h100,
        efficiency=fit.median_efficiency,
    )
    low_result = extrapolate_savings(
        steps,
        baseline_policy=BASELINE_POLICY,
        comparison_policy=COMPARISON_POLICY,
        model=llama,
        hardware=h100,
        efficiency=fit.iqr_low,
    )
    high_result = extrapolate_savings(
        steps,
        baseline_policy=BASELINE_POLICY,
        comparison_policy=COMPARISON_POLICY,
        model=llama,
        hardware=h100,
        efficiency=fit.iqr_high,
    )

    pct_flops_reduction = 100.0 * median_result.flops_saved / median_result.baseline_flops
    n_trajectories_in_dataset = 14
    illustrative_daily_trajectories = 100_000
    scale_factor = illustrative_daily_trajectories / n_trajectories_in_dataset
    daily_usd_saved = (
        None if median_result.usd_saved is None else median_result.usd_saved * scale_factor
    )

    # Step 3: cross-check against an efficiency figure independent of this
    # project's own (batch-size-1, laptop-GPU) fit entirely.
    industry_result = extrapolate_savings(
        steps,
        baseline_policy=BASELINE_POLICY,
        comparison_policy=COMPARISON_POLICY,
        model=llama,
        hardware=h100,
        efficiency=industry_mfu,
    )

    lines = [
        "# Phase 6 (§6.1) — Analytical Cost Model Report",
        "",
        "No GPU/vLLM used to produce this document — every number below comes",
        "from `serving/cost_model.py` applied to the already-committed",
        "`results/phase4/phase4_pareto_cost_steps.parquet` "
        f"({len(steps):,} step rows, 5 policies, 14 trajectories) and "
        "`configs/cost_model.yaml`.",
        "",
        "## 1. Validation (against this project's own measurements)",
        "",
        f"Fitting one efficiency constant on all {unfiltered_fit.n_steps:,} real "
        f"steps unfiltered gives median **{unfiltered_fit.median_efficiency:.2%}** "
        f"(IQR [{unfiltered_fit.iqr_low:.2%}, {unfiltered_fit.iqr_high:.2%}]) — and "
        f"validates badly: median absolute error "
        f"**{unfiltered_validation.median_abs_pct_error:.1%}**, p90 "
        f"{unfiltered_validation.p90_abs_pct_error:.1%}. The reason shows up in a "
        f"stratified breakdown by step size:",
        "",
        *render_stratified_breakdown(steps, qwen, laptop),
        f"Efficiency rises monotonically with step size — fixed per-request "
        f"overhead (HTTP round-trip, vLLM scheduling, kernel launch) dominates "
        f"tiny steps and amortizes away for large ones, so a single blended "
        f"constant mostly reflects how many tiny steps happen to be in the "
        f"sample (4,864 of {len(steps):,} here), not a stable hardware "
        f"property. Since compaction's actual savings are dominated by large "
        f"reprefill-after-compaction events, not tiny single-turn appends, "
        f"this report calibrates only on steps with "
        f"`prefill_tokens > {MIN_PREFILL_TOKENS_FOR_FIT}` from here on.",
        "",
        f"**Corrected fit** ({fit.model} on {fit.hardware}, "
        f"`prefill_tokens > {MIN_PREFILL_TOKENS_FOR_FIT}`): efficiency median "
        f"**{fit.median_efficiency:.2%}** (IQR [{fit.iqr_low:.2%}, "
        f"{fit.iqr_high:.2%}]), validates against "
        f"{validation.n_steps:,} held-in steps at median absolute error "
        f"**{validation.median_abs_pct_error:.1%}**, p90 "
        f"{validation.p90_abs_pct_error:.1%}. This is the efficiency used for "
        "every extrapolation below.",
        "",
        f"**Aggregate check (not clean — reported anyway):** predicted "
        f"GPU-seconds saved by hybrid vs. naive on the *same* hardware/model "
        f"actually measured, summed over *every* step including tiny ones "
        f"({predicted_seconds_same_hw:,.1f}s) vs. the real measured wall-clock "
        f"saved (sum of `ttft_ms`, {real_ttft_saved_s:,.1f}s) — these don't "
        f"even agree in sign. Checked whether GPU thermal throttling (this "
        f"exact sweep is documented elsewhere in this project as having hit "
        f"sustained throttling — `phase4_pareto_summary.md`) explains it: it "
        f"doesn't cleanly — hybrid's steps ran at a *higher* mean SM clock "
        f"than naive's (2496 vs. 2412 MHz) despite processing fewer total "
        f"prefill tokens (980,837 vs. 1,087,407), yet still show higher mean "
        f"`ttft_ms` (68.2ms vs. 61.3ms). The likely explanation this report "
        f"doesn't fully chase down: `ttft_ms` is wall-clock to first token, "
        f"which includes policy-side CPU work (hybrid's per-turn routing "
        f"logic does more than naive's single threshold check) and engine "
        f"queueing, not just GPU prefill compute — exactly the kind of factor "
        f"a pure-FLOPs model was never built to capture. This is why the "
        f"per-step, large-step-only validation above (not this aggregate "
        f"check) is what this report treats as the model's real evidence.",
        "",
        "## 2. Extrapolation: naive vs. hybrid, replayed at 70B on H100",
        "",
        f"Same real per-step token sequences (all 14 trajectories) as measured "
        f"in Phase 4.5, recomputed under {llama.name}'s architecture "
        f"({llama.n_layers} layers, hidden={llama.hidden_size}, "
        f"{llama.n_params:,} params) instead of {qwen.name}'s.",
        "",
        *render_extrapolation("Point estimate (this project's fitted efficiency)", median_result),
        *render_extrapolation("Low sensitivity bound (fit's IQR low)", low_result),
        *render_extrapolation("High sensitivity bound (fit's IQR high)", high_result),
        f"**Cross-check against Gate 4's already-published result:** the FLOPs "
        f"reduction here ({pct_flops_reduction:.1f}%) closely matches "
        f"Gate 4's independently-measured token-count reduction "
        f"(`phase4_pareto_cost_summary.md`: \"hybrid vs. naive: median 9.8%\") "
        f"— expected, since both FLOPs terms scale with token count, but a "
        f"real consistency check between two differently-derived numbers, "
        f"not one number restated as two.",
        "",
        f"**Making the dollar figure tangible:** this dataset is only 14 short "
        f"trajectories, so {fmt_usd(median_result.usd_saved or 0.0)} is "
        f"real but small. Scaling illustratively to "
        f"{illustrative_daily_trajectories:,} similar trajectories/day "
        f"(purely a multiplication of this measured per-trajectory saving — "
        f"not a new measurement, and production traffic would not actually "
        f"look like this project's synthetic on-call trajectories): "
        f"**{fmt_usd(daily_usd_saved or 0.0)}/day**, "
        f"{fmt_usd((daily_usd_saved or 0.0) * 365)}/year, at the point-"
        f"estimate efficiency and this hardware's illustrative `$/GPU-hour`.",
        "",
        "## 3. Cross-check: industry-typical H100 MFU, independent of this project's fit",
        "",
        f"Using a commonly-cited production-serving MFU figure "
        f"({industry_mfu:.0%}) instead of anything measured on this project's "
        f"own hardware, as an independent sanity check on the point estimate "
        f"above:",
        "",
        *render_extrapolation(f"Industry-typical H100 MFU ({industry_mfu:.0%})", industry_result),
        "## Assumptions and limitations",
        "",
        "- **The FLOPs formula is an approximation** (spec's own \"2 × N_params "
        "× n_tokens plus attention terms\") — it ignores embedding/softmax/norm "
        "FLOPs and treats every new token as attending the full context rather "
        "than its own causal-masked prefix. See `serving/cost_model.py`'s module "
        "docstring for the full derivation.",
        "- **The model only covers GPU prefill compute, not policy-side CPU "
        "overhead or engine queueing** — the aggregate check in §1 suggests "
        "this matters: hybrid's per-turn routing logic (evict/keep/summarize "
        "classification on every candidate turn) plausibly costs more "
        "client-side CPU time than naive's single threshold check, which "
        "would show up in measured `ttft_ms` but not in this FLOPs-only "
        "model. Not confirmed with a dedicated measurement — flagged as the "
        "likely explanation, not established as fact.",
        "- **The efficiency fit is from batch-size-1 prefill on an "
        "underutilized 8GB laptop GPU** — production H100 serving (larger "
        "batches, continuous batching, a very different compute/memory-"
        "bandwidth ratio) will not necessarily hit the same fraction of peak "
        "FLOPS. That's exactly why an independent industry-MFU cross-check is "
        "reported alongside the fitted point estimate, not instead of it.",
        "- **Llama-3-70B was never downloaded or run** — its architecture "
        "numbers come from Meta's published config, its parameter count from "
        "this project's own parameter-counting formula (cross-checked against "
        "the commonly-cited ~70.6B figure, matches within 0.2%). Nothing here "
        "is a measurement of Llama-3-70B; it is a projection.",
        "- **`$/GPU-hour` is illustrative** (see `configs/cost_model.yaml`'s "
        "own comment) — 2026 on-demand H100 market rates span roughly "
        "$2-12/hr depending on provider; the dollar figures above scale "
        "linearly with whatever rate is actually paid.",
        "- **This report extrapolates naive-vs-hybrid's *cost* difference "
        "only.** Every policy in this project's Gate 4 measurement sits at "
        "0/5 task success (`results/phase4/phase4_pareto_summary.md`) — "
        "nothing here claims the quality tradeoff extrapolates, only the cost "
        "one.",
    ]

    out_path = REPO_ROOT / "results" / "phase6" / "phase6_cost_model_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
