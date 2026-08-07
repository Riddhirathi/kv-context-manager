#!/usr/bin/env python3
"""Phase 2 — cache-preserving context layout (AGENTKV_SPEC.md §2).

Replays every recorded trajectory under both `policies/naive.py` (baseline)
and `policies/append_only.py` (the L1 contribution) against the project's
measurement model, and reports whether append-only reduces cumulative
prefill tokens per trajectory by a statistically significant margin (Gate 2).
Produces:

- results/phase2_steps.parquet    (one row per agent step, both policies, every trajectory)
- results/phase2_events.parquet   (one row per fired compaction event, both policies)
- results/phase2_comparison.png   (cumulative prefill, both policies overlaid, one trajectory)
- results/phase2_gate2_summary.md (Gate 2 statement: per-trajectory reduction + paired
  Wilcoxon test)

Interleaving (spec §0.2: "never run all of policy A then all of policy B."):
each trajectory runs both policies back-to-back, alternating which policy
goes first from one trajectory to the next, so thermal/clock drift is spread
evenly across conditions rather than concentrated in an early-vs-late split.
This is coarser than per-request interleaving (which would require driving
two independent `ContextState` machines truly concurrently against one
physical cache) but keeps the harness correct and simple for this first cut.

Usage (inside the WSL venv):
    python experiments/phase2_layout.py
    python experiments/phase2_layout.py --trajectories traj-000 --max-steps 40
    python experiments/phase2_layout.py --align   # apply block alignment (spec §2.3)
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from agentkv.bench.divergence import (  # noqa: E402
    CompactionEventCollector,
    TheoryMismatchError,
    assert_theory_matches,
    build_compaction_event_record,
    compute_divergence_check,
)
from agentkv.bench.replay import Turn, iter_trajectories  # noqa: E402
from agentkv.bench.stats import wilcoxon_signed_rank  # noqa: E402
from agentkv.context.alignment import render_context_state_aligned  # noqa: E402
from agentkv.context.layout import Tokenizer, render_turns_to_token_ids  # noqa: E402
from agentkv.metrics.collector import MetricsCollector, StepRecord  # noqa: E402
from agentkv.metrics.rigor import (  # noqa: E402
    ClockDriftExceeded,
    GpuClockLock,
    GpuSample,
    GpuSampler,
    RigorConfig,
)
from agentkv.policies.append_only import AppendOnlyPolicy  # noqa: E402
from agentkv.policies.base import CompactionPolicy  # noqa: E402
from agentkv.policies.naive import LLMSummarizer, NaivePolicy  # noqa: E402
from agentkv.serving.engine import ModelConfig, VLLMEngine  # noqa: E402
from agentkv.viz.figures import plot_policy_comparison  # noqa: E402

DEFAULT_EXCLUDE = {"traj-007"}  # short/interrupted — see results/phase0_summary.md


class _ZeroGpuSampler:
    def sample(self) -> GpuSample:
        return GpuSample(temp_c=0, sm_clock_mhz=0, power_w=0.0)

    def shutdown(self) -> None:
        return None


def _make_gpu_sampler() -> _ZeroGpuSampler | GpuSampler:
    try:
        return GpuSampler()
    except Exception as exc:  # noqa: BLE001 - NVML init failure, not a code bug
        print(f"warning: GpuSampler unavailable ({exc}); logging zeroed GPU telemetry.")
        return _ZeroGpuSampler()


@dataclass
class TrajectoryRunResult:
    trajectory_id: str
    policy: str
    n_steps: int
    n_events: int
    total_prefill_tokens: int
    event_prefill_tokens: int
    total_wallclock_ms: float
    event_wallclock_ms: float
    # append_only only: divergence_token_idx of each fired event, in order —
    # spec §2.2's "key property to verify": this should be non-decreasing,
    # since `frozen` only ever grows and earlier segments are never rewritten.
    event_divergence_token_indices: list[int] = field(default_factory=list)


def run_trajectory(
    *,
    engine: VLLMEngine,
    tokenizer: Tokenizer,
    policy: CompactionPolicy,
    turns: list[Turn],
    trajectory_id: str,
    seed: int,
    block_size: int,
    gpu_sampler: _ZeroGpuSampler | GpuSampler,
    clock_lock: GpuClockLock | None,
    clock_window: deque[int],
    step_collector: MetricsCollector,
    event_collector: CompactionEventCollector,
    max_steps: int | None,
    initial_prev_prompt_ids: list[int],
    align: bool,
    filler_token_id: int | None,
) -> tuple[TrajectoryRunResult, list[int]]:
    """See `experiments/phase1_cliff.py`'s `run_trajectory` for why
    `prev_prompt_ids` threads across trajectory (and here, policy) runs
    rather than resetting to `[]`: the physical vLLM cache is one shared pool
    for the whole sweep, regardless of which policy or trajectory issued the
    previous request."""
    context_turns: list[Turn] = []
    prev_prompt_ids: list[int] = initial_prev_prompt_ids
    total_prefill = 0
    event_prefill = 0
    total_wallclock = 0.0
    event_wallclock = 0.0
    n_events = 0
    event_divergence_indices: list[int] = []

    limit = len(turns) if max_steps is None else min(max_steps, len(turns))
    for step_idx in range(limit):
        turns_before_compaction = context_turns + [turns[step_idx]]
        tokens_before = len(render_turns_to_token_ids(turns_before_compaction, tokenizer))

        context_turns, fired = policy.maybe_compact(turns_before_compaction, step_idx)

        if align and isinstance(policy, AppendOnlyPolicy):
            assert filler_token_id is not None
            new_prompt_ids = render_context_state_aligned(
                policy.state, tokenizer, block_size=block_size, filler_token_id=filler_token_id
            )
        else:
            new_prompt_ids = render_turns_to_token_ids(context_turns, tokenizer)

        result = engine.generate_step(new_prompt_ids, max_tokens=8)
        check = compute_divergence_check(
            step_idx=step_idx,
            prev_prompt_ids=prev_prompt_ids,
            new_prompt_ids=new_prompt_ids,
            measured_cached_tokens=result.cached_tokens,
            block_size=block_size,
        )
        # Spec §1.4, still enforced here: a mismatch means the cache model is
        # wrong, and every downstream Phase 2 number would be untrustworthy.
        # Re-raised with policy/trajectory context — unlike Phase 1 (one
        # policy, one continuous stream), Phase 2 runs two independently
        # stateful policies back-to-back against one shared physical cache,
        # so knowing *which* run hit the mismatch is essential to diagnose it.
        try:
            assert_theory_matches(check)
        except TheoryMismatchError as exc:
            raise TheoryMismatchError(
                f"[{trajectory_id}/{policy.name}, fired={fired}] {exc} "
                f"(prev_prompt_tokens={check.prev_prompt_tokens}, "
                f"new_prompt_tokens={check.new_prompt_tokens})"
            ) from exc

        gpu = gpu_sampler.sample()
        if clock_lock is not None and not fired:
            # See phase1_cliff.py's run_trajectory for the full rationale
            # (rolling-median comparison, warn-not-fail on breach) — identical
            # policy, reused verbatim here.
            clock_window.append(gpu.sm_clock_mhz)
            try:
                clock_lock.check_drift(int(statistics.median(clock_window)))
            except ClockDriftExceeded as exc:
                print(f"warning: step {step_idx}: {exc}")

        wallclock_ms = result.ttft_ms + result.decode_ms
        step_collector.record(
            StepRecord(
                step_idx=step_idx,
                prompt_tokens=result.prompt_tokens,
                cached_tokens=result.cached_tokens,
                prefill_tokens=result.prefill_tokens,
                ttft_ms=result.ttft_ms,
                decode_ms=result.decode_ms,
                output_tokens=result.output_tokens,
                gpu_mem_bytes=0,
                policy=policy.name,
                seed=seed,
                temp_c=gpu.temp_c,
                sm_clock_mhz=gpu.sm_clock_mhz,
                power_w=gpu.power_w,
            )
        )
        total_prefill += result.prefill_tokens
        total_wallclock += wallclock_ms

        if fired:
            n_events += 1
            event_prefill += result.prefill_tokens
            event_wallclock += wallclock_ms
            event_divergence_indices.append(check.divergence_token_idx)
            event_collector.record(
                build_compaction_event_record(
                    step_idx=step_idx,
                    policy=policy.name,
                    seed=seed,
                    check=check,
                    tokens_before_compaction=tokens_before,
                    tokens_after_compaction=len(new_prompt_ids),
                    tokens_reprefilled=result.prefill_tokens,
                    ttft_ms=result.ttft_ms,
                )
            )

        prev_prompt_ids = new_prompt_ids

    return (
        TrajectoryRunResult(
            trajectory_id=trajectory_id,
            policy=policy.name,
            n_steps=limit,
            n_events=n_events,
            total_prefill_tokens=total_prefill,
            event_prefill_tokens=event_prefill,
            total_wallclock_ms=total_wallclock,
            event_wallclock_ms=event_wallclock,
            event_divergence_token_indices=event_divergence_indices,
        ),
        prev_prompt_ids,
    )


def write_gate2_summary(
    naive_results: list[TrajectoryRunResult],
    append_only_results: list[TrajectoryRunResult],
    out_path: Path,
) -> str:
    """Spec §2's Gate 2: "Append-only layout reduces cumulative prefill
    tokens per 100-step trajectory by a measurable, statistically
    significant margin vs. naive, at equal or better action agreement."

    Action agreement itself is Phase 3 infrastructure (`bench/agreement.py`)
    and is out of scope here — this reports the prefill-reduction half of
    Gate 2 only; the summary says so explicitly rather than implying a full
    gate pass.
    """
    by_traj_naive = {r.trajectory_id: r for r in naive_results}
    by_traj_append = {r.trajectory_id: r for r in append_only_results}
    shared_ids = sorted(set(by_traj_naive) & set(by_traj_append))

    naive_totals = [float(by_traj_naive[t].total_prefill_tokens) for t in shared_ids]
    append_totals = [float(by_traj_append[t].total_prefill_tokens) for t in shared_ids]
    reductions_pct = [
        100.0 * (n - a) / n for n, a in zip(naive_totals, append_totals, strict=True) if n > 0
    ]
    test = wilcoxon_signed_rank(naive_totals, append_totals)

    monotonic_ok = 0
    monotonic_checked = 0
    for r in append_only_results:
        if len(r.event_divergence_token_indices) < 2:
            continue
        monotonic_checked += 1
        indices = r.event_divergence_token_indices
        if all(b >= a for a, b in zip(indices, indices[1:], strict=False)):
            monotonic_ok += 1

    def fmt(values: list[float]) -> str:
        if not values:
            return "n/a"
        if len(values) < 2:
            return f"{values[0]:.1f} (n=1, too few runs for an IQR)"
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
        return f"median {statistics.median(values):.1f}, IQR [{q1:.1f}, {q3:.1f}] (n={len(values)})"

    lines = [
        "# Phase 2 — Gate 2 summary (prefill-reduction half; action agreement is Phase 3)",
        "",
        f"Trajectories compared: {len(shared_ids)}",
        "",
        f"- Prefill-token reduction, append_only vs. naive: {fmt(reductions_pct)}%",
        f"- Paired Wilcoxon signed-rank test on total prefill tokens (naive vs. append_only): "
        f"W={test.w_statistic:.1f}, z={test.z_statistic:.3f}, p={test.p_value:.4g} "
        f"(n_pairs={test.n_pairs}, n_nonzero={test.n_nonzero}) -> "
        f"{'SIGNIFICANT' if test.significant_at_05 else 'not significant'} at alpha=0.05",
        f"- Monotonic prefix growth (successive append_only events' divergence index "
        f"non-decreasing): {monotonic_ok}/{monotonic_checked} trajectories "
        f"(spec §2.2's headline systems property)",
        "",
        "Per-trajectory detail:",
        "",
        "| trajectory | naive prefill | append_only prefill | reduction % |",
        "|---|---|---|---|",
    ]
    for t in shared_ids:
        n_val = by_traj_naive[t].total_prefill_tokens
        a_val = by_traj_append[t].total_prefill_tokens
        pct = 100.0 * (n_val - a_val) / n_val if n_val else 0.0
        lines.append(f"| {t} | {n_val} | {a_val} | {pct:.1f}% |")

    text = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories-dir", type=Path, default=REPO_ROOT / "trajectories")
    parser.add_argument("--trajectories", nargs="*", default=None)
    parser.add_argument("--exclude", nargs="*", default=sorted(DEFAULT_EXCLUDE))
    parser.add_argument("--comparison-trajectory", default="traj-000")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--use-fallback", action="store_true")
    parser.add_argument(
        "--align", action="store_true", help="Apply block alignment (spec §2.3) to append_only."
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    config = ModelConfig.from_yaml(
        REPO_ROOT / "configs" / "model.yaml", use_fallback=args.use_fallback
    )
    naive_config = yaml.safe_load((REPO_ROOT / "configs" / "policies" / "naive.yaml").read_text())
    append_config = yaml.safe_load(
        (REPO_ROOT / "configs" / "policies" / "append_only.yaml").read_text()
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)  # type: ignore[no-untyped-call]
    filler_token_id = tokenizer.encode(" ")[0] if args.align else None

    all_trajectories = list(iter_trajectories(args.trajectories_dir))
    excluded = set(args.exclude)
    if args.trajectories is not None:
        wanted = set(args.trajectories)
        trajectories = [t for t in all_trajectories if t.trajectory_id in wanted]
    else:
        trajectories = [t for t in all_trajectories if t.trajectory_id not in excluded]
    if not trajectories:
        raise SystemExit("No trajectories selected.")
    print(f"Running {len(trajectories)} trajectories: {[t.trajectory_id for t in trajectories]}")

    rigor_config = RigorConfig.from_yaml(REPO_ROOT / "configs" / "rigor.yaml")
    gpu_sampler = _make_gpu_sampler()
    has_real_gpu_telemetry = isinstance(gpu_sampler, GpuSampler)
    clock_lock = GpuClockLock(rigor_config) if has_real_gpu_telemetry else None
    clock_window: deque[int] = deque(maxlen=5)
    step_collector = MetricsCollector()
    event_collector = CompactionEventCollector()

    threshold_naive = int(naive_config["threshold_pct_of_window"] * config.max_model_len)
    threshold_append = int(append_config["threshold_pct_of_window"] * config.max_model_len)

    naive_results: list[TrajectoryRunResult] = []
    append_results: list[TrajectoryRunResult] = []

    try:
        with VLLMEngine(config, startup_timeout_s=600.0) as engine:
            print(f"KV cache capacity: {engine.kv_cache_capacity_tokens()} tokens")
            warmup_clock_samples: list[int] = []
            for _ in range(15):
                engine.generate_step(tokenizer.encode("warmup"), max_tokens=4)
                warmup_clock_samples.append(gpu_sampler.sample().sm_clock_mhz)
            if clock_lock is not None:
                clock_lock.acquire(warmup_clock_samples)
            summarizer = LLMSummarizer(
                engine, tokenizer, max_summary_tokens=naive_config["max_summary_tokens"]
            )

            prev_prompt_ids: list[int] = []
            for i, trajectory in enumerate(trajectories):
                naive_policy = NaivePolicy(
                    tokenizer,
                    summarizer,
                    threshold_tokens=threshold_naive,
                    retire_fraction=naive_config["retire_fraction"],
                )
                append_policy = AppendOnlyPolicy(
                    tokenizer,
                    summarizer,
                    threshold_tokens=threshold_append,
                    retire_fraction=append_config["retire_fraction"],
                )
                # Alternate which policy runs first per trajectory (spec
                # §0.2's interleaved A/B, applied at trajectory granularity —
                # see module docstring).
                order = (
                    [("naive", naive_policy), ("append_only", append_policy)]
                    if i % 2 == 0
                    else [("append_only", append_policy), ("naive", naive_policy)]
                )
                print(f"--- {trajectory.trajectory_id} ({len(trajectory)} turns) ---")
                for policy_name, policy in order:
                    result, prev_prompt_ids = run_trajectory(
                        engine=engine,
                        tokenizer=tokenizer,
                        policy=policy,
                        turns=trajectory.turns,
                        trajectory_id=trajectory.trajectory_id,
                        seed=config.seed,
                        block_size=config.block_size,
                        gpu_sampler=gpu_sampler,
                        clock_lock=clock_lock,
                        clock_window=clock_window,
                        step_collector=step_collector,
                        event_collector=event_collector,
                        max_steps=args.max_steps,
                        initial_prev_prompt_ids=prev_prompt_ids,
                        align=args.align,
                        filler_token_id=filler_token_id,
                    )
                    (naive_results if policy_name == "naive" else append_results).append(result)
                    print(
                        f"  [{policy_name}] steps={result.n_steps} events={result.n_events} "
                        f"total_prefill={result.total_prefill_tokens} "
                        f"event_prefill={result.event_prefill_tokens}"
                    )
    finally:
        if clock_lock is not None:
            clock_lock.release()
        gpu_sampler.shutdown()

    steps_frame = step_collector.to_frame()
    step_collector.flush(args.out_dir / "phase2_steps.parquet")
    event_collector.flush(args.out_dir / "phase2_events.parquet")

    comparison_id = args.comparison_trajectory
    runs: dict[str, object] = {}
    for policy_name, results in (("naive", naive_results), ("append_only", append_results)):
        idx = next((i for i, r in enumerate(results) if r.trajectory_id == comparison_id), 0)
        mask = steps_frame["policy"] == policy_name
        policy_steps = steps_frame[mask].reset_index(drop=True)
        start = sum(r.n_steps for r in results[:idx])
        end = start + results[idx].n_steps
        run_len = results[idx].n_steps
        runs[policy_name] = policy_steps.iloc[start:end].assign(step_idx=range(run_len))

    out_png = args.out_dir / "phase2_comparison.png"
    plot_policy_comparison(runs, out_png)  # type: ignore[arg-type]
    print(f"Wrote {out_png}")

    out_summary = args.out_dir / "phase2_gate2_summary.md"
    summary_text = write_gate2_summary(naive_results, append_results, out_summary)
    print()
    print(summary_text)


if __name__ == "__main__":
    main()
