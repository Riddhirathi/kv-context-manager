#!/usr/bin/env python3
"""Phase 4.5 — prefill-cost half of the Pareto plot (AGENTKV_SPEC.md §4.5).

Extends `phase4_hybrid.py`'s naive/append_only/kv_evict/hybrid sweep with a
fifth policy, `none` (`policies/none.py`'s `NoOpPolicy` — never compacts,
ever). Gate 4 asks for "≥5 policies"; `attn_evict` (§4.2) was proven
infeasible on this hardware and never built, and `offload` (§4.3) is an
orthogonal serving-layer overlay measured on wall-clock, not a distinct
content policy on this prefill-token axis — see
`results/phase4/phase4_pareto_summary.md` for that decision. `none` is a
real, honestly-labeled ceiling point: maximum prefill cost, since nothing is
ever dropped or rewritten.

Same per-request harness, same theory check, as every other Phase 1-4.4
measurement. Writes to fresh `phase4_pareto_cost_*` output files rather than
appending to `phase4_hybrid_*` (`RecordCollector.flush` appends to an
existing file — reusing those names would silently mix a 4-policy run with a
5-policy one in the same parquet).

This script produces the X-axis (cumulative prefill tokens) half of Gate 4's
Pareto plot. The Y-axis (task success rate) comes from a separate script,
`phase4_pareto_success.py`, which drives live ledger-task episodes rather
than replaying recorded trajectories — see that script's docstring for why
the two axes are measured on different episode sets.

Usage (inside the WSL venv):
    python experiments/phase4_pareto_cost.py --trajectories traj-000 --max-steps 40
    python experiments/phase4_pareto_cost.py   # full sweep, all trajectories
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
from agentkv.policies.hybrid import HybridPolicy  # noqa: E402
from agentkv.policies.kv_evict import KVEvictPolicy  # noqa: E402
from agentkv.policies.naive import LLMSummarizer, NaivePolicy  # noqa: E402
from agentkv.policies.none import NoOpPolicy  # noqa: E402
from agentkv.serving.engine import ModelConfig, VLLMEngine  # noqa: E402
from agentkv.viz.figures import plot_policy_comparison  # noqa: E402

DEFAULT_EXCLUDE = {"traj-007"}  # short/interrupted — see results/phase0_summary.md
POLICY_ORDER = ["naive", "append_only", "kv_evict", "hybrid", "none"]


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
    event_divergence_token_indices: list[int] = field(default_factory=list)
    context_exceeded: bool = False


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
    max_model_len: int,
) -> tuple[TrajectoryRunResult, list[int]]:
    """Same shape as `phase4_hybrid.py`'s `run_trajectory` — duplicated rather
    than imported, matching this project's convention of self-contained
    experiment scripts. `max_model_len` is only needed here because this
    script (unlike `phase4_hybrid.py`) includes `none` (`NoOpPolicy`), whose
    context grows without bound — every other policy here compacts before
    getting anywhere near the model's context limit, so this check never
    fired in any prior Phase 4 sweep. See `phase4_pareto_summary.md` for the
    real crash this caught (a 165-turn trajectory overflowed `none`'s
    context at max_model_len=8192 partway through the first full sweep
    attempt) and why the fix is "stop this policy's run early and record
    it," not "silently truncate every policy to whatever `none` can survive."
    """
    context_turns: list[Turn] = []
    prev_prompt_ids: list[int] = initial_prev_prompt_ids
    total_prefill = 0
    event_prefill = 0
    total_wallclock = 0.0
    event_wallclock = 0.0
    n_events = 0
    event_divergence_indices: list[int] = []
    context_exceeded = False

    limit = len(turns) if max_steps is None else min(max_steps, len(turns))
    for step_idx in range(limit):
        turns_before_compaction = context_turns + [turns[step_idx]]
        tokens_before = len(render_turns_to_token_ids(turns_before_compaction, tokenizer))

        context_turns, fired = policy.maybe_compact(turns_before_compaction, step_idx)
        new_prompt_ids = render_turns_to_token_ids(context_turns, tokenizer)

        if len(new_prompt_ids) + 8 > max_model_len:
            context_exceeded = True
            break

        result = engine.generate_step(new_prompt_ids, max_tokens=8)
        check = compute_divergence_check(
            step_idx=step_idx,
            prev_prompt_ids=prev_prompt_ids,
            new_prompt_ids=new_prompt_ids,
            measured_cached_tokens=result.cached_tokens,
            block_size=block_size,
        )
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
            n_steps=step_idx if context_exceeded else limit,
            n_events=n_events,
            total_prefill_tokens=total_prefill,
            event_prefill_tokens=event_prefill,
            total_wallclock_ms=total_wallclock,
            event_wallclock_ms=event_wallclock,
            event_divergence_token_indices=event_divergence_indices,
            context_exceeded=context_exceeded,
        ),
        prev_prompt_ids,
    )


def write_pareto_cost_summary(
    results_by_policy: dict[str, list[TrajectoryRunResult]],
    out_path: Path,
) -> str:
    """Prefill-cost half of what Gate 4's Pareto plot needs — task success is
    a separate axis (spec §4.5), measured by `phase4_pareto_success.py`."""
    by_policy_by_traj = {
        policy: {r.trajectory_id: r for r in results}
        for policy, results in results_by_policy.items()
    }
    shared_ids = sorted(set.intersection(*(set(d) for d in by_policy_by_traj.values())))

    naive_totals = [float(by_policy_by_traj["naive"][t].total_prefill_tokens) for t in shared_ids]

    def fmt(values: list[float]) -> str:
        if not values:
            return "n/a"
        if len(values) < 2:
            return f"{values[0]:.1f} (n=1, too few runs for an IQR)"
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
        return f"median {statistics.median(values):.1f}, IQR [{q1:.1f}, {q3:.1f}] (n={len(values)})"

    lines = [
        "# Phase 4.5 — prefill-cost summary, all 5 policies (vs. naive)",
        "",
        f"Trajectories compared: {len(shared_ids)}",
        "",
    ]
    for policy in POLICY_ORDER:
        if policy == "naive":
            continue
        results = [by_policy_by_traj[policy][t] for t in shared_ids]
        incomplete = [r.trajectory_id for r in results if r.context_exceeded]
        if incomplete:
            lines.append(
                f"- {policy} vs. naive: SKIPPED — {len(incomplete)}/{len(results)} trajectories "
                f"hit `context_exceeded` (ran out of context before the trajectory finished: "
                f"{', '.join(incomplete)}). A reduction-vs-naive percentage or paired test on "
                f"`total_prefill_tokens` would be comparing a full-length run against a "
                f"truncated one, which would look like a cost *win* for the wrong reason. See "
                f"the per-trajectory table below and the `none` baseline discussion in "
                f"`phase4_pareto_summary.md` for what this actually shows."
            )
            lines.append("")
            continue
        totals = [float(r.total_prefill_tokens) for r in results]
        reductions_pct = [
            100.0 * (n - a) / n for n, a in zip(naive_totals, totals, strict=True) if n > 0
        ]
        test = wilcoxon_signed_rank(naive_totals, totals)
        lines.append(f"- Prefill-token reduction, {policy} vs. naive: {fmt(reductions_pct)}%")
        lines.append(
            f"- Paired Wilcoxon signed-rank test on total prefill tokens (naive vs. {policy}): "
            f"W={test.w_statistic:.1f}, z={test.z_statistic:.3f}, p={test.p_value:.4g} "
            f"(n_pairs={test.n_pairs}, n_nonzero={test.n_nonzero}) -> "
            f"{'SIGNIFICANT' if test.significant_at_05 else 'not significant'} at alpha=0.05"
        )
        lines.append("")

    lines.append(
        "Per-trajectory detail (a `*` marks a run that hit `context_exceeded` — the total is "
        "only over the steps actually completed, not the full trajectory):"
    )
    lines.append("")
    header = "| trajectory | " + " | ".join(f"{p} prefill" for p in POLICY_ORDER) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(POLICY_ORDER) + 1))
    for t in shared_ids:
        cells = [
            f"{by_policy_by_traj[p][t].total_prefill_tokens}"
            f"{'*' if by_policy_by_traj[p][t].context_exceeded else ''}"
            for p in POLICY_ORDER
        ]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")

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
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "phase4")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    config = ModelConfig.from_yaml(
        REPO_ROOT / "configs" / "model.yaml", use_fallback=args.use_fallback
    )
    naive_config = yaml.safe_load((REPO_ROOT / "configs" / "policies" / "naive.yaml").read_text())
    append_config = yaml.safe_load(
        (REPO_ROOT / "configs" / "policies" / "append_only.yaml").read_text()
    )
    kv_evict_config = yaml.safe_load(
        (REPO_ROOT / "configs" / "policies" / "kv_evict.yaml").read_text()
    )
    hybrid_config = yaml.safe_load(
        (REPO_ROOT / "configs" / "policies" / "hybrid.yaml").read_text()
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)  # type: ignore[no-untyped-call]

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
    threshold_kv_evict = int(kv_evict_config["threshold_pct_of_window"] * config.max_model_len)
    threshold_hybrid = int(hybrid_config["threshold_pct_of_window"] * config.max_model_len)

    results_by_policy: dict[str, list[TrajectoryRunResult]] = {p: [] for p in POLICY_ORDER}
    n_policies = len(POLICY_ORDER)

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
                policies: dict[str, CompactionPolicy] = {
                    "naive": NaivePolicy(
                        tokenizer,
                        summarizer,
                        threshold_tokens=threshold_naive,
                        retire_fraction=naive_config["retire_fraction"],
                    ),
                    "append_only": AppendOnlyPolicy(
                        tokenizer,
                        summarizer,
                        threshold_tokens=threshold_append,
                        retire_fraction=append_config["retire_fraction"],
                    ),
                    "kv_evict": KVEvictPolicy(
                        tokenizer,
                        threshold_tokens=threshold_kv_evict,
                        window_turns=kv_evict_config["window_turns"],
                    ),
                    "hybrid": HybridPolicy(
                        tokenizer,
                        summarizer,
                        threshold_tokens=threshold_hybrid,
                        protect_recent_turns=hybrid_config["protect_recent_turns"],
                        large_tool_output_tokens=hybrid_config["large_tool_output_tokens"],
                        small_turn_tokens=hybrid_config["small_turn_tokens"],
                    ),
                    "none": NoOpPolicy(),
                }
                # Interleaved A/B/C/D/E (spec §0.2, extended to N policies):
                # rotate which policy runs first per trajectory.
                order = POLICY_ORDER[i % n_policies :] + POLICY_ORDER[: i % n_policies]

                print(f"--- {trajectory.trajectory_id} ({len(trajectory)} turns) ---")
                for policy_name in order:
                    result, prev_prompt_ids = run_trajectory(
                        engine=engine,
                        tokenizer=tokenizer,
                        policy=policies[policy_name],
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
                        max_model_len=config.max_model_len,
                    )
                    results_by_policy[policy_name].append(result)
                    exceeded_note = " CONTEXT_EXCEEDED" if result.context_exceeded else ""
                    print(
                        f"  [{policy_name}] steps={result.n_steps} events={result.n_events} "
                        f"total_prefill={result.total_prefill_tokens} "
                        f"event_prefill={result.event_prefill_tokens}{exceeded_note}"
                    )
    finally:
        if clock_lock is not None:
            clock_lock.release()
        gpu_sampler.shutdown()

    steps_frame = step_collector.to_frame()
    step_collector.flush(args.out_dir / "phase4_pareto_cost_steps.parquet")
    event_collector.flush(args.out_dir / "phase4_pareto_cost_events.parquet")

    comparison_id = args.comparison_trajectory
    runs: dict[str, object] = {}
    for policy_name in POLICY_ORDER:
        results = results_by_policy[policy_name]
        idx = next((i for i, r in enumerate(results) if r.trajectory_id == comparison_id), 0)
        mask = steps_frame["policy"] == policy_name
        policy_steps = steps_frame[mask].reset_index(drop=True)
        start = sum(r.n_steps for r in results[:idx])
        end = start + results[idx].n_steps
        run_len = results[idx].n_steps
        runs[policy_name] = policy_steps.iloc[start:end].assign(step_idx=range(run_len))

    out_png = args.out_dir / "phase4_pareto_cost_comparison.png"
    plot_policy_comparison(runs, out_png)  # type: ignore[arg-type]
    print(f"Wrote {out_png}")

    out_summary = args.out_dir / "phase4_pareto_cost_summary.md"
    summary_text = write_pareto_cost_summary(results_by_policy, out_summary)
    print()
    print(summary_text)


if __name__ == "__main__":
    main()
