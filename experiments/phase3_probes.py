#!/usr/bin/env python3
"""Phase 3.2 — retention probes (AGENTKV_SPEC.md §3.2).

For each trajectory: inject synthetic facts at controlled depths into the
first `--query-step` turns, replay that under both `policies/naive.py` and
`policies/append_only.py` (advancing each policy's own compacted context
exactly as `phase2_layout.py`/`phase3_agreement.py` do), then — once — query
each fact against each policy's final compacted context and record whether
the injected value survived compaction. Producing a "retention-vs-depth
curve per policy" (spec §3.2) is then just grouping `results/phase3_probes.parquet`
by (policy, depth).

Unlike `phase3_agreement.py`, this never touches the raw uncompacted context,
so it doesn't hit `max_model_len` the way that experiment did — a policy's
compacted context stays bounded by its own threshold config by construction.

Usage (inside the WSL venv):
    python experiments/phase3_probes.py --trajectories traj-000 --n-facts 2
    python experiments/phase3_probes.py   # full sweep, all trajectories
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from agentkv.bench.replay import Turn, iter_trajectories  # noqa: E402
from agentkv.bench.tasks.probes import (  # noqa: E402
    ANSWER_JSON_SCHEMA,
    RetentionEventCollector,
    RetentionRecord,
    build_query_prompt_ids,
    check_retention,
    generate_facts,
    inject_facts,
)
from agentkv.context.layout import Tokenizer  # noqa: E402
from agentkv.policies.append_only import AppendOnlyPolicy  # noqa: E402
from agentkv.policies.base import CompactionPolicy  # noqa: E402
from agentkv.policies.naive import LLMSummarizer, NaivePolicy  # noqa: E402
from agentkv.serving.engine import ModelConfig, VLLMEngine  # noqa: E402

DEFAULT_EXCLUDE = {"traj-007"}  # short/interrupted — see results/phase0_summary.md
DEFAULT_INSERTION_STEPS = [10, 30, 50, 70, 90]


@dataclass
class TrajectoryProbeResult:
    trajectory_id: str
    n_facts_queried: int


def replay_to_final_context(
    policy: CompactionPolicy, turns: list[Turn]
) -> list[Turn]:
    context: list[Turn] = []
    for step_idx, turn in enumerate(turns):
        context, _ = policy.maybe_compact(context + [turn], step_idx)
    return context


def run_trajectory_probes(
    *,
    engine: VLLMEngine,
    tokenizer: Tokenizer,
    naive_policy: NaivePolicy,
    append_policy: AppendOnlyPolicy,
    turns: list[Turn],
    trajectory_id: str,
    seed: int,
    fact_seed: int,
    n_facts: int,
    insertion_steps: list[int],
    query_step: int,
    max_answer_tokens: int,
    min_answer_tokens: int,
    event_collector: RetentionEventCollector,
) -> TrajectoryProbeResult:
    base = turns[:query_step]
    facts = generate_facts(seed=fact_seed, n=n_facts)
    combined = inject_facts(base, facts, insertion_steps)
    depths = [query_step - step for step in insertion_steps]

    for policy_name, policy in (("naive", naive_policy), ("append_only", append_policy)):
        final_context = replay_to_final_context(policy, combined)
        for fact, depth in zip(facts, depths, strict=True):
            prompt_ids = build_query_prompt_ids(final_context, fact.question, tokenizer)
            answer = engine.complete_text(
                prompt_ids,
                max_tokens=max_answer_tokens,
                min_tokens=min_answer_tokens,
                guided_json=ANSWER_JSON_SCHEMA,
            )
            event_collector.record(
                RetentionRecord(
                    fact_id=fact.fact_id,
                    fact_kind=fact.kind,
                    policy=policy_name,
                    seed=seed,
                    depth=depth,
                    retained=check_retention(fact, answer),
                    answer_text=answer,
                )
            )

    return TrajectoryProbeResult(trajectory_id=trajectory_id, n_facts_queried=len(facts))


def summarize_retention(event_collector: RetentionEventCollector) -> str:
    frame = event_collector.to_frame()
    if frame.empty:
        return "No facts queried."
    lines = ["policy | depth | n | retention_rate", "---|---|---|---"]
    grouped = frame.groupby(["policy", "depth"])["retained"].agg(["mean", "count"])
    for (policy_name, depth), row in grouped.sort_index().iterrows():
        lines.append(f"{policy_name} | {depth} | {int(row['count'])} | {row['mean']:.3f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories-dir", type=Path, default=REPO_ROOT / "trajectories")
    parser.add_argument("--trajectories", nargs="*", default=None)
    parser.add_argument("--exclude", nargs="*", default=sorted(DEFAULT_EXCLUDE))
    parser.add_argument("--n-facts", type=int, default=len(DEFAULT_INSERTION_STEPS))
    parser.add_argument(
        "--insertion-steps", nargs="*", type=int, default=DEFAULT_INSERTION_STEPS
    )
    parser.add_argument("--query-step", type=int, default=100)
    parser.add_argument("--max-answer-tokens", type=int, default=40)
    parser.add_argument("--min-answer-tokens", type=int, default=3)
    parser.add_argument("--use-fallback", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--out-name", default="phase3_probes.parquet")
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

    all_trajectories = list(iter_trajectories(args.trajectories_dir))
    excluded = set(args.exclude)
    if args.trajectories is not None:
        wanted = set(args.trajectories)
        trajectories = [t for t in all_trajectories if t.trajectory_id in wanted]
    else:
        trajectories = [t for t in all_trajectories if t.trajectory_id not in excluded]
    trajectories = [t for t in trajectories if len(t) > args.query_step]
    if not trajectories:
        raise SystemExit(
            f"No trajectories with more than {args.query_step} turns were selected."
        )
    print(f"Running {len(trajectories)} trajectories: {[t.trajectory_id for t in trajectories]}")

    threshold_naive = int(naive_config["threshold_pct_of_window"] * config.max_model_len)
    threshold_append = int(append_config["threshold_pct_of_window"] * config.max_model_len)
    event_collector = RetentionEventCollector()

    with VLLMEngine(config, startup_timeout_s=600.0) as engine:
        print(f"KV cache capacity: {engine.kv_cache_capacity_tokens()} tokens")
        summarizer = LLMSummarizer(
            engine, tokenizer, max_summary_tokens=naive_config["max_summary_tokens"]
        )
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
            print(f"--- {trajectory.trajectory_id} ({len(trajectory)} turns) ---")
            result = run_trajectory_probes(
                engine=engine,
                tokenizer=tokenizer,
                naive_policy=naive_policy,
                append_policy=append_policy,
                turns=trajectory.turns,
                trajectory_id=trajectory.trajectory_id,
                seed=config.seed,
                fact_seed=config.seed * 1000 + i,
                n_facts=args.n_facts,
                insertion_steps=args.insertion_steps,
                query_step=args.query_step,
                max_answer_tokens=args.max_answer_tokens,
                min_answer_tokens=args.min_answer_tokens,
                event_collector=event_collector,
            )
            print(f"  facts_queried={result.n_facts_queried}")

    summary_text = summarize_retention(event_collector)
    out_path = args.out_dir / args.out_name
    event_collector.flush(out_path)
    print(f"Wrote {out_path}")
    print()
    print(summary_text)


if __name__ == "__main__":
    main()
