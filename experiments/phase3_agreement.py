#!/usr/bin/env python3
"""Phase 3.1 — next-action agreement (AGENTKV_SPEC.md §3.1).

Replays a trajectory under both `policies/naive.py` and `policies/append_only.py`
exactly as `experiments/phase2_layout.py` does, but at every decision point
(a recorded assistant turn that made a tool call) also elicits a fresh
counterfactual action from the *measurement* model under (a) the raw,
never-compacted context and (b) each policy's actual compacted context at
that point, and scores how often they agree (`bench/agreement.py`).

This intentionally does not touch cache-hit/prefill accounting — that's
Phase 2's job and is already measured. Every call here goes through
`VLLMEngine.complete_text`, the same method `LLMSummarizer` uses, so summary
generation and action elicitation share one accounting-free code path.

Usage (inside the WSL venv):
    python experiments/phase3_agreement.py --trajectories traj-000 --max-steps 30
    python experiments/phase3_agreement.py   # full sweep, all trajectories
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from agentkv.bench.agreement import (  # noqa: E402
    ACTION_JSON_SCHEMA,
    AgreementEventCollector,
    build_decision_prompt_ids,
    is_decision_point,
    parse_action,
    score_agreement,
)
from agentkv.bench.replay import Turn, iter_trajectories  # noqa: E402
from agentkv.context.layout import Tokenizer  # noqa: E402
from agentkv.policies.append_only import AppendOnlyPolicy  # noqa: E402
from agentkv.policies.naive import LLMSummarizer, NaivePolicy  # noqa: E402
from agentkv.serving.engine import ModelConfig, VLLMEngine  # noqa: E402

DEFAULT_EXCLUDE = {"traj-007"}  # short/interrupted — see results/phase0_summary.md


@dataclass
class TrajectoryAgreementResult:
    trajectory_id: str
    n_decision_points: int
    n_full_parse_failures: int
    n_naive_parse_failures: int
    n_append_parse_failures: int
    # Step at which the raw, never-compacted "full" context first would have
    # exceeded max_model_len (None if it never did over this run). Once this
    # happens, "the model's next action given full uncompacted context"
    # (spec §3.1) is no longer even askable — full context only ever grows,
    # so every later decision point in this trajectory has the same problem.
    # This is Phase 0/1's cliff arriving inside a different experiment, not a
    # bug: naive/append_only's *compacted* contexts stay bounded by
    # construction, so their own actions remain fully measurable regardless
    # — only the (a)-vs-(b) agreement comparison becomes unavailable past
    # this point.
    context_exceeded_at_step: int | None


def run_trajectory_agreement(
    *,
    engine: VLLMEngine,
    tokenizer: Tokenizer,
    naive_policy: NaivePolicy,
    append_policy: AppendOnlyPolicy,
    turns: list[Turn],
    trajectory_id: str,
    seed: int,
    max_action_tokens: int,
    max_model_len: int,
    event_collector: AgreementEventCollector,
    max_steps: int | None,
    max_decision_points: int | None,
) -> TrajectoryAgreementResult:
    naive_context: list[Turn] = []
    append_context: list[Turn] = []
    n_decision_points = 0
    n_full_failures = 0
    n_naive_failures = 0
    n_append_failures = 0
    context_exceeded_at_step: int | None = None

    limit = len(turns) if max_steps is None else min(max_steps, len(turns))
    for step_idx in range(limit):
        turn = turns[step_idx]

        if is_decision_point(turn) and (
            max_decision_points is None or n_decision_points < max_decision_points
        ):
            full_prompt_ids = build_decision_prompt_ids(turns[:step_idx], tokenizer)
            if len(full_prompt_ids) + max_action_tokens > max_model_len:
                context_exceeded_at_step = step_idx
                break
            full_text = engine.complete_text(
                full_prompt_ids, max_tokens=max_action_tokens, guided_json=ACTION_JSON_SCHEMA
            )
            full_action = parse_action(full_text)
            n_full_failures += int(not full_action.parse_ok)

            for policy_name, context_before in (
                ("naive", naive_context),
                ("append_only", append_context),
            ):
                compacted_prompt_ids = build_decision_prompt_ids(context_before, tokenizer)
                compacted_text = engine.complete_text(
                    compacted_prompt_ids,
                    max_tokens=max_action_tokens,
                    guided_json=ACTION_JSON_SCHEMA,
                )
                compacted_action = parse_action(compacted_text)
                if not compacted_action.parse_ok:
                    if policy_name == "naive":
                        n_naive_failures += 1
                    else:
                        n_append_failures += 1
                event_collector.record(
                    score_agreement(
                        step_idx=step_idx,
                        policy=policy_name,
                        seed=seed,
                        full_action=full_action,
                        compacted_action=compacted_action,
                    )
                )
            n_decision_points += 1

        naive_context, _ = naive_policy.maybe_compact(naive_context + [turn], step_idx)
        append_context, _ = append_policy.maybe_compact(append_context + [turn], step_idx)

    return TrajectoryAgreementResult(
        trajectory_id=trajectory_id,
        n_decision_points=n_decision_points,
        n_full_parse_failures=n_full_failures,
        n_naive_parse_failures=n_naive_failures,
        n_append_parse_failures=n_append_failures,
        context_exceeded_at_step=context_exceeded_at_step,
    )


def summarize_agreement(event_collector: AgreementEventCollector) -> str:
    frame = event_collector.to_frame()
    if frame.empty:
        return "No decision points evaluated."
    lines = [
        "policy | n | tool_name_match | args_match | mean_semantic_similarity",
        "---|---|---|---|---",
    ]
    for policy_name, group in frame.groupby("policy"):
        lines.append(
            f"{policy_name} | {len(group)} | "
            f"{group['tool_name_match'].mean():.3f} | "
            f"{group['args_match'].mean():.3f} | "
            f"{group['semantic_similarity'].mean():.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories-dir", type=Path, default=REPO_ROOT / "trajectories")
    parser.add_argument("--trajectories", nargs="*", default=None)
    parser.add_argument("--exclude", nargs="*", default=sorted(DEFAULT_EXCLUDE))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-decision-points", type=int, default=None)
    parser.add_argument("--max-action-tokens", type=int, default=200)
    parser.add_argument("--use-fallback", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--out-name", default="phase3_agreement.parquet")
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
    if not trajectories:
        raise SystemExit("No trajectories selected.")
    print(f"Running {len(trajectories)} trajectories: {[t.trajectory_id for t in trajectories]}")

    threshold_naive = int(naive_config["threshold_pct_of_window"] * config.max_model_len)
    threshold_append = int(append_config["threshold_pct_of_window"] * config.max_model_len)
    event_collector = AgreementEventCollector()

    trajectory_results: list[TrajectoryAgreementResult] = []
    with VLLMEngine(config, startup_timeout_s=600.0) as engine:
        print(f"KV cache capacity: {engine.kv_cache_capacity_tokens()} tokens")
        summarizer = LLMSummarizer(
            engine, tokenizer, max_summary_tokens=naive_config["max_summary_tokens"]
        )
        for trajectory in trajectories:
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
            result = run_trajectory_agreement(
                engine=engine,
                tokenizer=tokenizer,
                naive_policy=naive_policy,
                append_policy=append_policy,
                turns=trajectory.turns,
                trajectory_id=trajectory.trajectory_id,
                seed=config.seed,
                max_action_tokens=args.max_action_tokens,
                max_model_len=config.max_model_len,
                event_collector=event_collector,
                max_steps=args.max_steps,
                max_decision_points=args.max_decision_points,
            )
            print(
                f"  decision_points={result.n_decision_points} "
                f"full_parse_failures={result.n_full_parse_failures} "
                f"naive_parse_failures={result.n_naive_parse_failures} "
                f"append_parse_failures={result.n_append_parse_failures} "
                f"context_exceeded_at_step={result.context_exceeded_at_step}"
            )
            trajectory_results.append(result)

    n_truncated = sum(1 for r in trajectory_results if r.context_exceeded_at_step is not None)
    if n_truncated:
        print(
            f"\nNOTE: {n_truncated}/{len(trajectory_results)} trajectories hit the "
            f"max_model_len={config.max_model_len} ceiling on the raw uncompacted context "
            "before finishing — decision points beyond that were NOT evaluated for those "
            "trajectories (see per-trajectory context_exceeded_at_step above)."
        )

    summary_text = summarize_agreement(event_collector)
    out_path = args.out_dir / args.out_name
    event_collector.flush(out_path)
    print(f"Wrote {out_path}")
    print()
    print(summary_text)


if __name__ == "__main__":
    main()
