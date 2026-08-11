#!/usr/bin/env python3
"""Demo dashboard precompute — naive-vs-hybrid next-action agreement on one
trajectory (AGENTKV_SPEC.md §6.1's "running quality readout: action agreement
so far").

`experiments/phase3_agreement.py` measured naive vs. append_only (Phase 3,
before `hybrid` existed). `viz/dashboard.py` needs naive vs. hybrid on the
same demo trajectory it replays for the context-map strip and prefill
counters (`phase4_pareto_cost_steps.parquet`, `--comparison-trajectory
traj-000` by convention) — hybrid's agreement was never measured anywhere in
the project, so this script fills exactly that gap. Same method as
`phase3_agreement.py` (`bench/agreement.py`'s grammar-constrained elicitation,
scored against a fresh "full uncompacted context" counterfactual), same
append-only parquet contract (spec §7), deliberately scoped to a single
trajectory rather than a full sweep — this exists to feed one demo replay,
not to produce a new statistical claim.

Usage (inside the WSL venv):
    python experiments/phase4_dashboard_agreement.py --use-fallback
    python experiments/phase4_dashboard_agreement.py --use-fallback --trajectory traj-001
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
from agentkv.policies.hybrid import HybridPolicy  # noqa: E402
from agentkv.policies.naive import LLMSummarizer, NaivePolicy  # noqa: E402
from agentkv.serving.engine import ModelConfig, VLLMEngine  # noqa: E402


@dataclass
class TrajectoryAgreementResult:
    trajectory_id: str
    n_decision_points: int
    n_full_parse_failures: int
    n_naive_parse_failures: int
    n_hybrid_parse_failures: int
    context_exceeded_at_step: int | None


def run_trajectory_agreement(
    *,
    engine: VLLMEngine,
    tokenizer: Tokenizer,
    naive_policy: NaivePolicy,
    hybrid_policy: HybridPolicy,
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
    hybrid_context: list[Turn] = []
    n_decision_points = 0
    n_full_failures = 0
    n_naive_failures = 0
    n_hybrid_failures = 0
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
                ("hybrid", hybrid_context),
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
                        n_hybrid_failures += 1
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
        hybrid_context, _ = hybrid_policy.maybe_compact(hybrid_context + [turn], step_idx)

    return TrajectoryAgreementResult(
        trajectory_id=trajectory_id,
        n_decision_points=n_decision_points,
        n_full_parse_failures=n_full_failures,
        n_naive_parse_failures=n_naive_failures,
        n_hybrid_parse_failures=n_hybrid_failures,
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
    parser.add_argument("--trajectory", default="traj-000")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-decision-points", type=int, default=None)
    parser.add_argument("--max-action-tokens", type=int, default=200)
    parser.add_argument("--use-fallback", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "phase4")
    parser.add_argument("--out-name", default="phase4_dashboard_agreement.parquet")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    config = ModelConfig.from_yaml(
        REPO_ROOT / "configs" / "model.yaml", use_fallback=args.use_fallback
    )
    naive_config = yaml.safe_load((REPO_ROOT / "configs" / "policies" / "naive.yaml").read_text())
    hybrid_config = yaml.safe_load(
        (REPO_ROOT / "configs" / "policies" / "hybrid.yaml").read_text()
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)  # type: ignore[no-untyped-call]

    all_trajectories = list(iter_trajectories(args.trajectories_dir))
    trajectory = next((t for t in all_trajectories if t.trajectory_id == args.trajectory), None)
    if trajectory is None:
        raise SystemExit(f"Trajectory {args.trajectory!r} not found in {args.trajectories_dir}")
    print(f"--- {trajectory.trajectory_id} ({len(trajectory)} turns) ---")

    threshold_naive = int(naive_config["threshold_pct_of_window"] * config.max_model_len)
    threshold_hybrid = int(hybrid_config["threshold_pct_of_window"] * config.max_model_len)
    event_collector = AgreementEventCollector()

    with VLLMEngine(config, startup_timeout_s=600.0) as engine:
        print(f"KV cache capacity: {engine.kv_cache_capacity_tokens()} tokens")
        summarizer = LLMSummarizer(
            engine, tokenizer, max_summary_tokens=naive_config["max_summary_tokens"]
        )
        naive_policy = NaivePolicy(
            tokenizer,
            summarizer,
            threshold_tokens=threshold_naive,
            retire_fraction=naive_config["retire_fraction"],
        )
        hybrid_policy = HybridPolicy(
            tokenizer,
            summarizer,
            threshold_tokens=threshold_hybrid,
            protect_recent_turns=hybrid_config["protect_recent_turns"],
            large_tool_output_tokens=hybrid_config["large_tool_output_tokens"],
            small_turn_tokens=hybrid_config["small_turn_tokens"],
        )
        result = run_trajectory_agreement(
            engine=engine,
            tokenizer=tokenizer,
            naive_policy=naive_policy,
            hybrid_policy=hybrid_policy,
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
            f"hybrid_parse_failures={result.n_hybrid_parse_failures} "
            f"context_exceeded_at_step={result.context_exceeded_at_step}"
        )

    if result.context_exceeded_at_step is not None:
        print(
            f"\nNOTE: hit max_model_len={config.max_model_len} on the raw uncompacted context "
            f"at step {result.context_exceeded_at_step} — decision points beyond that were NOT "
            "evaluated (the dashboard should stop its quality readout there too)."
        )

    summary_text = summarize_agreement(event_collector)
    out_path = args.out_dir / args.out_name
    event_collector.flush(out_path)
    print(f"Wrote {out_path}")
    print()
    print(summary_text)


if __name__ == "__main__":
    main()
