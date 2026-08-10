#!/usr/bin/env python3
"""Phase 4.5 — task-success half of the Pareto plot (AGENTKV_SPEC.md §4.5).

Extends `phase3_ledger.py`'s naive/append_only live-episode sweep to all 5
policies this project has real prefill-cost data for (`kv_evict`, `hybrid`,
and `none` — `policies/none.py`'s never-compact baseline — added; see
`policies/none.py`'s docstring and `results/phase4/phase4_pareto_summary.md`
for why `attn_evict`/`offload` aren't in this set). Same live agent loop,
same `LedgerEnv` (§3.3), same binary-reward `verify()`.

This measures the Y-axis (task success rate) of Gate 4's Pareto plot on a
*different* episode set than `phase4_pareto_cost.py`'s X-axis (recorded
trajectories, replayed): the ledger task has no recorded trajectory to
replay — it's driven live, one tool call at a time, against the real
environment (see `bench/tasks/ledger.py`'s module docstring). Spec's Gate 4
only asks for "one point per policy per configuration," not that both axes
come from literally the same episode, and mixing the replay harness's rigor
scaffolding (clock lock, theory check, interleaving) with a live decision
loop that needs `guided_json`-elicited real actions would fight the two
harnesses' different purposes rather than combine them cleanly.

Phase 3.3 already established (`results/phase3/phase3_ledger_summary.md`)
that naive/append_only both score 0/5 on this task, on both the fallback and
primary models — a floor effect, not a policy effect, since both retire
turns through the identical `LLMSummarizer`. Extending to kv_evict/hybrid/
none is a real, open empirical question worth measuring regardless: hybrid's
keep-verbatim route in particular protects exactly the kind of short,
numeric transaction-result turns this task's ground truth depends on, which
naive/append_only's blanket summarization does not.

Usage (inside the WSL venv):
    python experiments/phase4_pareto_success.py --seeds 0 --policies none \
        --n-transactions 5 --max-tool-calls 15
    python experiments/phase4_pareto_success.py   # full sweep, default seeds/policies/size
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from agentkv.bench.agreement import parse_action  # noqa: E402
from agentkv.bench.replay import Turn  # noqa: E402
from agentkv.bench.stats import wilson_confidence_interval  # noqa: E402
from agentkv.bench.tasks.ledger import LedgerEnv, LedgerRunCollector, LedgerRunRecord  # noqa: E402
from agentkv.context.layout import Tokenizer, render_turns_to_token_ids  # noqa: E402
from agentkv.policies.append_only import AppendOnlyPolicy  # noqa: E402
from agentkv.policies.base import CompactionPolicy  # noqa: E402
from agentkv.policies.hybrid import HybridPolicy  # noqa: E402
from agentkv.policies.kv_evict import KVEvictPolicy  # noqa: E402
from agentkv.policies.naive import LLMSummarizer, NaivePolicy  # noqa: E402
from agentkv.policies.none import NoOpPolicy  # noqa: E402
from agentkv.serving.engine import ModelConfig, VLLMEngine  # noqa: E402

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_POLICIES = ["naive", "append_only", "kv_evict", "hybrid", "none"]


def _advance(
    policy: CompactionPolicy, context_turns: list[Turn], new_turn: Turn, step_idx: int
) -> list[Turn]:
    context_turns, _ = policy.maybe_compact(context_turns + [new_turn], step_idx)
    return context_turns


def _bootstrap_apply(env: LedgerEnv, instruction: str) -> tuple[str, dict[str, object], str]:
    """Scripted, deterministic parse+execute of one instruction — mirrors
    exactly the text format `LedgerEnv._generate_transactions` produces.
    Used only to bootstrap the very first cycle (see `run_ledger_episode`)."""
    if instruction.startswith("Deposit"):
        _, amount, _, account = instruction.rstrip(".").split(" ", 3)
        args: dict[str, object] = {"account": account, "amount": int(amount)}
        return "deposit", args, env.deposit(account, int(amount))
    if instruction.startswith("Withdraw"):
        _, amount, _, account = instruction.rstrip(".").split(" ", 3)
        args = {"account": account, "amount": int(amount)}
        return "withdraw", args, env.withdraw(account, int(amount))
    if instruction.startswith("Transfer"):
        parts = instruction.rstrip(".").split(" ")
        amount, from_account, to_account = int(parts[1]), parts[3], parts[5]
        args = {"from_account": from_account, "to_account": to_account, "amount": amount}
        return "transfer", args, env.transfer(from_account, to_account, amount)
    raise ValueError(f"Cannot bootstrap-parse instruction: {instruction!r}")


def run_ledger_episode(
    *,
    engine: VLLMEngine,
    tokenizer: Tokenizer,
    policy: CompactionPolicy,
    policy_name: str,
    env: LedgerEnv,
    seed: int,
    n_transactions: int,
    max_tool_calls: int,
    max_action_tokens: int,
    max_model_len: int,
    n_bootstrap_cycles: int = 2,
) -> LedgerRunRecord:
    """Same shape as `phase3_ledger.py`'s `run_ledger_episode` — duplicated
    rather than imported, matching this project's convention of
    self-contained experiment scripts."""
    anchor = Turn(role="system", content=env.system_prompt())
    context_turns = _advance(policy, [], anchor, 0)
    step_idx = 1
    action_schema = env.action_schema()
    stopped_reason = "step_budget_exhausted"
    n_tool_calls = 0

    for _ in range(n_bootstrap_cycles):
        if env.instructions_remaining == 0:
            break
        instruction = env.get_next_instruction()
        fetch_call = Turn(
            role="assistant", tool_calls=[{"name": "get_next_instruction", "args": {}}]
        )
        context_turns = _advance(policy, context_turns, fetch_call, step_idx)
        step_idx += 1
        fetch_result = Turn(
            role="tool", tool_results=[{"name": "get_next_instruction", "output": instruction}]
        )
        context_turns = _advance(policy, context_turns, fetch_result, step_idx)
        step_idx += 1
        n_tool_calls += 1

        tool_name, tool_args, result_text = _bootstrap_apply(env, instruction)
        apply_call = Turn(role="assistant", tool_calls=[{"name": tool_name, "args": tool_args}])
        context_turns = _advance(policy, context_turns, apply_call, step_idx)
        step_idx += 1
        apply_result = Turn(role="tool", tool_results=[{"name": tool_name, "output": result_text}])
        context_turns = _advance(policy, context_turns, apply_result, step_idx)
        step_idx += 1
        n_tool_calls += 1

    for _ in range(max_tool_calls - n_tool_calls):
        prompt_ids = render_turns_to_token_ids([*context_turns, Turn(role="assistant")], tokenizer)
        if len(prompt_ids) + max_action_tokens > max_model_len:
            stopped_reason = "context_exceeded"
            break

        raw_text = engine.complete_text(
            prompt_ids, max_tokens=max_action_tokens, guided_json=action_schema
        )
        action = parse_action(raw_text)

        if action.parse_ok and action.tool_name is not None:
            tool_name = action.tool_name
            tool_args = action.args or {}
            result_text = env.call(tool_name, tool_args)
            assistant_turn = Turn(
                role="assistant", tool_calls=[{"name": tool_name, "args": tool_args}]
            )
        else:
            # guided_json makes this structurally near-impossible, but stay
            # honest about the possibility rather than assuming it away.
            tool_name = None
            result_text = "Invalid tool call — no action taken."
            assistant_turn = Turn(role="assistant", content=raw_text)

        context_turns = _advance(policy, context_turns, assistant_turn, step_idx)
        step_idx += 1
        tool_turn = Turn(role="tool", tool_results=[{"name": tool_name, "output": result_text}])
        context_turns = _advance(policy, context_turns, tool_turn, step_idx)
        step_idx += 1
        n_tool_calls += 1

        if tool_name == "submit_final_report":
            stopped_reason = "submitted"
            break

    return LedgerRunRecord(
        seed=seed,
        policy=policy_name,
        n_transactions=n_transactions,
        n_tool_calls=n_tool_calls,
        instructions_remaining=env.instructions_remaining,
        stopped_reason=stopped_reason,
        success=env.verify(),
    )


def summarize_ledger(event_collector: LedgerRunCollector) -> str:
    frame = event_collector.to_frame()
    if frame.empty:
        return "No episodes run."
    lines = [
        "policy | n | successes | success_rate | 95% Wilson CI | mean_tool_calls | stopped_reasons",
        "---|---|---|---|---|---|---",
    ]
    for policy_name, group in frame.groupby("policy"):
        successes = int(group["success"].sum())
        n = len(group)
        ci = wilson_confidence_interval(successes, n)
        reasons = group["stopped_reason"].value_counts().to_dict()
        lines.append(
            f"{policy_name} | {n} | {successes} | {ci.proportion:.3f} | "
            f"[{ci.low:.3f}, {ci.high:.3f}] | {group['n_tool_calls'].mean():.1f} | {reasons}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--policies", nargs="*", default=DEFAULT_POLICIES)
    parser.add_argument("--n-transactions", type=int, default=50)
    parser.add_argument("--n-accounts", type=int, default=5)
    parser.add_argument("--max-tool-calls", type=int, default=120)
    parser.add_argument("--max-action-tokens", type=int, default=100)
    parser.add_argument("--n-bootstrap-cycles", type=int, default=2)
    parser.add_argument("--use-fallback", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "phase4")
    parser.add_argument("--out-name", default="phase4_pareto_success.parquet")
    args = parser.parse_args()

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

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)  # type: ignore[no-untyped-call]

    threshold_naive = int(naive_config["threshold_pct_of_window"] * config.max_model_len)
    threshold_append = int(append_config["threshold_pct_of_window"] * config.max_model_len)
    threshold_kv_evict = int(kv_evict_config["threshold_pct_of_window"] * config.max_model_len)
    threshold_hybrid = int(hybrid_config["threshold_pct_of_window"] * config.max_model_len)
    event_collector = LedgerRunCollector()

    policy_order = list(args.policies)
    n_policies = len(policy_order)
    print(
        f"Running {len(args.seeds)} seeds x {n_policies} policies "
        f"({args.n_transactions} transactions, {args.max_tool_calls} tool-call budget)"
    )

    with VLLMEngine(config, startup_timeout_s=600.0) as engine:
        print(f"KV cache capacity: {engine.kv_cache_capacity_tokens()} tokens")
        summarizer = LLMSummarizer(
            engine, tokenizer, max_summary_tokens=naive_config["max_summary_tokens"]
        )
        for i, seed in enumerate(args.seeds):
            # Interleaved A/B/C/D/E (spec §0.2, extended to N policies):
            # rotate which policy runs first per seed.
            order = policy_order[i % n_policies :] + policy_order[: i % n_policies]
            for policy_name in order:
                env = LedgerEnv(
                    seed=seed, n_transactions=args.n_transactions, n_accounts=args.n_accounts
                )
                if policy_name == "naive":
                    policy: CompactionPolicy = NaivePolicy(
                        tokenizer,
                        summarizer,
                        threshold_tokens=threshold_naive,
                        retire_fraction=naive_config["retire_fraction"],
                    )
                elif policy_name == "append_only":
                    policy = AppendOnlyPolicy(
                        tokenizer,
                        summarizer,
                        threshold_tokens=threshold_append,
                        retire_fraction=append_config["retire_fraction"],
                    )
                elif policy_name == "kv_evict":
                    policy = KVEvictPolicy(
                        tokenizer,
                        threshold_tokens=threshold_kv_evict,
                        window_turns=kv_evict_config["window_turns"],
                    )
                elif policy_name == "hybrid":
                    policy = HybridPolicy(
                        tokenizer,
                        summarizer,
                        threshold_tokens=threshold_hybrid,
                        protect_recent_turns=hybrid_config["protect_recent_turns"],
                        large_tool_output_tokens=hybrid_config["large_tool_output_tokens"],
                        small_turn_tokens=hybrid_config["small_turn_tokens"],
                    )
                elif policy_name == "none":
                    policy = NoOpPolicy()
                else:
                    raise SystemExit(f"Unknown policy: {policy_name}")

                result = run_ledger_episode(
                    engine=engine,
                    tokenizer=tokenizer,
                    policy=policy,
                    policy_name=policy_name,
                    env=env,
                    seed=seed,
                    n_transactions=args.n_transactions,
                    max_tool_calls=args.max_tool_calls,
                    max_action_tokens=args.max_action_tokens,
                    max_model_len=config.max_model_len,
                    n_bootstrap_cycles=args.n_bootstrap_cycles,
                )
                event_collector.record(result)
                print(
                    f"  seed={seed} policy={policy_name} success={result.success} "
                    f"tool_calls={result.n_tool_calls} "
                    f"instructions_remaining={result.instructions_remaining} "
                    f"stopped={result.stopped_reason}"
                )

    out_path = args.out_dir / args.out_name
    summary_text = summarize_ledger(event_collector)
    event_collector.flush(out_path)
    print(f"Wrote {out_path}")
    print()
    print(summary_text)


if __name__ == "__main__":
    main()
