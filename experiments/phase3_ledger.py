#!/usr/bin/env python3
"""Phase 3.3 — synthetic verifiable ledger task (AGENTKV_SPEC.md §3.3).

Unlike every other experiment in this project, this one is a LIVE agent
loop, not a replay: there is no recorded trajectory. At each step the
measurement model picks its next tool call (via `guided_json`, same
mechanism `bench/agreement.py` uses for 3.1, but enum-restricted to
`LedgerEnv`'s real tools — see `LedgerEnv.action_schema`), the environment
executes it for real and returns a result, and the growing raw turn history
is fed through a compaction policy (`naive` or `append_only`) exactly like
every other experiment here. An episode ends when the model calls
`submit_final_report` or a tool-call budget is exhausted, and is scored by
`LedgerEnv.verify()` — spec §3.3's binary reward.

Turn-append discipline matters here more than elsewhere: `AppendOnlyPolicy`
treats `turns[-1]` as "the one new raw turn since last call" (see its module
docstring), so each of the two new raw turns a single tool call produces
(the assistant's tool call, then the tool's result) gets its own
`maybe_compact` call — appending both at once would silently drop the
assistant turn from `AppendOnlyPolicy`'s internal state.

Usage (inside the WSL venv):
    python experiments/phase3_ledger.py --seeds 0 --policies naive \
        --n-transactions 5 --max-tool-calls 15
    python experiments/phase3_ledger.py   # full sweep, default seeds/policies/size
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
from agentkv.bench.tasks.ledger import LedgerEnv, LedgerRunCollector, LedgerRunRecord  # noqa: E402
from agentkv.context.layout import Tokenizer, render_turns_to_token_ids  # noqa: E402
from agentkv.policies.append_only import AppendOnlyPolicy  # noqa: E402
from agentkv.policies.base import CompactionPolicy  # noqa: E402
from agentkv.policies.naive import LLMSummarizer, NaivePolicy  # noqa: E402
from agentkv.serving.engine import ModelConfig, VLLMEngine  # noqa: E402

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_POLICIES = ["naive", "append_only"]


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
    anchor = Turn(role="system", content=env.system_prompt())
    context_turns = _advance(policy, [], anchor, 0)
    step_idx = 1
    action_schema = env.action_schema()
    stopped_reason = "step_budget_exhausted"
    n_tool_calls = 0

    # Bootstrap: perform `n_bootstrap_cycles` full real cycles (fetch +
    # matching apply) ourselves before handing control to the model.
    # Verified empirically that with zero in-context precedent the model
    # just picks submit_final_report immediately with garbage args,
    # regardless of what the system prompt says to do — the same class of
    # cold-start failure `bench/agreement.py` and `bench/tasks/probes.py`
    # hit, fixed there with a synthetic exemplar. A *single* bootstrapped
    # cycle wasn't enough either (also verified empirically): the model
    # completed the one demonstrated cycle correctly, then jumped straight
    # to submit_final_report instead of continuing the pattern — one
    # example doesn't reliably convey "keep repeating this," two does.
    # Real worked examples are available for free here, so use those
    # instead of fake exemplar turns.
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
    lines = ["policy | n | success_rate | mean_tool_calls | stopped_reasons", "---|---|---|---|---"]
    for policy_name, group in frame.groupby("policy"):
        reasons = group["stopped_reason"].value_counts().to_dict()
        lines.append(
            f"{policy_name} | {len(group)} | {group['success'].mean():.3f} | "
            f"{group['n_tool_calls'].mean():.1f} | {reasons}"
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
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--out-name", default="phase3_ledger.parquet")
    args = parser.parse_args()

    config = ModelConfig.from_yaml(
        REPO_ROOT / "configs" / "model.yaml", use_fallback=args.use_fallback
    )
    naive_config = yaml.safe_load((REPO_ROOT / "configs" / "policies" / "naive.yaml").read_text())
    append_config = yaml.safe_load(
        (REPO_ROOT / "configs" / "policies" / "append_only.yaml").read_text()
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)  # type: ignore[no-untyped-call]

    threshold_naive = int(naive_config["threshold_pct_of_window"] * config.max_model_len)
    threshold_append = int(append_config["threshold_pct_of_window"] * config.max_model_len)
    event_collector = LedgerRunCollector()

    print(
        f"Running {len(args.seeds)} seeds x {len(args.policies)} policies "
        f"({args.n_transactions} transactions, {args.max_tool_calls} tool-call budget)"
    )

    with VLLMEngine(config, startup_timeout_s=600.0) as engine:
        print(f"KV cache capacity: {engine.kv_cache_capacity_tokens()} tokens")
        summarizer = LLMSummarizer(
            engine, tokenizer, max_summary_tokens=naive_config["max_summary_tokens"]
        )
        for seed in args.seeds:
            for policy_name in args.policies:
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
