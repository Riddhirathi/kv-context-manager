#!/usr/bin/env python3
"""Gate 0 reproducibility check (AGENTKV_SPEC.md §0.4 gate).

"Running the same trajectory twice under the same policy produces TTFT
medians within 5% of each other, and cache hit rate is bit-identical. If you
can't reproduce your own numbers, nothing downstream is meaningful."

Replays ONE recorded trajectory twice, verbatim (no compaction policy exists
yet — that's Phase 1+), against the project's own small measurement model
(configs/model.yaml), and checks:
  1. Per-step cached_tokens sequences are bit-identical across the two runs.
  2. Per-step TTFT medians (spec §0.2: median, never a bare mean) are within
     5% of each other.

The prompt-rendering here (turns -> token ids) is a minimal deterministic
flatten-to-text-then-tokenize — good enough for testing harness
reproducibility, since Gate 0 only cares that the SAME rendering is replayed
identically twice, not that it matches the eventual Phase 2 context layout.

Usage (inside the WSL venv):
    python experiments/gate0_check.py trajectories/traj-000.jsonl
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentkv.bench.replay import Turn, load_trajectory  # noqa: E402
from agentkv.context.layout import render_turns_to_token_ids  # noqa: E402
from agentkv.serving.engine import ModelConfig, VLLMEngine  # noqa: E402

DRIFT_TOLERANCE_PCT = 5.0


def replay_once(
    engine: VLLMEngine, turns_seq: list[list[Turn]], tokenizer: Any
) -> tuple[list[int], list[float]]:
    cached_tokens_seq: list[int] = []
    ttft_seq: list[float] = []
    for context in turns_seq:
        prompt_token_ids = render_turns_to_token_ids(context, tokenizer)
        result = engine.generate_step(prompt_token_ids, max_tokens=8)
        cached_tokens_seq.append(result.cached_tokens)
        ttft_seq.append(result.ttft_ms)
    return cached_tokens_seq, ttft_seq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument(
        "--use-fallback", action="store_true", help="Use the 0.6B fallback model, not 1.7B."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap replay to the first N steps (for a quick check).",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    config = ModelConfig.from_yaml(
        REPO_ROOT / "configs" / "model.yaml", use_fallback=args.use_fallback
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)  # type: ignore[no-untyped-call]

    trajectory = load_trajectory(args.trajectory)
    turns_seq = [trajectory.turns[: i + 1] for i in range(len(trajectory.turns))]
    if args.max_steps is not None:
        turns_seq = turns_seq[: args.max_steps]
    print(f"Loaded {args.trajectory} ({len(trajectory)} turns, replaying {len(turns_seq)} steps)")

    # Two SEPARATE server instances, not one shared engine across both runs —
    # verified this matters: sharing one server's warm prefix cache between
    # "run 1" and "run 2" makes run 2 silently benefit from run 1's cached
    # blocks (observed: run 2's cached_tokens sequence was exactly run 1's,
    # shifted by one step), which defeats the entire point of the gate.
    print("Run 1 (fresh server)...")
    with VLLMEngine(config, startup_timeout_s=600.0) as engine:
        print(f"KV cache capacity: {engine.kv_cache_capacity_tokens()} tokens")
        cached_1, ttft_1 = replay_once(engine, turns_seq, tokenizer)

    print("Run 2 (fresh server)...")
    with VLLMEngine(config, startup_timeout_s=600.0) as engine:
        cached_2, ttft_2 = replay_once(engine, turns_seq, tokenizer)

    identical = cached_1 == cached_2
    median_1 = statistics.median(ttft_1)
    median_2 = statistics.median(ttft_2)
    drift_pct = abs(median_1 - median_2) / median_1 * 100 if median_1 else 0.0

    print()
    print(f"cached_tokens bit-identical across runs: {identical}")
    if not identical:
        mismatches = [
            (i, a, b) for i, (a, b) in enumerate(zip(cached_1, cached_2, strict=True)) if a != b
        ]
        print(f"  first mismatches (step, run1, run2): {mismatches[:10]}")
    print(f"TTFT median run 1: {median_1:.2f}ms, run 2: {median_2:.2f}ms, drift: {drift_pct:.2f}%")

    passed = identical and drift_pct <= DRIFT_TOLERANCE_PCT
    print()
    print("GATE 0: PASS" if passed else "GATE 0: FAIL")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
