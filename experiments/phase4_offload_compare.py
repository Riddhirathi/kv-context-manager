#!/usr/bin/env python3
"""Phase 4.3 — comparative sweep for the trajectory-aware offload connector
(AGENTKV_SPEC.md §4.3).

Replays the same set of trajectories under `AppendOnlyPolicy` twice, through
two separate engine boots (kv_transfer_config is set at server startup, so it
can't be toggled mid-run): once with `TrajectoryAwareOffloadConnector`
enabled, once without (a clean vLLM boot, native GPU prefix caching only).
Each boot starts from an empty cache, so this is a fair A/B comparison, not
contaminated by leftover state from the other condition.

`append_only` specifically (not naive/kv_evict) because its anchor is
byte-identical across every trajectory in this project — every trajectory's
`synth_env.SyntheticIncidentEnv.system_prompt()` is seed-independent (spec
Phase 0 recording) — so if the GPU's own cache evicts an early trajectory's
anchor by the time a much later trajectory needs it again, that's exactly the
cross-trajectory reuse scenario `kv/offload.py`'s policy is designed to catch
that plain LRU (or no offload tier at all) would lose.

**Metric note, discovered during verification, not assumed**: this project's
existing `StepResult.cached_tokens`/`prefill_tokens` are derived entirely from
vLLM's *native* GPU prefix-cache Prometheus counter
(`vllm:gpu_prefix_cache_hits_total`) — confirmed empirically that this counter
does not move for the offload connector's own external cache hits, since
those are a structurally separate code path (`start_load_kv` injecting
already-computed KV, not a native APC block match). So this script's primary
comparison metric is **wall-clock time** (`ttft_ms + decode_ms`, already
collected), not prefill token counts — the latter would silently under-report
any real benefit from the connector.

Usage (inside the WSL venv):
    python experiments/phase4_offload_compare.py --trajectories traj-000 traj-001 --max-steps 30
    python experiments/phase4_offload_compare.py   # full default subset
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from agentkv.bench.replay import Turn, iter_trajectories  # noqa: E402
from agentkv.context.layout import Tokenizer, render_turns_to_token_ids  # noqa: E402
from agentkv.policies.append_only import AppendOnlyPolicy  # noqa: E402
from agentkv.policies.naive import LLMSummarizer  # noqa: E402
from agentkv.serving.engine import ModelConfig, VLLMEngine  # noqa: E402

DEFAULT_TRAJECTORIES = [
    "traj-000",
    "traj-001",
    "traj-002",
    "traj-003",
    "traj-004",
    "traj-005",
]


def replay_trajectory_wallclock(
    *,
    engine: VLLMEngine,
    tokenizer: Tokenizer,
    policy: AppendOnlyPolicy,
    turns: list[Turn],
    max_steps: int | None,
) -> tuple[float, int]:
    """Returns (total wall-clock ms, step count) — no divergence/theory
    checking here (this script measures timing, not cache-invalidation
    correctness, which Phase 2/4.1 already established for append_only)."""
    context_turns: list[Turn] = []
    total_wallclock = 0.0
    limit = len(turns) if max_steps is None else min(max_steps, len(turns))
    for step_idx in range(limit):
        context_turns, _ = policy.maybe_compact(context_turns + [turns[step_idx]], step_idx)
        prompt_ids = render_turns_to_token_ids(context_turns, tokenizer)
        result = engine.generate_step(prompt_ids, max_tokens=8)
        total_wallclock += result.ttft_ms + result.decode_ms
    return total_wallclock, limit


def run_condition(
    *,
    config: ModelConfig,
    trajectories: list,
    tokenizer: Tokenizer,
    append_config: dict,
    max_steps: int | None,
    kv_transfer_config: dict[str, object] | None,
    label: str,
) -> tuple[float, int]:
    threshold_append = int(append_config["threshold_pct_of_window"] * config.max_model_len)
    total_wallclock = 0.0
    total_steps = 0
    t_wall_start = time.monotonic()
    engine_ctx = VLLMEngine(
        config, startup_timeout_s=420.0, kv_transfer_config=kv_transfer_config
    )
    with engine_ctx as engine:
        print(f"--- [{label}] booted, KV cache capacity: {engine.kv_cache_capacity_tokens()} ---")
        summarizer = LLMSummarizer(
            engine, tokenizer, max_summary_tokens=append_config["max_summary_tokens"]
        )
        for trajectory in trajectories:
            policy = AppendOnlyPolicy(
                tokenizer,
                summarizer,
                threshold_tokens=threshold_append,
                retire_fraction=append_config["retire_fraction"],
            )
            traj_wallclock, n_steps = replay_trajectory_wallclock(
                engine=engine,
                tokenizer=tokenizer,
                policy=policy,
                turns=trajectory.turns,
                max_steps=max_steps,
            )
            total_wallclock += traj_wallclock
            total_steps += n_steps
            print(
                f"  [{label}] {trajectory.trajectory_id}: steps={n_steps} "
                f"wallclock_ms={traj_wallclock:.1f}"
            )
    real_elapsed_s = time.monotonic() - t_wall_start
    print(
        f"[{label}] TOTAL: steps={total_steps} summed_wallclock_ms={total_wallclock:.1f} "
        f"real_elapsed_s={real_elapsed_s:.1f}"
    )
    return total_wallclock, total_steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories-dir", type=Path, default=REPO_ROOT / "trajectories")
    parser.add_argument("--trajectories", nargs="*", default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--use-fallback", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    config = ModelConfig.from_yaml(
        REPO_ROOT / "configs" / "model.yaml", use_fallback=args.use_fallback
    )
    append_config = yaml.safe_load(
        (REPO_ROOT / "configs" / "policies" / "append_only.yaml").read_text()
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)  # type: ignore[no-untyped-call]

    all_trajectories = list(iter_trajectories(args.trajectories_dir))
    wanted = set(args.trajectories)
    trajectories = [t for t in all_trajectories if t.trajectory_id in wanted]
    if not trajectories:
        raise SystemExit("No trajectories selected.")
    print(f"Comparing {len(trajectories)} trajectories: {[t.trajectory_id for t in trajectories]}")

    baseline_wallclock, baseline_steps = run_condition(
        config=config,
        trajectories=trajectories,
        tokenizer=tokenizer,
        append_config=append_config,
        max_steps=args.max_steps,
        kv_transfer_config=None,
        label="baseline",
    )

    offload_kv_transfer_config = {
        "kv_connector": "TrajectoryAwareOffloadConnector",
        "kv_role": "kv_both",
    }
    offload_wallclock, offload_steps = run_condition(
        config=config,
        trajectories=trajectories,
        tokenizer=tokenizer,
        append_config=append_config,
        max_steps=args.max_steps,
        kv_transfer_config=offload_kv_transfer_config,
        label="offload",
    )

    print()
    print("=== SUMMARY ===")
    print(f"baseline: steps={baseline_steps} summed_wallclock_ms={baseline_wallclock:.1f}")
    print(f"offload:  steps={offload_steps} summed_wallclock_ms={offload_wallclock:.1f}")
    if baseline_wallclock > 0:
        pct = 100.0 * (baseline_wallclock - offload_wallclock) / baseline_wallclock
        print(f"wall-clock change (offload vs baseline): {pct:+.1f}%")


if __name__ == "__main__":
    main()
