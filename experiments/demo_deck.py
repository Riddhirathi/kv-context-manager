#!/usr/bin/env python3
"""Demo design §6.2 — the three-figure deck (AGENTKV_SPEC.md §6.2):

    1. The cliff.  Cumulative prefill vs. step, naive policy.
       "Here is a cost nobody is measuring."
    2. The fix.    Same axes, all [compaction] policies overlaid.
       "Here is the cost removed."
    3. The Pareto frontier.  Cost vs. task success.
       "Here is proof I didn't cheat."

Pure post-processing, no GPU/vLLM needed — assembles already-committed data
(spec §7: never hand-edited) into one self-contained `results/demo_deck/`
folder for the interview pitch.

Panels 1 and 2 are regenerated here (not reused from `results/phase1/
phase1_cliff.png`) rather than copied, because Phase 1 predates this
project's `--use-fallback` convention (README's Environment section:
Phase 2 introduced running on the fallback model after the primary's KV
headroom proved too tight) — reusing phase1_cliff.png as-is would silently
put a different model on panel 1 than panels 2 and 3 both use. Both are
sliced from `results/phase4/phase4_pareto_cost_steps.parquet` (fallback
model, `--comparison-trajectory traj-000` by convention — see
`viz/dashboard.py`'s module docstring for why the same run-index means the
same underlying trajectory across policies), so all three panels are the
same model, and panels 1-2 are the same trajectory.

Panel 2 overlays the 4 real `CompactionPolicy` implementations (naive,
append_only, kv_evict, hybrid) — not `none`. Spec's "all policies overlaid"
predates `none`, which Phase 4.5 added specifically as Gate 4's 5th Pareto
point (a ceiling reference, not a compaction strategy competing on "cost
removed" — see `results/phase4/phase4_pareto_summary.md`'s "The 5th policy").
Including a policy whose line stops early from a context-length crash, not a
finished trajectory, would read as a 5th contender in a race it was never
entered in.

Panel 3 is copied verbatim from `results/phase4/phase4_pareto.png` — it's
already the canonical, already-defended Gate 4 figure, and aggregates across
all 14 trajectories rather than one, so it can't be "sliced" the way panels
1-2 are.

Usage (no GPU/vLLM needed):
    python experiments/demo_deck.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from agentkv.viz.dashboard import invalidated_block_count, split_into_runs  # noqa: E402
from agentkv.viz.figures import plot_cliff, plot_policy_comparison  # noqa: E402

FIX_POLICIES = ["naive", "append_only", "kv_evict", "hybrid"]


def fired_events_frame(run: pd.DataFrame, block_size: int) -> pd.DataFrame:
    """A minimal `events`-shaped frame (`plot_cliff` only reads `step_idx`)
    for the steps in `run` where a compaction actually invalidated cache —
    recomputed from `prompt_tokens`/`cached_tokens` rather than joined
    against the events parquet, which has no run-boundary markers to
    disambiguate one trajectory's events from another's (see
    `viz.dashboard.invalidated_block_count`'s docstring)."""
    fired_steps: list[int] = []
    prev_prompt_tokens = 0
    for row in run.itertuples():
        invalidated = invalidated_block_count(
            prev_prompt_tokens=prev_prompt_tokens,
            cached_tokens=int(row.cached_tokens),
            block_size=block_size,
        )
        if invalidated > 0:
            fired_steps.append(int(row.step_idx))
        prev_prompt_tokens = int(row.prompt_tokens)
    return pd.DataFrame({"step_idx": fired_steps})


def main() -> None:
    out_dir = REPO_ROOT / "results" / "demo_deck"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_config = yaml.safe_load((REPO_ROOT / "configs" / "model.yaml").read_text())
    block_size = model_config["engine"]["block_size"]

    phase4_dir = REPO_ROOT / "results" / "phase4"
    steps = pd.read_parquet(phase4_dir / "phase4_pareto_cost_steps.parquet")

    naive_run = split_into_runs(steps, "naive")[0]
    events = fired_events_frame(naive_run, block_size)
    cliff_path = out_dir / "deck_1_cliff.png"
    plot_cliff(
        naive_run,
        events,
        cliff_path,
        title='"Here is a cost nobody is measuring." — naive compaction, one trajectory',
    )
    print(f"Wrote {cliff_path}")

    runs = {policy: split_into_runs(steps, policy)[0] for policy in FIX_POLICIES}
    fix_path = out_dir / "deck_2_fix.png"
    plot_policy_comparison(
        runs,
        fix_path,
        title='"Here is the cost removed." — same trajectory, all compaction policies',
    )
    print(f"Wrote {fix_path}")

    pareto_src = phase4_dir / "phase4_pareto.png"
    pareto_dst = out_dir / "deck_3_pareto.png"
    shutil.copyfile(pareto_src, pareto_dst)
    print(f"Copied {pareto_src} -> {pareto_dst}")


if __name__ == "__main__":
    main()
