#!/usr/bin/env python3
"""Phase 4.5 — the Pareto plot itself (AGENTKV_SPEC.md §4.5, Gate 4).

Combines the two halves measured separately:

- X axis (cumulative prefill tokens): `phase4_pareto_cost.py`'s 14-trajectory
  replay sweep, `results/phase4/phase4_pareto_cost_steps.parquet`. Per-
  trajectory totals are reconstructed directly from that committed parquet
  (never hand-edited, spec §7) by detecting `step_idx == 0` boundaries — each
  policy's rows are one trajectory's steps followed by the next, back to
  back, and a fresh trajectory always restarts `step_idx` at 0.
- Y axis (task success rate): `phase4_pareto_success.py`'s 25-episode live
  ledger sweep, `results/phase4/phase4_pareto_success.parquet`.

No GPU/vLLM needed here — this is pure post-processing over already-measured,
committed data, run separately from either sweep.

Usage (inside the WSL venv, no server required):
    python experiments/phase4_pareto_plot.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from agentkv.bench.stats import wilson_confidence_interval  # noqa: E402
from agentkv.viz.figures import plot_pareto  # noqa: E402

POLICY_ORDER = ["naive", "append_only", "kv_evict", "hybrid", "none"]
INCOMPLETE_POLICIES = {"none"}  # see phase4_pareto_cost_summary.md — hits context_exceeded 14/14


def per_trajectory_totals(steps: pd.DataFrame, policy: str) -> list[float]:
    rows = steps[steps["policy"] == policy].reset_index(drop=True)
    if rows.empty:
        return []
    totals: list[float] = []
    current = 0.0
    for i in range(len(rows)):
        if int(rows.loc[i, "step_idx"]) == 0 and i != 0:  # type: ignore[arg-type]
            totals.append(current)
            current = 0.0
        current += float(rows.loc[i, "prefill_tokens"])  # type: ignore[arg-type]
    totals.append(current)
    return totals


def iqr(values: list[float]) -> tuple[float, float, float]:
    if len(values) < 2:
        return values[0], values[0], values[0]
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return statistics.median(values), q1, q3


def main() -> None:
    out_dir = REPO_ROOT / "results" / "phase4"
    steps = pd.read_parquet(out_dir / "phase4_pareto_cost_steps.parquet")
    success = pd.read_parquet(out_dir / "phase4_pareto_success.parquet")

    points: dict[str, dict[str, float]] = {}
    header = (
        "| policy | n_traj | prefill median | prefill IQR | successes | success rate | "
        "95% Wilson CI |"
    )
    lines = [header, "|---|---|---|---|---|---|---|"]
    for policy in POLICY_ORDER:
        totals = per_trajectory_totals(steps, policy)
        med, q1, q3 = iqr(totals)

        s = success[success["policy"] == policy]
        n_episodes = len(s)
        n_successes = int(s["success"].sum())
        ci = wilson_confidence_interval(n_successes, n_episodes)

        points[policy] = {
            "x": med,
            "x_low": q1,
            "x_high": q3,
            "y": ci.proportion,
            "y_low": ci.low,
            "y_high": ci.high,
        }
        lines.append(
            f"| {policy} | {len(totals)} | {med:.0f} | [{q1:.0f}, {q3:.0f}] | "
            f"{n_successes}/{n_episodes} | {ci.proportion:.3f} | [{ci.low:.3f}, {ci.high:.3f}] |"
        )

    out_png = out_dir / "phase4_pareto.png"
    plot_pareto(points, out_png, incomplete=INCOMPLETE_POLICIES)
    print(f"Wrote {out_png}")
    print()
    table = "\n".join(lines)
    print(table)

    out_table = out_dir / "phase4_pareto_table.md"
    out_table.write_text(table + "\n", encoding="utf-8")
    print(f"Wrote {out_table}")


if __name__ == "__main__":
    main()
