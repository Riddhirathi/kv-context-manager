"""Paper figures, generated from committed parquet data only (spec §7: "figures
are generated from committed data, never hand-edited").
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_cliff(
    steps: pd.DataFrame,
    events: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "The cliff: cumulative prefill tokens vs. agent step",
) -> None:
    """AGENTKV_SPEC.md §1.2: "Plot cumulative prefill tokens vs. agent step, for
    a full 100-step trajectory. Expect a staircase: flat-ish growth punctuated
    by vertical jumps at each compaction event. This single figure is the
    entire justification for the project."

    `steps` needs columns `step_idx`, `prefill_tokens` (one row per agent
    step, e.g. metrics.collector.StepRecord's frame). `events` needs
    `step_idx` (one row per fired compaction event, e.g.
    bench.divergence.CompactionEventRecord's frame) — may be empty.
    """
    import matplotlib.pyplot as plt

    ordered = steps.sort_values("step_idx")
    cumulative_prefill = ordered["prefill_tokens"].cumsum()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        ordered["step_idx"],
        cumulative_prefill,
        color="#2563eb",
        linewidth=1.8,
        label="cumulative prefill tokens",
    )

    event_steps = set(events["step_idx"]) if len(events) else set()
    for step_idx in sorted(event_steps):
        ax.axvline(x=step_idx, color="#dc2626", linewidth=1.0, linestyle="--", alpha=0.6)
    if event_steps:
        ax.axvline(x=-1, color="#dc2626", linestyle="--", alpha=0.6, label="compaction event")

    ax.set_xlabel("agent step")
    ax.set_ylabel("cumulative prefill tokens")
    ax.set_title(title)
    ax.set_xlim(ordered["step_idx"].min(), ordered["step_idx"].max())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_policy_comparison(
    runs: dict[str, pd.DataFrame],
    out_path: Path,
    *,
    title: str = "The fix: cumulative prefill tokens vs. agent step, by policy",
) -> None:
    """AGENTKV_SPEC.md §2's Gate 2 figure — the same axes as `plot_cliff`, with
    every policy overlaid on one trajectory, so the reduction in the naive
    policy's staircase is visible directly rather than argued from a table.

    `runs` maps policy name -> that policy's per-step frame (same shape as
    `plot_cliff`'s `steps` argument) for one common trajectory.
    """
    import matplotlib.pyplot as plt

    colors = ["#dc2626", "#16a34a", "#2563eb", "#9333ea", "#ea580c"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for (policy, frame), color in zip(runs.items(), colors, strict=False):
        ordered = frame.sort_values("step_idx")
        ax.plot(
            ordered["step_idx"],
            ordered["prefill_tokens"].cumsum(),
            color=color,
            linewidth=1.8,
            label=policy,
        )

    ax.set_xlabel("agent step")
    ax.set_ylabel("cumulative prefill tokens")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
