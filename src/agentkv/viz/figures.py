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


def plot_pareto(
    points: dict[str, dict[str, float]],
    out_path: Path,
    *,
    incomplete: set[str] = frozenset(),  # type: ignore[assignment]
    title: str = "The Pareto frontier: prefill cost vs. task success",
) -> None:
    """AGENTKV_SPEC.md §4.5's Gate 4 figure — "X = cumulative prefill tokens
    (or $/trajectory), Y = task success rate. One point per policy per
    configuration. This is the money figure of the whole project."

    `points` maps policy name -> {"x": median prefill tokens, "x_low":,
    "x_high":, "y": success rate in [0, 1], "y_low":, "y_high":} (IQR for x,
    Wilson CI for y — spec §7's "report ... with error bars" applied to both
    axes). `incomplete` names policies whose `x` does not represent a
    full-trajectory cost (e.g. `none`, whose runs hit `context_exceeded`
    before finishing) — plotted with a hollow/dashed marker and a distinct
    legend entry, never silently mixed in with the complete points spec's
    "you can defend every point on it" bar depends on.
    """
    import matplotlib.pyplot as plt

    colors = ["#dc2626", "#16a34a", "#2563eb", "#9333ea", "#ea580c", "#0891b2", "#65a30d"]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for (policy, p), color in zip(points.items(), colors, strict=False):
        x_err = [[p["x"] - p["x_low"]], [p["x_high"] - p["x"]]]
        y_err = [[p["y"] - p["y_low"]], [p["y_high"] - p["y"]]]
        is_incomplete = policy in incomplete
        ax.errorbar(
            [p["x"]],
            [p["y"]],
            xerr=x_err,
            yerr=y_err,
            fmt="o",
            markersize=11,
            markerfacecolor="white" if is_incomplete else color,
            markeredgecolor=color,
            markeredgewidth=2,
            linestyle="none",
            ecolor=color,
            elinewidth=1.3,
            capsize=4,
            label=f"{policy} (partial run)" if is_incomplete else policy,
        )
        ax.annotate(
            policy,
            (p["x"], p["y"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            color=color,
        )

    ax.set_xlabel("median cumulative prefill tokens per trajectory (IQR error bars)")
    ax.set_ylabel("task success rate (95% Wilson CI error bars)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
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
