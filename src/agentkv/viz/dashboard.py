"""Live side-by-side demo dashboard (AGENTKV_SPEC.md §6.1): "the 90-second
version. Two panes, same trajectory, same model, different policy: naive
(left) vs. hybrid (right) ... this is the visual that sells the entire
project in three seconds."

Replays two already-measured, committed runs rather than driving vLLM live.
Spec's own bar for this artifact is "must run reliably during an interview on
a laptop with no internet" — this project has repeatedly hit real GPU/WSL
fragility during live sweeps (see `results/phase4/phase4_pareto_summary.md`'s
"sustained GPU thermal throttling" episode, and the process stall recorded in
the same phase), which is exactly the risk a live-inference demo would
reintroduce for no benefit: every number this dashboard shows was already
measured once, correctly, with GPU telemetry and the theory-match check
(`bench/divergence.py`) — replaying it is strictly safer and needs no GPU,
model download, or network at demo time (module has no `vllm`/`torch`
import). Renders as a `rich` terminal TUI — `rich` is already a pinned
dependency and no web framework is, so a browser/port-based UI would be
strictly more moving parts for the same "renders fast and reliably" bar
spec explicitly says either approach clears.

Data sources (spec §7: "figures are generated from committed data, never
hand-edited"):
  - `results/phase4/phase4_pareto_cost_steps.parquet` — per-step
    prompt/cached/prefill tokens and TTFT, all 5 Gate-4 policies interleaved
    over the same 14 trajectories in the same order (`experiments/
    phase4_pareto_cost.py`) — naive's and hybrid's Nth trajectory-run are
    therefore guaranteed to be the same underlying trajectory, which is what
    `split_into_runs` + `--run-index` exploit instead of needing an explicit
    trajectory-id column.
  - `results/phase4/phase4_dashboard_agreement.parquet` — naive-vs-hybrid
    next-action agreement on that one demo trajectory
    (`experiments/phase4_dashboard_agreement.py`), a gap this project's
    earlier agreement measurement (`phase3_agreement.py`) never filled since
    it predates `hybrid`. Missing/partial data degrades to an honest "not
    measured" readout rather than fabricating a number.

Context-map-strip color model (spec: "green (cache hit) / red (invalidated
this step) / grey (offloaded to CPU)"), derived per step purely from
`prompt_tokens`/`cached_tokens`/`block_size` — the same measured quantities
`bench/divergence.py`'s theory-match check already validates, so no new
accounting is introduced here:
  - `reused = cached_tokens // block_size` blocks at the front: GREEN (still
    resident from the previous step's cache).
  - `prev_cacheable - reused` blocks right after that: RED — blocks that
    *were* cacheable a step ago but weren't reused now. This is the "invalidated
    this step" segment; its width is naturally large for naive (compacts most
    of the middle away) and small for hybrid (protects everything but a tail),
    which is exactly spec's "goes red from the middle to the end" vs. "reddens
    only the tail" without needing any policy-specific logic.
  - anything beyond that up to the current prompt's own block count: GREEN
    again (freshly computed this step, now resident going forward).
  - GREY (offloaded to CPU) is in the legend/palette for the full 3-color
    contract but never fires for a naive/hybrid comparison — CPU offload
    (§4.3) is an orthogonal serving-layer overlay, not one of the 5
    `CompactionPolicy` objects this dashboard can select between.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from rich.console import Console, Group  # noqa: E402
from rich.layout import Layout  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

GREEN = "green"
RED = "red"
GREY = "grey50"

# Illustrative only (spec's real analytical cost model is Phase 6.1,
# `serving/cost_model.py`, not built yet) — a representative small-hosted-
# model input-token rate, so the $ counter means something during a demo
# without overclaiming a measured figure. Override with --rate-usd-per-1m.
DEFAULT_RATE_USD_PER_1M = 0.15


def invalidated_block_count(
    *, prev_prompt_tokens: int, cached_tokens: int, block_size: int
) -> int:
    """Blocks that were cacheable a step ago but weren't reused now — the same
    quantity `bench.divergence.compute_divergence_check` calls
    `measured_invalidated_blocks`, recomputed here from the committed
    `prompt_tokens`/`cached_tokens` columns alone so callers (this module's
    context-map strip, `experiments/demo_deck.py`'s "did this step compact"
    check) don't need the separate events parquet, which has no per-run
    boundary markers to disambiguate."""
    prev_cacheable = prev_prompt_tokens // block_size
    reused = min(cached_tokens // block_size, prev_cacheable)
    return prev_cacheable - reused


def block_segments(
    *, prev_prompt_tokens: int, prompt_tokens: int, cached_tokens: int, block_size: int
) -> list[tuple[str, int]]:
    """Ordered (color, block_count) segments for one step's context-map strip.

    See module docstring for the derivation. Always tiles exactly
    `[0, max(current_total_blocks, prev_cacheable_blocks))` with no gaps —
    the three segments are computed from a single split point
    (`prev_cacheable_blocks`) so they can never overlap or leave a hole.
    """
    prev_cacheable = prev_prompt_tokens // block_size
    reused = min(cached_tokens // block_size, prev_cacheable)
    current_total = math.ceil(prompt_tokens / block_size) if prompt_tokens else 0
    invalidated = invalidated_block_count(
        prev_prompt_tokens=prev_prompt_tokens, cached_tokens=cached_tokens, block_size=block_size
    )

    segments: list[tuple[str, int]] = []
    if reused:
        segments.append((GREEN, reused))
    if invalidated:
        segments.append((RED, invalidated))
    if current_total > prev_cacheable:
        segments.append((GREEN, current_total - prev_cacheable))
    return segments


def compress_segments(
    segments: list[tuple[str, int]], max_cells: int
) -> list[tuple[str, int]]:
    """Downsamples `segments` to fit `max_cells` display columns, never
    hiding a RED block behind averaging — any display cell whose source
    blocks include at least one RED renders RED (spec: this strip "is the
    visual that sells the entire project in three seconds," so a real
    compaction event must never disappear just because the strip got wide).
    """
    total = sum(n for _, n in segments)
    if total <= max_cells or total == 0:
        return segments

    colors = [color for color, n in segments for _ in range(n)]
    ratio = total / max_cells
    bucketed: list[str] = []
    for i in range(max_cells):
        lo = int(i * ratio)
        hi = max(lo + 1, int((i + 1) * ratio))
        bucket = colors[lo:hi]
        if RED in bucket:
            bucketed.append(RED)
        elif GREY in bucket:
            bucketed.append(GREY)
        else:
            bucketed.append(GREEN)

    out: list[tuple[str, int]] = []
    for color in bucketed:
        if out and out[-1][0] == color:
            out[-1] = (color, out[-1][1] + 1)
        else:
            out.append((color, 1))
    return out


def render_strip(segments: list[tuple[str, int]]) -> Text:
    text = Text()
    for color, count in segments:
        text.append("█" * count, style=color)
    return text


def split_into_runs(steps: pd.DataFrame, policy: str) -> list[pd.DataFrame]:
    """Splits one policy's rows (in on-disk order) into per-trajectory runs by
    detecting `step_idx == 0` restarts — the same boundary-detection this
    project already uses in `experiments/phase4_pareto_plot.py`'s
    `per_trajectory_totals`, generalized to keep full rows instead of just a
    summed total."""
    rows = steps[steps["policy"] == policy].reset_index(drop=True)
    if rows.empty:
        return []
    runs: list[pd.DataFrame] = []
    start = 0
    for i in range(1, len(rows)):
        if int(rows.loc[i, "step_idx"]) == 0:  # type: ignore[arg-type]
            runs.append(rows.iloc[start:i].reset_index(drop=True))
            start = i
    runs.append(rows.iloc[start:].reset_index(drop=True))
    return runs


def load_agreement(path: Path, policy: str) -> dict[int, tuple[bool, bool]]:
    """step_idx -> (tool_name_match, args_match) for `policy`, or `{}` if the
    file doesn't exist or has no rows for this policy — the caller renders an
    honest "not measured" readout rather than treating this as fatal."""
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    rows = frame[frame["policy"] == policy]
    return {
        int(r.step_idx): (bool(r.tool_name_match), bool(r.args_match))  # type: ignore[arg-type]
        for r in rows.itertuples()
    }


@dataclass
class PaneState:
    label: str
    cumulative_prefill_tokens: int = 0
    cumulative_wallclock_ms: float = 0.0
    ttft_ms: float = 0.0
    n_decisions_seen: int = 0
    n_tool_name_matches: int = 0
    n_args_matches: int = 0
    agreement_available: bool = False
    _prev_prompt_tokens: int = field(default=0, repr=False)

    def step(
        self,
        row: pd.Series,
        block_size: int,
        agreement: dict[int, tuple[bool, bool]],
        strip_width: int,
    ) -> Text:
        segments = block_segments(
            prev_prompt_tokens=self._prev_prompt_tokens,
            prompt_tokens=int(row["prompt_tokens"]),
            cached_tokens=int(row["cached_tokens"]),
            block_size=block_size,
        )
        self._prev_prompt_tokens = int(row["prompt_tokens"])
        self.cumulative_prefill_tokens += int(row["prefill_tokens"])
        self.cumulative_wallclock_ms += float(row["ttft_ms"]) + float(row["decode_ms"])
        self.ttft_ms = float(row["ttft_ms"])

        step_idx = int(row["step_idx"])
        if step_idx in agreement:
            self.agreement_available = True
            tool_match, args_match = agreement[step_idx]
            self.n_decisions_seen += 1
            self.n_tool_name_matches += int(tool_match)
            self.n_args_matches += int(args_match)

        return render_strip(compress_segments(segments, strip_width))

    def agreement_text(self) -> str:
        if not self.agreement_available:
            return "action agreement: not measured"
        if self.n_decisions_seen == 0:
            return "action agreement so far: n/a (no decision point yet)"
        tool_rate = self.n_tool_name_matches / self.n_decisions_seen
        args_rate = self.n_args_matches / self.n_decisions_seen
        return (
            f"action agreement so far: {tool_rate:.0%} tool-name, "
            f"{args_rate:.0%} args (n={self.n_decisions_seen})"
        )

    def counters_table(self, rate_usd_per_1m: float) -> Table:
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="right", style="bold")
        table.add_column()
        est_usd = self.cumulative_prefill_tokens / 1_000_000 * rate_usd_per_1m
        table.add_row("cumulative prefill:", f"{self.cumulative_prefill_tokens:,} tokens")
        table.add_row("TTFT (this step):", f"{self.ttft_ms:.0f} ms")
        table.add_row("elapsed wall clock:", f"{self.cumulative_wallclock_ms / 1000:.1f} s")
        table.add_row("est. $ (illustrative rate):", f"${est_usd:.4f}")
        return table

    def panel(self, strip: Text, rate_usd_per_1m: float) -> Panel:
        body = Group(strip, Text(""), self.counters_table(rate_usd_per_1m), Text(""),
                      Text(self.agreement_text(), style="italic"))
        return Panel(body, title=self.label, border_style="cyan")


def build_layout(naive_panel: Panel, hybrid_panel: Panel, header: str, footer: str) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(Text(header, justify="center", style="bold"), name="header", size=1),
        Layout(name="body"),
        Layout(Text(footer, justify="center", style="dim"), name="footer", size=1),
    )
    layout["body"].split_row(Layout(naive_panel, name="left"), Layout(hybrid_panel, name="right"))
    return layout


LEGEND = (
    "[green]█[/] cache hit   [red]█[/] invalidated this step   "
    "[grey50]█[/] offloaded to CPU (n/a for this pair)   —   Ctrl+C to exit"
)


def run_dashboard(
    *,
    steps_path: Path,
    agreement_path: Path,
    left_policy: str,
    right_policy: str,
    run_index: int,
    block_size: int,
    interval: float,
    strip_width: int,
    rate_usd_per_1m: float,
    max_steps: int | None,
    animate: bool,
    console: Console,
) -> None:
    steps = pd.read_parquet(steps_path)
    left_runs = split_into_runs(steps, left_policy)
    right_runs = split_into_runs(steps, right_policy)
    if run_index >= len(left_runs) or run_index >= len(right_runs):
        raise SystemExit(
            f"--run-index {run_index} out of range "
            f"({left_policy}: {len(left_runs)} runs, {right_policy}: {len(right_runs)} runs)"
        )
    left_run, right_run = left_runs[run_index], right_runs[run_index]

    left_agreement = load_agreement(agreement_path, left_policy)
    right_agreement = load_agreement(agreement_path, right_policy)

    n_steps = min(len(left_run), len(right_run))
    if max_steps is not None:
        n_steps = min(n_steps, max_steps)

    left_state = PaneState(label=f"{left_policy} (left)")
    right_state = PaneState(label=f"{right_policy} (right)")

    try:
        with Live(console=console, refresh_per_second=12, screen=True) as live:
            for i in range(n_steps):
                left_strip = left_state.step(
                    left_run.iloc[i], block_size, left_agreement, strip_width
                )
                right_strip = right_state.step(
                    right_run.iloc[i], block_size, right_agreement, strip_width
                )
                header = f"AgentKV — {left_policy} vs {right_policy}   step {i + 1}/{n_steps}"
                layout = build_layout(
                    left_state.panel(left_strip, rate_usd_per_1m),
                    right_state.panel(right_strip, rate_usd_per_1m),
                    header,
                    LEGEND,
                )
                live.update(layout)
                if animate:
                    time.sleep(interval)
    except KeyboardInterrupt:
        pass

    summary = Table(title="Replay finished")
    summary.add_column("")
    summary.add_column(left_policy, justify="right")
    summary.add_column(right_policy, justify="right")
    summary.add_row(
        "cumulative prefill tokens",
        f"{left_state.cumulative_prefill_tokens:,}",
        f"{right_state.cumulative_prefill_tokens:,}",
    )
    summary.add_row(
        "elapsed wall clock",
        f"{left_state.cumulative_wallclock_ms / 1000:.1f} s",
        f"{right_state.cumulative_wallclock_ms / 1000:.1f} s",
    )
    summary.add_row("action agreement", left_state.agreement_text(), right_state.agreement_text())
    console.print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps-path",
        type=Path,
        default=REPO_ROOT / "results" / "phase4" / "phase4_pareto_cost_steps.parquet",
    )
    parser.add_argument(
        "--agreement-path",
        type=Path,
        default=REPO_ROOT / "results" / "phase4" / "phase4_dashboard_agreement.parquet",
    )
    parser.add_argument("--left", default="naive")
    parser.add_argument("--right", default="hybrid")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--interval", type=float, default=0.4, help="seconds between steps")
    parser.add_argument("--strip-width", type=int, default=48, help="max block cells per pane")
    parser.add_argument("--rate-usd-per-1m", type=float, default=DEFAULT_RATE_USD_PER_1M)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--no-animate",
        action="store_true",
        help="render every step with no delay between frames (smoke test, not a real demo run)",
    )
    args = parser.parse_args()

    block_size = args.block_size
    if block_size is None:
        model_config = yaml.safe_load((REPO_ROOT / "configs" / "model.yaml").read_text())
        block_size = model_config["engine"]["block_size"]

    console = Console()
    run_dashboard(
        steps_path=args.steps_path,
        agreement_path=args.agreement_path,
        left_policy=args.left,
        right_policy=args.right,
        run_index=args.run_index,
        block_size=block_size,
        interval=args.interval,
        strip_width=args.strip_width,
        rate_usd_per_1m=args.rate_usd_per_1m,
        max_steps=args.max_steps,
        animate=not args.no_animate,
        console=console,
    )


if __name__ == "__main__":
    main()
