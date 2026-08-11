# Phase 6 (Demo design) Summary — the dashboard and the deck

Status: both artifacts spec §6 requires are built and smoke-tested. §6.3
(README structure), §6.4 (technical writeup), and Phase 6's own §6.1
(analytical cost model, `serving/cost_model.py`) are separate, not-yet-started
items — see "Next" below for how this work relates to them.

## What was built

- `src/agentkv/viz/dashboard.py` — the live side-by-side dashboard (§6.1).
  `rich`-based terminal TUI, replays committed data, no GPU needed at demo
  time.
- `experiments/phase4_dashboard_agreement.py` — precompute script measuring
  naive-vs-hybrid next-action agreement on one demo trajectory, feeding the
  dashboard's quality readout.
- `results/phase4/phase4_dashboard_agreement.parquet` — that script's output
  (11 decision points, traj-000).
- `experiments/demo_deck.py` — the three-figure deck (§6.2). Pure
  post-processing, no GPU needed.
- `results/demo_deck/deck_1_cliff.png`, `deck_2_fix.png`, `deck_3_pareto.png`,
  `rehearsal.md` — that script's output, plus the 60-second/5-minute scripts.
- `tests/test_dashboard.py`, `tests/test_demo_deck.py` — 13 unit tests
  covering the block-invalidation math, the strip-compression algorithm, and
  the compaction-event detection used to draw the cliff figure's dashed
  lines.

## Decision 1: replay committed data, never drive vLLM live

Spec's bar for §6.1: "must run reliably during an interview on a laptop with
no internet." This project has hit real GPU/WSL fragility repeatedly this
session alone — sustained thermal throttling during the Phase 4.5 sweep
(temperatures climbing 63°C→85°C, utilization dropping to 7-45%) and one
genuine process stall requiring a `kill -9` (see `phase4_pareto_summary.md`).
Driving live inference during an actual interview would reintroduce exactly
that risk for no benefit — every number the dashboard shows was already
measured once, correctly, with GPU telemetry and the theory-match check
(`bench/divergence.py`). `dashboard.py` has no `vllm`/`torch` import; it only
needs `pandas`, `pyyaml`, and `rich` to run, and only reads already-committed
parquet files.

The one-time cost of this decision: `viz/dashboard.py` and
`experiments/phase4_dashboard_agreement.py` had to be two separate files
(dashboard = pure replay, agreement precompute = one real GPU run), rather
than one live script. Precompute output is committed like every other
figure's source data (spec §7: "never hand-edited").

## Decision 2: terminal TUI (`rich`), not a web UI

Spec explicitly allows either ("a simple web UI or a terminal TUI both
work"). `rich` was already a pinned dependency (`pyproject.toml`); no web
framework (Flask/FastAPI/Streamlit) is. Given spec's own "do not
over-engineer this" instruction, adding a new dependency class (server,
port, browser) for a capability the existing dependency set already covers
wasn't a genuine trade-off worth raising — it followed directly from what
was already in the project.

## Decision 3: naive-vs-hybrid agreement had to be precomputed, not just replayed

The dashboard's "running quality readout" (§6.1: "action agreement so far")
needs per-step data for **both** panes. Only naive/append_only agreement had
ever been measured (`results/phase3/phase3_agreement.parquet`,
`experiments/phase3_agreement.py`) — hybrid's agreement was never computed
anywhere in the project, since it predates `hybrid` (Phase 4.4). This was a
genuine fork, put to the user directly: (a) precompute real hybrid agreement
data first, (b) ship the dashboard now with an honest "not measured"
placeholder for hybrid, or (c) drop the quality readout from this pass
entirely. **Chosen: (a).** `experiments/phase4_dashboard_agreement.py`
mirrors `phase3_agreement.py`'s method exactly (grammar-constrained
elicitation via `bench/agreement.py`, scored against a fresh "full
uncompacted context" counterfactual) but swaps `append_only` for `hybrid` and
is deliberately scoped to one trajectory (traj-000), not a full 14-trajectory
sweep — this exists to feed one demo replay, not to produce a new
statistical claim.

Result (11 decision points before the raw-uncompacted counterfactual itself
hit `max_model_len=16384` at step 59 — the same known limitation
`phase3_agreement.py` already documents for this methodology): hybrid agrees
with its own uncompacted counterfactual more often than naive does (64%
tool-name / 55% args vs. naive's 45%/45%). Real, if small-n, signal — not
fabricated, and the dashboard degrades to "not measured" automatically for
any policy pair without precomputed data (e.g. `--left kv_evict`).

## Decision 4: how the context-map strip's colors are derived

Spec: "one cell per KV block, colored green (cache hit) / red (invalidated
this step) / grey (offloaded to CPU)." No new accounting was introduced —
the strip is computed purely from `prompt_tokens`/`cached_tokens`/
`block_size`, the same measured quantities `bench/divergence.py`'s
theory-match check already validates:

- `reused = cached_tokens // block_size` blocks at the front: **green**
  (still resident from the previous step).
- `prev_cacheable_blocks - reused` blocks right after: **red** — blocks that
  *were* cacheable a step ago but weren't reused now (`invalidated_block_count`
  in `dashboard.py`, factored out so `demo_deck.py` reuses the identical
  math to decide which steps get a dashed line on the cliff figure, instead
  of joining against the ambiguous events parquet — see Decision 6).
- anything beyond that up to the current prompt's own block count: **green**
  again (freshly computed this step, resident going forward).
- **grey** (offloaded to CPU) is in the legend/palette for the full 3-color
  contract but never fires for a naive/hybrid comparison — CPU offload
  (§4.3) is an orthogonal serving-layer overlay, not one of the 5
  `CompactionPolicy` objects this dashboard can select between. Documented
  rather than silently dropped from the legend.

This width-based model naturally reproduces spec's own framing ("naive goes
red from the middle to the end... yours reddens only the tail") without any
policy-specific logic — but the real, measured shape of that difference is
more precise than the spec's wording suggests (see "A nuance worth stating
honestly" below).

Wide strips (a trajectory can reach 600+ blocks) are downsampled to fit the
terminal (`compress_segments`, default 48 cells), bucketing multiple blocks
into one display cell. The bucketing always prefers RED over GREEN/GREY if
any source block in that bucket was red — a real compaction event can never
disappear from the strip just because it got compressed (unit-tested:
`test_compress_segments_never_hides_red`).

## Decision 5: which trajectory, and how "same trajectory, different policy" is guaranteed without a trajectory-id column

`results/phase4/phase4_pareto_cost_steps.parquet` has no trajectory-id
column — only `step_idx`, which resets to 0 at each new trajectory. But
`experiments/phase4_pareto_cost.py`'s main loop iterates the same 14
trajectories in the same fixed order for every policy (only *which policy
runs first per trajectory* rotates, per spec §0.2's interleaving). That
guarantees naive's Nth trajectory-run and hybrid's Nth trajectory-run are the
same underlying trajectory. `split_into_runs` (in `dashboard.py`, reused by
`demo_deck.py`) exploits exactly this: it detects run boundaries by
`step_idx == 0` resets (the same technique `experiments/
phase4_pareto_plot.py`'s `per_trajectory_totals` already used for Gate 4's
figure) and returns full per-run frames instead of just summed totals.
Both the dashboard and the deck use run index 0, which is `traj-000` by this
project's established `--comparison-trajectory`/`--cliff-trajectory`
default convention.

## Decision 6: the deck regenerates the cliff and fix panels instead of reusing `phase1_cliff.png`

`results/phase1/phase1_cliff.png` already existed and looked like an obvious
candidate to reuse for the deck's first panel. It wasn't used: README's
Environment section documents that Phase 2 introduced the
`--use-fallback` convention after the primary model's KV headroom proved too
tight (1.15x, "tight" per `configs/model.yaml`'s own comment) — Phase 1
predates that decision. Reusing `phase1_cliff.png` as-is would have silently
put a different model on the deck's panel 1 than panels 2 and 3 both use
(the fallback model, via `phase4_pareto_cost_steps.parquet`). Both panels are
instead regenerated from that same committed fallback-model data — zero new
GPU compute, since it's the same data the dashboard already reads, just
sliced and handed to the existing `plot_cliff`/`plot_policy_comparison`
functions in `viz/figures.py`.

The events needed to draw the cliff figure's dashed compaction-event lines
have the same ambiguity problem as Decision 5 (no trajectory-id column, and
events are sparse so `step_idx == 0` boundary detection doesn't apply to
them directly). Rather than trying to join the events parquet positionally,
`demo_deck.py`'s `fired_events_frame` recomputes "did this step invalidate
cache" directly from the steps data using the same `invalidated_block_count`
helper the context-map strip uses — one source of truth for "did a
compaction fire here," not two.

## Decision 7: the fix panel excludes `none`

Spec's §6.2 wording ("all policies overlaid") predates `none`, which Phase
4.5 added specifically as Gate 4's 5th Pareto point — a ceiling reference
("what if you never compact"), not a compaction strategy competing on "cost
removed" (see `phase4_pareto_summary.md`'s "The 5th policy"). `none` also
hits `context_exceeded` on every trajectory in the Pareto sweep, so its line
would stop early from a context-length crash, not because it finished — that
would read as a 5th contender in a race it was never entered in. The fix
panel overlays the 4 real `CompactionPolicy` implementations only: naive,
append_only, kv_evict, hybrid.

## A nuance worth stating honestly

Spec's framing — "naive goes red from the middle to the end in one frame;
yours reddens only the tail" — suggested hybrid's red segments would simply
be *smaller* than naive's at each compaction event. Checked this against the
real data (`traj-000`, fallback model) before writing it up anywhere: at each
individual compaction event, the **raw count** of invalidated blocks is
actually similar in magnitude for both policies (naive: 542-605 red blocks
per event; hybrid: 507-594). What differs is hybrid's **green (reused)
prefix growing** across successive compactions — 3 blocks at its first event,
96 blocks by its eighth — while naive's stays flat at ~4 blocks every time.
The honest version: hybrid's red *fraction* of the strip shrinks over the
trajectory as it accumulates a protected prefix, not "hybrid's red is small
on every event." The strip renders this correctly because it's driven by the
real per-step numbers, not a hardcoded story — the more flattering-but-wrong
version was never in the code.

## Known limitations

- **Agreement data covers 11 of `traj-000`'s 27 decision points** — the
  elicitation method's "full uncompacted context" counterfactual hits
  `max_model_len` at step 59, a limitation `phase3_agreement.py` already has
  for the same reason. The dashboard's readout is correct as far as it goes,
  just partial.
- **Agreement data exists only for naive/hybrid, one trajectory, one seed**
  — `--left`/`--right` with any other policy shows "not measured" rather
  than fabricating a number; re-running
  `experiments/phase4_dashboard_agreement.py` with different
  `--trajectory`/policy wiring would need code changes (it's hardcoded to
  naive+hybrid, matching what the dashboard actually needs).
- **The "$ at production API rate" counter is illustrative, not measured**
  — `DEFAULT_RATE_USD_PER_1M = 0.15` is a representative small-hosted-model
  input rate, clearly commented as such in `dashboard.py`. The project's
  real analytical cost model (`serving/cost_model.py`, spec's Phase 6 §6.1)
  is separate, not-yet-built work.
- **Grey (offloaded to CPU) never fires** for any policy pair the dashboard
  can currently show — CPU offload isn't one of the 5 `CompactionPolicy`
  objects. The legend still documents it per spec's 3-color contract.
- **The deck's panel 1/2 numbers are one trajectory, not the aggregate** —
  clearly labeled as such in `rehearsal.md`; the aggregate, statistically-
  tested numbers (median reductions, Wilcoxon significance) are also in
  `rehearsal.md`'s 5-minute version, sourced from
  `phase4_pareto_cost_summary.md`, not re-derived.

## How to run it

Both commands assume the WSL venv (`source .venv/bin/activate` inside
Ubuntu-24.04, per README's Environment section).

**The dashboard** (interactive, ~65s for the full replay at default pacing):

```bash
python src/agentkv/viz/dashboard.py
```

Flags: `--no-animate` (render instantly, no per-step delay — useful for a
quick check, not a real demo pace), `--interval SECONDS` (default 0.4),
`--max-steps N` (preview just the first N steps), `--left`/`--right` (policy
pair, default naive/hybrid), `--strip-width N` (display columns per pane,
default 48), `--rate-usd-per-1m` (override the illustrative $ rate). Ctrl+C
exits cleanly at any point and still prints the final summary table.

Also runnable as `python -m agentkv.viz.dashboard` (the Makefile's `make
demo` target already expects this entrypoint) or `make demo`.

**The deck** (instant, no GPU):

```bash
python experiments/demo_deck.py
```

Regenerates `results/demo_deck/deck_1_cliff.png`, `deck_2_fix.png`, and
copies `deck_3_pareto.png`. Then open `results/demo_deck/rehearsal.md` for
the 60-second and 5-minute scripts to present them with.

**Regenerating the agreement precompute** (needs the WSL venv + GPU; only
necessary if changing the demo trajectory or wiring a different policy
pair):

```bash
python experiments/phase4_dashboard_agreement.py --use-fallback --trajectory traj-001
```

## Validation

`python -m pytest -q`: 156 passed. `ruff check .`: all checks passed. `mypy
src`: no issues found in 32 source files. Both `dashboard.py` (`--no-animate
--max-steps N`) and `demo_deck.py` were run end-to-end and their output
visually inspected (the strip renders correctly for both empty-agreement and
real-agreement states; all three deck figures render with correct titles,
legends, and — for the fix panel — hybrid visibly lowest of the four lines).

## Next

Not started here: §6.3 (README structure — figure first, one-sentence
result, Pareto plot, reproduce steps, limitations section) and §6.4
(2,000-3,000 word technical writeup). Phase 6's own §6.1 (the analytical
cost model, `serving/cost_model.py`) and §6.2 (`make reproduce`/
`experiments/run_all.py`, which doesn't exist yet — the Makefile's
`reproduce` target currently points at a file that isn't built) are also
separate, not-yet-started work; this summary's rehearsal script deliberately
doesn't claim `make reproduce` works today.
