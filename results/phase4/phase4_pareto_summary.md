# Phase 4.5 Summary — The Pareto Plot (Gate 4)

Status: built and measured. Final sub-part of Phase 4 (§4.1 kv_evict, §4.2
attn_evict, §4.3 offload, §4.4 hybrid all complete and written up
separately).

## What was built

- `src/agentkv/policies/none.py` — `NoOpPolicy`: never compacts, ever. Added
  as a real, honestly-labeled 5th policy to reach Gate 4's "≥5 policies" bar
  (see "The 5th policy" below for why `attn_evict`/`offload` don't fill that
  slot).
- `src/agentkv/bench/stats.py` — `wilson_confidence_interval`: Wilson score
  interval for a binomial proportion, used for the Y-axis (task success
  rate) error bars.
- `experiments/phase4_pareto_cost.py` — extends `phase4_hybrid.py`'s 4-policy
  replay sweep to 5 (adds `none`), full 14-trajectory run. X-axis data.
- `experiments/phase4_pareto_success.py` — extends `phase3_ledger.py`'s live
  ledger-task sweep to 5 policies, 5 seeds each (25 episodes). Y-axis data.
- `src/agentkv/viz/figures.py` — `plot_pareto`: the actual Gate 4 figure,
  X = median prefill tokens (IQR error bars), Y = task success rate (95%
  Wilson CI error bars), one point per policy, incomplete-run policies drawn
  as hollow markers rather than silently mixed in with complete ones.
- `experiments/phase4_pareto_plot.py` — combines both sweeps' committed
  parquet output into `results/phase4/phase4_pareto.png` +
  `phase4_pareto_table.md`. Pure post-processing, no GPU/vLLM needed.

## Gate 4

> "A Pareto plot exists with ≥5 policies and ≥5 seeds each, and you can
> defend every point on it."

| policy | n_traj | prefill median | prefill IQR | successes | success rate | 95% Wilson CI |
|---|---|---|---|---|---|---|
| naive | 14 | 78436 | [73727, 83012] | 0/5 | 0.000 | [0.000, 0.434] |
| append_only | 14 | 85496 | [80968, 91878] | 0/5 | 0.000 | [0.000, 0.434] |
| kv_evict | 14 | 106030 | [91427, 117312] | 0/5 | 0.000 | [0.000, 0.434] |
| hybrid | 14 | 70026 | [64753, 76374] | 0/5 | 0.000 | [0.000, 0.434] |
| none | 14 | 16388 | [16276, 16596] | 0/5 | 0.000 | [0.000, 0.434] |

5 policies, ≥5 seeds each (14 trajectories for the cost axis, 5 ledger
episodes for the success axis). The plot exists at
`results/phase4/phase4_pareto.png`. Every point is defensible — including
`none`'s, which is deliberately drawn differently (see below) rather than
presented as equivalent to the other four.

## The 5th policy

Gate 4 literally asks for "≥5 policies." This project has 4 real
`CompactionPolicy` implementations with prefill-cost data (naive,
append_only, kv_evict, hybrid). `attn_evict` (§4.2) was proven infeasible on
this hardware and never built; `offload` (§4.3) is an orthogonal
serving-layer overlay measured on wall-clock, not a distinct content policy
on this prefill-token axis — neither is a natural 5th point here.

Presented this as a genuine three-way fork to the user (add a `none`
baseline / report honestly with 4 policies / use offload as a secondary,
differently-labeled point). **Chosen: add a `none` baseline** — a real,
cheap, honestly-labeled ceiling: never compact, ever. It is exactly the
reference point every other policy is implicitly measured against, and
adding it required touching exactly one new file (spec §10's own
convention).

`none`'s cost is real but **incomplete**: on every one of the 14
trajectories, its ever-growing uncompacted context blew past the fallback
model's 16,384-token limit partway through (median ~50-63 of ~161-169
steps). This is itself informative — no compaction cannot survive most of a
long trajectory at all, on this hardware — but it is not a full-trajectory
cost, so it is never averaged into the other four policies' comparisons and
is drawn as a hollow marker on the Pareto plot, distinct from the four
complete points.

## Results: cost axis differentiates cleanly, quality axis is flat at zero

The X-axis (prefill cost) tells a clear, internally consistent story:
**hybrid is cheapest among the complete policies (70,026 median), kv_evict
is most expensive (106,030), naive and append_only sit in between**. This
matches the direction of every earlier Phase 4 sub-phase's findings (hybrid
wins, kv_evict regresses) — see "A note on why the numbers differ slightly
from §4.1/§4.4" below for why the exact figures here aren't byte-identical
to those separately-run sweeps.

The Y-axis (task success rate) does not differentiate at all: **0/5 for
every single policy**, extending Phase 3.3's naive/append_only finding
(`results/phase3/phase3_ledger_summary.md`) to kv_evict, hybrid, and none.
This was flagged as a real possibility before running anything (a genuine
floor effect, not a policy effect, since the model's raw capability to
complete this 50-transaction task is what's failing, and every policy uses
the identical underlying model). It held.

**This means Gate 4's Pareto plot cannot yet show a genuine cost/quality
*tradeoff* — only a cost *ranking*.** The plot is real, defensible, and
built exactly to spec, but its most interesting axis (Y) is degenerate for
this task/model/hardware combination. That is itself the finding, reported
plainly rather than hidden or dressed up.

## A real, if partial, quality signal underneath the 0/5 floor

Task success is binary and identical across policies, but the *ledger
events* underneath it are not. Averaged across the 5 seeds:

| policy | mean tool calls used | mean instructions left unprocessed at failure |
|---|---|---|
| naive | 93.2 | 9.2 |
| append_only | 93.2 | 9.2 |
| kv_evict | 94.6 | 8.6 |
| **hybrid** | **110.6** | **0.6** |
| none | 93.2 | 9.2 |

Naive/append_only/kv_evict/none all left a similar tail of the 50-item
transaction queue unprocessed when they failed (~9 of 50, on average) —
essentially indistinguishable from each other. **Hybrid did not**: it used
substantially more of its tool-call budget (110.6 vs ~93-95) and left
almost nothing unprocessed (0.6 vs ~9). Hybrid's episodes got the model
through nearly the entire mechanical task before failing at the final
reported-balance step; the other four failed both mechanically (stopped
partway through the queue) and on the final numbers. The *reported* final
balance was still wrong all 5 times for hybrid too — so this is not a
success-rate win — but it is a real, measurable difference in how far
compaction lets the model get, consistent with hybrid's keep-verbatim
routing protecting exactly the short numeric transaction-result turns this
task's ground truth depends on. Worth a dedicated follow-up (e.g., loosening
the `n_transactions`/`max_tool_calls` budget specifically for hybrid) rather
than treated as settled by this one measurement.

## A note on why the numbers differ slightly from §4.1/§4.4

`append_only`/`kv_evict`/`hybrid`'s prefill totals here (e.g. hybrid median
70,026) are close to but not identical to the separately-run 3-way (§4.1)
and 4-way (§4.4) sweeps (e.g. hybrid median there was ~73,552, computed
across a slightly different comparison set). This is real, expected
variance, not a bug: this sweep interleaves **5** policies per trajectory
(spec §0.2's A/B/C/D/E rotation, extended), while §4.1 interleaved 3 and
§4.4 interleaved 4. Trajectory anchors are shared/seed-independent across
this project's synthetic trajectories, so whichever policy happens to run
*first* on a given trajectory inherits a cold cache, and whichever runs
*after* another policy on the *same* trajectory can inherit a warm one from
shared anchor blocks — changing the rotation order changes which policy
gets which starting condition, trajectory by trajectory. Every comparison
*within* this sweep is still validly paired and interleaved per spec §0.2;
it is simply a different, self-consistent 5-way configuration than the
separately-run 3-way/4-way ones, not a contradiction of them.

## Bugs found and fixed while building this

Three real bugs, all caught before they reached committed results:

1. **`none` overflowing `max_model_len` crashed the replay harness.** Every
   other policy in this project actively compacts, so no prior sweep ever
   sent an over-length prompt to vLLM. `none` does, on every trajectory.
   Fixed in `phase4_pareto_cost.py`'s `run_trajectory` with a proactive
   length check before each `generate_step` call (mirroring the pattern
   `phase4_pareto_success.py`'s live ledger loop already used for the same
   reason) — the run stops early for that policy/trajectory, records what
   was actually completed, and flags it (`context_exceeded=True`) rather
   than crashing or silently truncating unlabeled.
2. **Ran the first full sweep on the primary (1.7B, 8192-token) model
   instead of the fallback (0.6B, 16384-token) model** every earlier Phase 4
   sub-phase used (`configs/model.yaml`'s own comment documents this exact
   class of overflow being hit once before, during Phase 2). On primary,
   `append_only`'s monotonically-growing `frozen` list and `kv_evict`'s
   near-total per-step cache invalidation both genuinely don't fit the
   tight budget — confirmed by reproducing the identical crash on the
   original, unmodified `phase4_kv_evict.py`, proving it wasn't a bug in
   this phase's new code. Fixed by adding `--use-fallback`.
3. **`RecordCollector.flush()`'s append-not-overwrite behavior mixed two
   sweeps' data in the same parquet.** Reused the `phase4_pareto_cost_*`
   output filenames across the failed primary-model attempt and the
   corrected fallback rerun; the first attempt's `flush()` call had already
   written 14 trajectories' worth of (bad) primary-model data before
   crashing on `none`, and the second run's `flush()` appended 14 more
   fallback-model trajectories on top — 28 "trajectories" of mixed data per
   policy. This is exactly the failure mode `phase4_hybrid.py`'s own
   docstring warns about (why it writes fresh filenames instead of reusing
   `phase4_kv_evict_*`'s), and this phase should have followed that
   precedent on every rerun, not just the first attempt. Fixed by deleting
   the contaminated parquet and rerunning clean once more.

## Known limitations

- **The Y-axis is degenerate for this task/model/hardware combination** —
  0/5 for all five policies means the Pareto plot cannot show a genuine
  cost-vs-quality tradeoff here, only a cost ranking. A model or task
  capable of >0% success would be needed to see whether hybrid's cost
  advantage also buys better quality, or costs some.
- **`none`'s point is a partial-trajectory cost**, not comparable
  apples-to-apples to the other four's full-trajectory totals; drawn
  distinctly on the plot for exactly this reason, but still worth restating
  in prose since a table row alone doesn't carry that caveat as visibly as
  the figure does.
- **Single configuration per policy** — one threshold/window/heuristic
  setting each, same as every earlier Phase 4 sub-phase.
- **The hybrid mechanical-completion signal (110.6 tool calls, 0.6
  instructions left) is n=5, one seed set, one task configuration** —
  suggestive, not yet a controlled finding on its own.
- **Fallback (0.6B) model** for both sweeps, consistent with every other
  Phase 4 sub-phase's precedent.

## Next

Not started here, and not decided: whether to invest in getting *any*
policy above the 0/5 floor (e.g. a larger model, an easier task
configuration, or specifically probing whether loosening hybrid's budget
converts its "got through nearly the whole queue" mechanical result into
occasional real successes) before treating the Pareto plot's quality axis as
final. Phase 6's cost-model extrapolation and writeup can proceed using the
cost axis and the 0/5 finding as-is — both are real, measured results — but
the project's central Pareto claim ("hybrid dominates both pure strategies")
is only half-tested: proven on cost, untested on quality, because quality
couldn't be measured above zero for anything yet.
