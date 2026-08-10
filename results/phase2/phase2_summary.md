# Phase 2 Summary — Cache-Preserving Context Layout

Status: **implementation complete, Gate 2 NOT PASSED**. See `AGENTKV_SPEC.md` §2
for the phase definition and gate criteria. The pipeline is built, tested, and
produced trustworthy measurements end-to-end (63 tests passing, mypy/ruff
clean, zero unexplained theory-vs-measurement mismatches across the full
sweep) — but the core hypothesis, that append-only layout reduces cumulative
prefill tokens by a statistically significant margin vs. naive, did not hold
on this workload.

## What was built

- `src/agentkv/context/segments.py` — `ContextState` (anchor/frozen/live) and
  `Segment` (spec §2.1): `frozen` is a type-level guarantee of append-only-ness,
  not just a convention. `AnchorHygieneGuard`/`hash_anchor_rendering` (spec
  §2.4): hashes the anchor's actual token-id *rendering* every step, catching
  real tokenization-affecting drift (reordered JSON keys, timestamps).
- `src/agentkv/policies/append_only.py` — `AppendOnlyPolicy` (spec §2.2):
  stateful across calls (unlike `NaivePolicy`'s per-call re-derivation), retires
  the oldest fraction of `live` into a new `Segment`, appends it to `frozen`,
  never rewrites prior segments.
- `src/agentkv/context/alignment.py` — block-padding helpers (spec §2.3):
  pads after the anchor and after each frozen segment (never after the
  trailing live group) so a divergence never lands mid-block.
- `src/agentkv/bench/stats.py` — paired Wilcoxon signed-rank test (no scipy
  dependency), used by Gate 2's significance check.
- `experiments/phase2_layout.py` — the runnable experiment: replays every
  trajectory under naive and append-only back-to-back (order alternated per
  trajectory, spec §0.2's interleaved A/B), against one shared vLLM engine,
  producing `phase2_steps.parquet`, `phase2_events.parquet`,
  `phase2_comparison.png`, and this gate summary.
- A correction to `src/agentkv/bench/divergence.py`'s theory-vs-measurement
  check, made one-directional — see "Debugging history" below; this was the
  main non-trivial engineering work of the phase.

## Gate 2 result

Spec: *"Append-only layout reduces cumulative prefill tokens per 100-step
trajectory by a measurable, statistically significant margin vs. naive, at
equal or better action agreement."*

Ran across all 14 valid trajectories, `Qwen/Qwen3-0.6B` (fallback model —
see "Model choice" below), `max_model_len=16384`, compaction threshold at 60%
of window, retire fraction 0.5 (both policies — identical configuration,
differing only in whether the compacted-away region is replaced or appended).
Every one of the ~4,600 steps passed the (corrected) theory-vs-measurement
check.

- **Prefill-token reduction, append-only vs. naive: median 0.7%** (IQR
  [-0.9%, 1.9%], n=14)
- **Paired Wilcoxon signed-rank test on total prefill tokens:** W=42.0,
  z=-0.628, **p=0.53 — not significant at α=0.05**
- **Monotonic prefix growth (spec §2.2's headline systems property — successive
  append-only events' divergence index non-decreasing): 14/14 trajectories.**
  The core mechanism works exactly as designed; it just doesn't move the
  prefill-token needle enough to clear significance (see below).
- Action agreement (the other half of Gate 2) is out of scope here — Phase 3
  infrastructure, not yet built.

Per-trajectory detail:

| trajectory | naive prefill | append_only prefill | reduction % |
|---|---|---|---|
| traj-000 | 93,276 | 92,872 | 0.4% |
| traj-001 | 96,350 | 93,306 | 3.2% |
| traj-002 | 85,528 | 84,042 | 1.7% |
| traj-003 | 94,264 | 92,393 | 2.0% |
| traj-004 | 92,947 | 93,934 | -1.1% |
| traj-005 | 64,073 | 65,535 | -2.3% |
| traj-006 | 75,828 | 80,303 | -5.9% |
| traj-008 | 83,174 | 81,102 | 2.5% |
| traj-009 | 84,057 | 84,495 | -0.5% |
| traj-010 | 75,781 | 75,205 | 0.8% |
| traj-011 | 81,410 | 86,496 | -6.2% |
| traj-012 | 91,000 | 90,332 | 0.7% |
| traj-013 | 90,401 | 89,856 | 0.6% |
| traj-014 | 83,862 | 80,902 | 3.5% |

Roughly half the trajectories show append-only *worse* than naive. This
isn't noise around zero from a policy that's secretly winning — it's a
policy whose net effect on total prefill tokens, over this trajectory
length, is genuinely close to a wash.

## Why it didn't work according to the hypothesis

The spec's intuition (§2.2) is that because `frozen` is never rewritten, a
compaction event only has to pay for what's genuinely new — the divergence
point should creep later and later, so "later compactions are cheaper than
earlier ones." **That part is true and confirmed** (monotonic prefix growth,
14/14). What breaks the hypothesis is a second effect the spec didn't
account for: **append-only's own total context size grows right along with
its cached prefix**, because `frozen` accumulates without ever being pruned
or re-merged. A bigger *cached* prefix and a bigger *total* prompt grow
together, and the token count that actually needs prefilling —
`tokens_after_compaction − divergence_point` — ends up similar in magnitude
for both policies even though the two sides of that subtraction look
completely different.

Concretely, from `traj-000`'s 9 compaction events (identical config, same
seed, run back-to-back on one shared cache):

| event | naive divergence idx | naive tokens after | naive reprefilled | append-only divergence idx | append-only tokens after | append-only reprefilled |
|---|---|---|---|---|---|---|
| 1 | 59 | 4,794 | 4,746 | 59 | 4,801 | 4,753 |
| 2 | 68 | 5,251 | 5,187 | 80 | 5,182 | 5,102 |
| 6 | 71 | 4,539 | 4,475 | 164 | 4,436 | 4,276 |
| 7 | 73 | 5,979 | 5,915 | 486 | 6,821 | 6,341 |
| 8 | 73 | 4,534 | 3,942 | 808 | 4,765 | 3,965 |
| 9 | 68 | 6,079 | 6,015 | 830 | 6,332 | 5,516 |

By event 9, append-only's divergence point has climbed to token 830 versus
naive's flat ~68 — a >12x larger untouched, cached prefix, exactly as
designed. But the actual reprefilled-token count is only ~8% lower (5,516 vs
6,015), because append-only's post-compaction context (6,332 tokens) is
*also* larger than naive's (6,079): naive replaces its one summary in place
and stays bounded; append-only keeps every prior segment, so the "new tail"
that must be prefilled after the (very late) divergence point is itself
bigger than naive's "everything" that gets reprefilled from a much earlier
point in a smaller context. The two effects — a growing cached prefix and a
growing total context — largely cancel out in absolute prefill-token terms.

This also explains the event-level statistics directly: across all 14
trajectories, naive fired 115 compaction events and append-only fired 116
(essentially identical — both trigger off the same 60%-of-window threshold
against comparably-sized contexts at fire time, mean `tokens_before` 10,129
vs 10,134), and the mean reprefill per event is nearly identical (5,159 vs
5,092 tokens; reprefill-to-saved ratio 1.118 vs 1.127). Cache preservation is
real and measurable at the level of *which* tokens get reused — it just
isn't large enough, over a ~165-step/~9-event trajectory with retire_fraction
0.5, to produce a significant reduction in *how many* tokens get reprefilled
overall.

**What this suggests, not yet tested:** the design tension flagged earlier in
this project (append-only's `frozen` growing forever vs. naive's bounded
resummarization — the reason `max_model_len` had to be split per model in
Phase 2's config) is the same tension responsible for this null result. A
longer trajectory (more compaction cycles) might let the compounding
monotonic-growth advantage eventually outrun the extra bytes `frozen`
accumulates; conversely, an even longer one might make append-only worse, as
seen on `results/phase0_summary.md`'s excluded `traj-007` case and Phase 2's
own `max_model_len` overflow finding. A policy that periodically re-merges
or prunes old frozen segments (a hybrid of naive's boundedness and
append-only's cache preservation) is a natural next experiment, not
attempted here — Phase 2's spec scope was the pure append-only policy as
specified.

## Debugging history worth remembering

1. **A settle-timing fix that turned out to be a red herring.** The first
   theory-vs-measurement mismatch found in Phase 2 (`traj-000`/naive, step
   139: predicted 584 invalidated blocks, measured 551) was initially
   diagnosed as `VLLMEngine.complete_text` not settling the
   `gpu_prefix_cache_hits_total` counter before returning (unlike
   `generate_step`). That fix was applied and unit-verified, but a
   *stub-summarizer* replay used to confirm it never actually exercised the
   real bug path — a stub never calls the LLM, so there was no real
   summarization prefill to race against. Re-running the real full sweep
   reproduced the *exact same* mismatch (584 vs 551, byte-for-byte) after the
   "fix" — proof the settle timing was never the cause.
2. **The real cause: legitimate cache reuse the theory model can't see.**
   Direct investigation (probing the full prompt-request history, not just
   the single immediately-previous request) found the new prompt's
   post-divergence content — the LLM-generated summary — verbatim inside a
   *non-adjacent* earlier request's prompt. With `temperature=0`, Qwen3-0.6B
   frequently quotes retired-turn text into its summaries rather than
   paraphrasing it, and that exact text is often still resident in vLLM's
   *global* prefix-cache pool (which spans the engine's entire request
   history, not just the one previous request) from a recent-but-not-adjacent
   request. This is real, deterministic, reproducible reuse — not a
   measurement bug.
3. **Fix: made the theory check one-directional** (`assert_theory_matches` in
   `src/agentkv/bench/divergence.py`). It now only raises when measured
   cache reuse falls *short* of the pairwise prediction (a genuine red flag —
   the cache doing worse than the model expects), not when it exceeds it.
   Gate 2's real prefill/token accounting comes from vLLM's own measured
   values, not this theoretical prediction, and the effect is symmetric
   across both policies (both share one summarizer), so it doesn't bias the
   naive-vs-append-only comparison. Locked in with a new test
   (`test_assert_theory_matches_allows_unbounded_extra_reuse`).
4. **Stale zombie GPU processes silently blocking the sweep.** Mid-run, a
   fresh `vllm serve` subprocess sat at 0% GPU utilization and 0MB VRAM for
   several minutes (should have loaded a 0.6B model in under 90s). Traced to
   two orphaned `multiprocessing` worker processes, days old, still holding
   registered (but unresolvable/zombie) CUDA compute contexts on the GPU from
   an earlier, unrelated crashed run. Killing them let the new server
   initialize normally within seconds. Not a code bug, but worth remembering
   as an environment failure mode on this shared WSL2/RTX-4060 setup:
   `nvidia-smi --query-compute-apps` is the fast way to check for this before
   assuming a hang is a code problem.
5. **Model choice for Phase 2.** `AppendOnlyPolicy.frozen` growing without
   bound pushed one trajectory to 9,449 tokens against the primary model's
   `max_model_len=8192` (a real HTTP 400). Fixed by splitting
   `max_model_len` per model in `configs/model.yaml` (primary stays 8192 —
   only 1.15x measured KV-cache headroom, too tight to safely raise;
   fallback raised to 16384 — 3.56x headroom) and running Phase 2 against the
   fallback model, `Qwen/Qwen3-0.6B`, per spec §8's documented mitigation.
   This means Phase 2's numbers are not directly comparable to Phase 1's
   (measured on the 1.7B primary model) in absolute terms — the spec's claim
   that the *relative* naive-vs-append-only effect is model-size-independent
   is asserted, not verified here.

## Known limitations

- **Action agreement not measured.** Gate 2 has two halves; this phase only
  addresses the prefill-reduction half. Phase 3's `bench/agreement.py` is
  required before Gate 2 can be fully evaluated even if the significance
  result above had been positive.
- **Trajectory length is fixed at ~165 steps / ~9 compaction events.** As
  argued above, this may simply be too short a horizon for append-only's
  compounding cache-preservation advantage to outweigh its unbounded-growth
  cost. The null result reported here is scoped to this trajectory length,
  not a general claim about append-only layout.
- **Single retire_fraction (0.5) and threshold (60% of window) tested**,
  identical for both policies. Whether a different retirement schedule
  changes the balance between "divergence point creeps later" and "context
  keeps growing" is unexplored.
- **Block alignment (spec §2.3) not exercised in this sweep** — the harness
  supports `--align` but the numbers above are unaligned, matching how naive
  was also measured. Its marginal contribution (spec §2.3: "will be small")
  is not separately reported here.

## Next: Phase 3 (quality measurement) or a Phase 2 follow-up?

Spec's Phase 3 (next-action agreement, retention probes, task success) can
proceed independently of Gate 2's outcome — it's needed regardless. But
given the mechanistic finding above, a short, targeted follow-up before
moving on is worth considering: either a longer-trajectory rerun (more
compaction cycles, to test whether the monotonic-growth advantage eventually
dominates) or a bounded/hybrid variant of append-only that periodically
merges old frozen segments. Neither is required by the spec's Phase 2 scope
as written, but both follow directly from this phase's own data.
