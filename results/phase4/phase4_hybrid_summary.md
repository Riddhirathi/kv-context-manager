# Phase 4.4 Summary — Hybrid Per-Segment Routing

Status: implementation complete, full 4-way sweep run. Fourth of Phase 4's
sub-parts (§4.1 kv_evict, §4.2 attn_evict, §4.3 offload complete; §4.5 the
Pareto plot and Gate 4 not built yet).

## What was built

- `src/agentkv/policies/hybrid.py` — `HybridPolicy`: routes every candidate
  live turn (outside a protected recent window) to one of three actions —
  **evict** (large tool output, dropped with no replacement, same mechanism
  as `kv_evict.py` but selective rather than blanket-windowed), **keep**
  (small turns, too cheap to be worth touching), or **summarize** (everything
  else, batched into one new frozen `Segment`, same mechanism as
  `append_only.py`).
- `configs/policies/hybrid.yaml` — `threshold_pct_of_window: 0.6`,
  `protect_recent_turns: 10`, `large_tool_output_tokens: 500`,
  `small_turn_tokens: 50`.
- `tests/test_hybrid_policy.py` — 9 tests covering all three routes, the
  protected-recent-window override, the `fired=False` no-op case, and
  constructor validation.
- `experiments/phase4_hybrid.py` — extends the 3-way naive/append_only/
  kv_evict sweep to a 4-way comparison including hybrid, through the
  identical measurement harness. Writes fresh `phase4_hybrid_*` output
  filenames rather than reusing `phase4_kv_evict_*`, to avoid
  `RecordCollector.flush()`'s append-not-overwrite behavior silently mixing
  a 3-policy run with a 4-policy one.

## What spec §4.4 asks for

> "The contribution. Per segment, choose one of three actions: keep verbatim
> ...; KV-evict ...; textually summarize .... Route by estimated value:
> predicted future attention mass × segment size × recompute cost. Start
> with a hand-tuned heuristic. A learned policy is a stretch goal, not a
> requirement."

Spec's own suggested routing signal ("predicted future attention mass") is
exactly what §4.2 found infeasible to compute affordably on this hardware
(see `phase4_attn_evict_summary.md`). Spec's own fallback bar — "start with
a hand-tuned heuristic" — is used instead: a `tool` turn's rendered size
stands in for predicted future value, reusing the same signal already
validated twice in this project (`kv_evict.py`'s spirit and
`kv/offload.py`'s `classify_decoded_text`, both keyed on spec's repeated
observation that large tool outputs are usually referenced once and never
again).

## Results (Qwen/Qwen3-0.6B fallback config, 14 trajectories, threshold=60%
## of window for all four policies)

- Prefill-token reduction, append_only vs. naive: median 0.7%, IQR [-0.9%,
  1.9%] — not significant (p=0.53).
- Prefill-token reduction, kv_evict vs. naive: median -19.8%, IQR [-27.2%,
  -7.5%] — SIGNIFICANT, but in the *wrong* direction (costs more).
- **Prefill-token reduction, hybrid vs. naive: median 16.6%, IQR [15.5%,
  20.1%] (n=14) — SIGNIFICANT (Wilcoxon W=0.0, p=0.0011).**

Unlike append_only's inconsistent null result and kv_evict's regression,
hybrid wins on **every single trajectory** — not just a favorable median
masking mixed results:

| trajectory | naive prefill | append_only prefill | kv_evict prefill | hybrid prefill |
|---|---|---|---|---|
| traj-000 | 93276 | 92872 | 111048 | 80116 |
| traj-001 | 96350 | 93306 | 122885 | 81067 |
| traj-002 | 85544 | 84042 | 100382 | 68666 |
| traj-003 | 94264 | 92409 | 119122 | 79071 |
| traj-004 | 92963 | 93934 | 132813 | 73576 |
| traj-005 | 64073 | 65535 | 64574 | 54343 |
| traj-006 | 75844 | 80303 | 111882 | 62631 |
| traj-008 | 83174 | 81118 | 85053 | 63433 |
| traj-009 | 84073 | 84495 | 89181 | 69781 |
| traj-010 | 75781 | 75221 | 74895 | 59538 |
| traj-011 | 81426 | 86496 | 98165 | 68923 |
| traj-012 | 91000 | 90348 | 111875 | 77290 |
| traj-013 | 90417 | 89856 | 101011 | 72100 |
| traj-014 | 83862 | 80902 | 189365 | 70302 |

Aggregate totals across all 14 trajectories: naive 1,192,047 tokens, hybrid
980,837 tokens — a 17.7% pooled reduction, consistent with the per-trajectory
median.

## Why: the mechanism, confirmed from the event log

`results/phase4/phase4_hybrid_events.parquet` explains the win on both axes
that determine total prefill cost — how often compaction fires, and how
expensive each firing is:

| policy | events | mean tokens reprefilled/event | mean tokens saved/event | mean reprefill:saved ratio |
|---|---|---|---|---|
| naive | 115 | 5,160 | 4,756 | 1.12 |
| append_only | 116 | 5,092 | 4,671 | 1.13 |
| kv_evict | 150 | 6,177 | 3,676 | 11.93 |
| hybrid | 97 | 3,799 | 5,692 | 0.71 |

Hybrid fires **fewer times** than any other policy (97 vs. 115–150), because
keep-verbatim routing removes small turns from ever needing compaction at
all, and evict routing removes large tool outputs without a summarizer round
trip. And each firing is **individually cheaper**: mean tokens reprefilled
per event (3,799) is the lowest of any policy, while mean tokens saved per
event (5,692) is the highest — the opposite of kv_evict, whose sliding
window saves the least per event (3,676) while reprefilling the most (6,177),
giving it a reprefill:saved ratio of ~12x. Hybrid's ratio (0.71, the only
policy below 1.0) means each compaction event saves more than it costs to
reprefill later — naive and append_only roughly break even (~1.12–1.13), and
kv_evict loses badly. This is a direct, measured explanation for why routing
by content type beats both "compact everything past a threshold" (naive/
append_only) and "always evict the sliding window's oldest turn" (kv_evict).

## What this shows

**This is the first policy in the entire project with a clean, statistically
significant, positive prefill-cost reduction versus naive** — better than
append_only's null result and the opposite direction of kv_evict's
regression, with a notably tight IQR (15.5–20.1%) rather than a wide or
outlier-driven spread. The mechanism is not mysterious or a training-set
artifact: selective, content-aware routing reduces both how often compaction
fires and how costly each firing is, exactly per spec's framing ("route by
estimated value"), even with a hand-tuned heuristic standing in for the
infeasible attention-based signal.

## Known limitations

- **Single configuration** — one threshold, one protected-window size, one
  pair of size cutoffs (`large_tool_output_tokens=500`,
  `small_turn_tokens=50`). Not swept; the heuristic's sensitivity to these
  constants is unmeasured.
- **No quality measurement yet** — same gap as every other Phase 4 sub-part
  so far. Hybrid's evict route is exactly as lossy/unrecoverable as
  kv_evict's for the turns it drops; nothing here measures whether that loss
  is acceptable. Phase 3's agreement/retention/task-success infrastructure
  could be pointed at hybrid directly, not done here.
- **Heuristic reuses an existing signal, not a new one** — the "large tool
  output" routing rule is the same signal `kv_evict.py` and
  `kv/offload.py` already used; the win comes from applying it *selectively*
  (mixed with keep/summarize) rather than *uniformly* (kv_evict's blanket
  windowed eviction), not from a materially different signal.
- **Fallback (0.6B) model**, consistent with every other Phase 4/3/2
  trajectory set.

## Environment check post-sweep

`nvidia-smi --query-compute-apps` shows no orphaned GPU processes. Full test
suite: 136 passed. `ruff check src tests experiments`: all checks passed.
`mypy src`: no issues found in 30 source files.

## Next

§4.5 (the Pareto plot / Gate 4 itself) needs task-success/quality data
alongside these prefill-cost numbers — spec's stated axes are "X = cumulative
prefill tokens, Y = task success rate." Not started here; would need Phase
3's quality-measurement infrastructure pointed at all four (or five, with
offload) policies swept in this phase.
