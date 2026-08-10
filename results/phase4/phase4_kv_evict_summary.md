# Phase 4.1 Summary — KV-Space Eviction Baseline

Status: implementation complete, full sweep run. First of Phase 4's sub-parts
(§4.1 kv_evict; §4.2 attn_evict, §4.3 offload, §4.4 hybrid, §4.5 the Pareto plot
and Gate 4 are not built yet).

## What was built

- `src/agentkv/policies/kv_evict.py` — `KVEvictPolicy`: keeps the anchor (this
  project's stand-in for StreamingLLM's "attention sink," per existing
  convention) plus the newest `window_turns` raw turns verbatim; everything
  older is **deleted with no replacement text** — no LLM summarizer call at
  all, unlike `naive`/`append_only`.
- `configs/policies/kv_evict.yaml` — `threshold_pct_of_window: 0.6`,
  `window_turns: 20`.
- `experiments/phase4_kv_evict.py` — extends `phase2_layout.py`'s naive-vs-
  append_only sweep to a three-way naive/append_only/kv_evict comparison,
  3-way interleaved per trajectory, through the identical measurement harness
  (same `divergence.py` theory check, same rigor/clock-lock instrumentation).
- `results/phase4/phase4_kv_evict_steps.parquet`, `..._events.parquet`,
  `..._comparison.png` — same shapes as Phase 2's artifacts, one more policy.

## What spec §4.1 asks for

> "StreamingLLM-style: retain the first few 'attention sink' tokens plus a
> sliding window; discard the middle *in KV space*. Text is never rewritten, so
> there is zero re-prefill. Cost is near-free; the question is what it costs in
> quality."

## The measurement question, decided before writing any code

Before implementing this, direct inspection of the installed vLLM 0.8.5
package (`vllm/distributed/kv_transfer/kv_connector/v1/base.py`) confirmed a
real architectural mismatch: "zero re-prefill" is a property of a *live,
persistent* decode session, where old KV entries are freed from GPU memory
mid-session and attention skips them via position-ID remapping. This project's
entire harness (Phases 0–3) instead sends each agent step as an *independent*
HTTP completions request to vLLM, with reuse coming from vLLM's hash-based
prefix cache matching across those requests. vLLM's real KV extension point
(`KVConnectorBase_V1`) is a per-request save/load interface, not a "keep this
session's cache alive and evict from it" primitive — that only exists at the
L3 (in-vLLM internals) layer spec explicitly defers to Phase 5.

Presented this to the user as a genuine three-way fork (measure via the
existing harness and report honestly / invest in real live-session KV
management / use an analytical-only cost model) rather than picking silently.
**Chosen: measure via the existing harness, report honestly** — reusing all
existing infrastructure, treating whatever the real numbers show as a valid
finding rather than something to route around.

## Results (Qwen/Qwen3-0.6B fallback config, 14 trajectories, threshold=60% of
## window for all three policies)

- Prefill-token reduction, append_only vs. naive: median 0.7%, IQR [-0.9%, 1.9%]
  (n=14) — not significant (p=0.53). (Reproduces Phase 2's Gate 2 numbers
  exactly, a useful consistency check that nothing regressed.)
- **Prefill-token reduction, kv_evict vs. naive: median -19.8%, IQR [-27.2%,
  -7.5%] (n=14) — SIGNIFICANT (Wilcoxon W=2.0, p=0.0017).**

The negative sign means kv_evict *costs more* prefill than naive, not less —
the opposite of its "near-free" theoretical framing, and unlike append_only's
null result, this is a real, statistically significant effect in 13 of 14
trajectories (only traj-010 came out slightly ahead of naive).

| trajectory | naive prefill | append_only prefill | kv_evict prefill |
|---|---|---|---|
| traj-000 | 93276 | 92872 | 111048 |
| traj-001 | 96350 | 93322 | 122885 |
| traj-002 | 85544 | 84042 | 100382 |
| traj-003 | 94264 | 92409 | 119122 |
| traj-004 | 92963 | 93934 | 132813 |
| traj-005 | 64073 | 65551 | 64574 |
| traj-006 | 75844 | 80303 | 111882 |
| traj-008 | 83254 | 81056 | 85053 |
| traj-009 | 84073 | 84495 | 89181 |
| traj-010 | 75781 | 75237 | 74895 |
| traj-011 | 81426 | 86486 | 98165 |
| traj-012 | 91000 | 90348 | 111875 |
| traj-013 | 90417 | 89856 | 101011 |
| traj-014 | 83854 | 80923 | 189365 |

traj-014 is a clear outlier (189,365 vs. 83,854 for naive, more than double) —
worth a closer look before citing as typical, but no theory mismatch or error
fired for it, so it's real data, not a bug artifact.

## Why: the mechanism, confirmed from the event log

`results/phase4/phase4_kv_evict_events.parquet` shows exactly why: kv_evict fired
**150 eviction events** across the sweep, vs. 115 (naive) and 116
(append_only) — about 30% more — and each event's mean reprefill cost was also
higher (6,177 tokens vs. 5,160/5,092). Both effects compound. This is exactly
the mechanism predicted before running anything: naive/append_only only
compact once their retirement threshold fires (a discrete, relatively rare
event — cutting ~50% of turns buys a long gap before the next one), while
kv_evict's fixed-size sliding window, once full, drops its single oldest turn
and re-fires *every subsequent step* — and each of those firings invalidates
most of the cached window under vLLM's hash-based prefix caching, because the
window's front boundary shifts by one turn's worth of tokens every time.
Measured cache behavior matches the predicted mechanism exactly; this is not a
surprise, it's a confirmation.

## What this shows

**KV-space eviction's real advantage (truly zero re-prefill) requires live
KV-cache management this project's request-based harness cannot express — and
approximating it with shorter *text* prompts sent to independent requests
doesn't just fail to capture the benefit, it actively performs worse than the
token-space baseline it was meant to beat.** This is a clean, well-measured
negative result in the same spirit as Gate 2's: the theoretical claim
("near-free") and the measured reality (significantly more expensive than the
status quo) point in opposite directions, and that gap is itself the finding.
It also sharpens what Phase 4.3 (`kv/offload.py`, via the real
`KVConnectorBase_V1` interface) and a genuine live-session eviction attempt
would need to prove to be worth building: this measurement is evidence *for*
investing in the real mechanism, not evidence that KV-space eviction is a bad
idea in general.

## Known limitations

- **Single configuration** — `window_turns=20`, one threshold. A larger window
  would fire less often (fewer, cheaper-per-event evictions) at the cost of a
  bigger anchor+window prompt; not swept here.
- **traj-014's outlier** (189k vs. the next-highest ~133k) wasn't investigated
  further — worth checking whether it reflects an unusually long or bursty
  trajectory before treating the median/IQR as fully representative.
- **No quality measurement yet** — spec's own framing ("the question is what
  it costs in quality") is unanswered for kv_evict; Phase 3's agreement/
  retention/task-success infrastructure could be pointed at it directly, not
  done here.
- **Fallback (0.6B) model**, consistent with Phase 2/3's established
  precedent for this trajectory set.

## Next

§4.2 (`attn_evict.py`, attention-score eviction) has its own real feasibility
question — does vLLM 0.8.5 expose per-token attention weights through any
supported API, or does scoring require the same kind of internals work §4.1's
"zero re-prefill" claim did? Worth investigating before writing code, the same
way §4.1 was. Not started here.
