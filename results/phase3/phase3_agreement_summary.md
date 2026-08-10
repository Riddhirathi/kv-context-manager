# Phase 3.1 Summary — Next-Action Agreement

Status: implementation complete, first full sweep run (§3.2 retention probes and §3.3
the verifiable ledger task — the other two legs Gate 3 needs — are not built yet).

## What was built

- `src/agentkv/bench/agreement.py` — `is_decision_point`, `build_decision_prompt_ids`,
  `parse_action`, `ACTION_JSON_SCHEMA`, `score_agreement`, `AgreementEventCollector`.
- `src/agentkv/serving/engine.py` — `VLLMEngine.complete_text` gained an optional
  `guided_json` parameter (vLLM's per-request grammar-constrained decoding; no server
  restart, still the plain `/v1/completions` endpoint).
- `experiments/phase3_agreement.py` — replays every trajectory under `naive` and
  `append_only` exactly as `phase2_layout.py` does, and at each decision point elicits
  a counterfactual action from (a) the raw uncompacted context and (b) each policy's
  actual compacted context, scoring how often they agree.
- `results/phase3_agreement.parquet` — one row per decision point per policy.

## What spec §3.1 asks for

> "For each step in a replayed trajectory, compute the model's next action given (a)
> full uncompacted context and (b) compacted context. Score exact tool-name match,
> argument match, and a semantic similarity fallback. Report agreement rate per policy."

## Elicitation: two rounds of empirical correction

The trajectories only ever held the strong recorder model's actions — nothing in the
codebase previously asked the *measurement* model (Qwen3-0.6B) to produce its own
action. Getting that to work reliably took two real, GPU-verified fixes before the
sweep below is trustworthy:

1. **Plain free-text continuation**, betting that prior in-context assistant turns
   (already rendered as `[{"name": ..., "args": {...}}]` by `context/layout.py`) would
   be enough precedent for the model to imitate. Mostly true, except at a trajectory's
   *first* decision point, where there's no precedent yet — the model degenerated into
   a 300+-word non-JSON ramble (traj-000 step 2).
2. Fixed that with a fixed one-shot exemplar spliced in after the anchor — which
   surfaced a *worse* failure: the model predicted immediate end-of-sequence right
   after the primed cue (`finish_reason=stop`, 1 completion token, empty text).
   Forcing generation past that (`min_tokens`) just produced plausible prose that
   still never closed a JSON call.

Root cause: nothing was forcing the model to actually emit valid JSON — it was free
to stop or ramble. `guided_json` (grammar-constrained decoding, forwarded through
`complete_text`) fixes that structurally: every token is constrained to
`ACTION_JSON_SCHEMA` from the first token on, so an immediate stop or a prose-only
answer are no longer even reachable. The one-shot exemplar is still needed on top of
that — grammar constraints only fix *validity*, not *semantics*: without it, the same
example-free first decision point still reliably produces well-formed JSON, but
hallucinates a nonexistent tool and repeats a garbage key until truncated
(`{"name": "oncall_engineer", "args": {"tool_call": "tool_call_1", ...}}`).

## The context ceiling — a real, expected limitation

`build_decision_prompt_ids` renders full history verbatim for condition (a) — by
design, since "full uncompacted context" is the whole point of the comparison. On the
first full sweep this crashed the run: at step ~139 of traj-000, condition (a)'s
prompt reached 17,826 tokens, over the fallback model's 16,384-token
`max_model_len`. This is Phase 0/1's "cliff" (the exact reason compaction exists)
showing up inside a different experiment, not a bug. Fixed by checking
`len(full_prompt_ids) + max_action_tokens` against `max_model_len` before each call and
stopping that trajectory's agreement evaluation once it would be exceeded (logged
per-trajectory as `context_exceeded_at_step`, never silently) — `naive`/`append_only`'s
own *compacted* contexts stay bounded by construction and remain unaffected; only the
(a)-vs-(b) comparison becomes unanswerable past that point.

In this run, **all 14/14 trajectories hit that ceiling**, between step 45 and step 69
(out of ~161–169 total turns per trajectory) — meaning only the first ~30–40% of each
trajectory's steps were reachable for agreement measurement. This is a real scope
limit on what follows, not a sampling choice.

## Results (Qwen/Qwen3-0.6B fallback config, max_model_len=16384, 14 trajectories,
## 145 decision points per policy before any trajectory hit the context ceiling)

| policy | n | tool_name_match | args_match | mean_semantic_similarity |
|---|---|---|---|---|
| append_only | 145 | 0.800 | 0.593 | 0.821 |
| naive | 145 | 0.779 | 0.572 | 0.818 |

append_only comes out slightly ahead on both exact-match scores — consistent with
(not yet statistically tested against) Gate 2's "at equal or better action agreement"
clause. No significance test has been run on this difference yet; treat it as
directional, not confirmed.

Per-trajectory decision points evaluated and where the context ceiling hit:

| trajectory | turns | decision points evaluated | full parse failures | naive parse failures | append parse failures | context exceeded at step |
|---|---|---|---|---|---|---|
| traj-000 | 164 | 11 | 0 | 1 | 1 | 59 |
| traj-001 | 169 | 11 | 0 | 0 | 0 | 69 |
| traj-002 | 166 | 13 | 1 | 0 | 0 | 54 |
| traj-003 | 163 | 12 | 0 | 0 | 0 | 62 |
| traj-004 | 165 | 9 | 3 | 3 | 3 | 53 |
| traj-005 | 164 | 16 | 0 | 0 | 1 | 61 |
| traj-006 | 162 | 10 | 2 | 2 | 2 | 55 |
| traj-008 | 161 | 9 | 0 | 0 | 0 | 68 |
| traj-009 | 162 | 6 | 0 | 0 | 0 | 62 |
| traj-010 | 164 | 8 | 0 | 0 | 0 | 61 |
| traj-011 | 161 | 7 | 0 | 0 | 0 | 54 |
| traj-012 | 161 | 6 | 0 | 0 | 0 | 52 |
| traj-013 | 164 | 11 | 2 | 2 | 2 | 57 |
| traj-014 | 165 | 16 | 1 | 1 | 1 | 45 |

(traj-007 excluded, same as Phase 2 — short/interrupted recording.)

Parse-failure rate overall: 9/145 (6.2%) on the full-context condition, 9/145 (6.2%)
naive, 10/145 (6.9%) append_only — non-zero but low, concentrated in a few
trajectories (traj-004, traj-006, traj-013) rather than spread evenly, worth a closer
look before this becomes a citable number.

## Known limitations

- **Not yet a full Gate 3 deliverable.** This is one of three required legs (§3.1
  agreement only); §3.2 retention probes and §3.3 the verifiable ledger task are not
  built.
- **No significance test yet** on the naive-vs-append_only agreement gap.
- **Only the first ~30–40% of each trajectory is covered**, because of the context
  ceiling above — this agreement rate describes early-to-mid trajectory behavior, not
  the full 100-step horizon Gate 2/3 language implies.
- **Parse failures are excluded, not imputed as disagreement** — `tool_name_match`
  requires both sides to parse successfully, so a policy that reliably fails to
  produce valid JSON would show *higher* apparent agreement with itself being wrong
  in the same way, not lower. Worth a dedicated look if the parse-failure rate rises
  in future runs.
- **`AgreementEventCollector`'s parquet rows have no trajectory_id column**, matching
  this codebase's existing `StepRecord`/`CompactionEventRecord` convention (flat,
  pooled-by-policy records) — the per-trajectory table above was reconstructed from
  this run's console log, not from the parquet file itself.

## Next: §3.2 retention probes, §3.3 the ledger task, or fix trajectory length first?

The context-ceiling finding suggests a real fork: build §3.2/§3.3 next as planned, or
first extend/regenerate trajectories so agreement (and Phase 2's own reprefill
numbers) can be measured over a fuller horizon before investing more in this
elicitation harness. Both are reasonable; this is an open question for the next step,
not a decision made here.
