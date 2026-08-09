# Phase 3 Summary — Quality Measurement (Gate 3)

Status: all three required legs built and run. Gate 3's literal bar is met (a
(prefill cost, task success) point exists for each policy, with error bars) — but
the point is not a flattering one, and this document says so directly, per spec
§3.4: "Report all three. Agreement is cheap and dense; probes are diagnostic; task
success is the honest bottom line. Never report only the flattering one."

Full detail for each leg lives in its own summary:
- [`results/phase3_agreement_summary.md`](phase3_agreement_summary.md) — §3.1
- [`results/phase3_probes_summary.md`](phase3_probes_summary.md) — §3.2
- [`results/phase3_ledger_summary.md`](phase3_ledger_summary.md) — §3.3

This document synthesizes the three against Gate 3 itself; it does not repeat their
debugging histories.

## Gate 3

> "For every policy, you can produce a (prefill cost, task success) point with
> error bars."

| policy | prefill-token reduction vs. naive (Phase 2) | task success rate (Phase 3.3) |
|---|---|---|
| naive | — (baseline) | 0/5 (95% Wilson CI: 0%–43%) |
| append_only | median 0.7%, IQR [-0.9%, 1.9%] (n=14, not significant, p=0.53) | 0/5 (95% Wilson CI: 0%–43%) |

The point exists for both policies. Both coordinates say the same thing: **no
measurable difference between the two policies**, on either cost or success. append-only
does not cost less (Gate 2 already found this) and does not succeed more or less
often at the actual task (this phase's finding). The 0/5 task-success figure was
confirmed on both the fallback (0.6B) and primary (1.7B) models — see
`phase3_ledger_summary.md` — so this is not an artifact of model choice.

## The three legs, together

**Agreement (§3.1, cheap and dense)** — 145 decision points/policy, only the first
~30–40% of each trajectory (context ceiling on the uncompacted baseline cut runs
short): append_only 0.800 tool-name match / 0.593 args match, naive 0.779 / 0.572.
A small, untested-for-significance edge for append_only, in the direction Gate 2
hoped for.

**Retention (§3.2, diagnostic)** — both policies retain perfectly at depth 10, then
collapse to near-zero by depth 50 and stay there. append_only shows **no** consistent
advantage over naive (mixed: naive better at depth 30, append_only better at depth
70/90, identical floor at depth 50). Diagnosis: retention depends on whether the
shared `LLMSummarizer` preserved a fact when it first retired that turn — a
summarization-fidelity question, not a layout question. This is *why* the ledger
result below looks the way it does.

**Task success (§3.3, the honest bottom line)** — 0/5 for both policies, and this
was retested on both the fallback (0.6B) and primary (1.7B) models: still 0/5 on
primary too, and the larger model's *mechanical* task-following (how much of the
50-instruction queue it actually attempted before submitting) was measurably worse,
not better, than the smaller model's — three of five 1.7B episodes submitted a
final report almost immediately after the scripted bootstrap ended, with 37–44 of 50
instructions still unprocessed. Model size, in the [0.6B, 1.7B] range tested, did not
rescue this task.

The throughline across all three: **layout policy (naive vs. append_only) does not
move quality, in any of the three ways this phase measured it.** Phase 2 already
found append-only doesn't reliably reduce cost either. Put together, Phases 2 and 3
say the same thing from two different angles: this project's specific append-only
implementation, measured on this hardware/model/trajectory set, has not yet
demonstrated the win it was designed to produce — on cost *or* on quality.

## Why, mechanistically

Both `naive` and `append_only` retire turns through the exact same `LLMSummarizer`
at the exact same threshold and retire_fraction (`configs/policies/*.yaml`).
Append-only's real, confirmed contribution is structural: monotonic prefix growth,
verified cleanly in Phase 2 (14/14 trajectories). But once a fact or a piece of
state is swept into a summary, whether that summary later sits in a
monotonically-growing `frozen` list (append_only) or gets re-swept by a dumber
re-summarization pass (naive) doesn't change whether the *specific value* survived
the summarization step itself. Layout changes where cached text sits, not what
information is in it. Quality, in every form measured this phase, turns out to
depend almost entirely on the latter.

## Known limitations (aggregated — see each leg's own summary for the rest)

- **Model**: §3.1/3.2 ran on the fallback (0.6B) model per Phase 2's established
  precedent (not retested on primary — §3.3's result suggests this likely wouldn't
  change the picture, but that's an inference, not a measurement). §3.3 ran on both
  models; see below.
- **Coverage**: §3.1's context ceiling limits agreement measurement to early/mid
  trajectory; §3.3 ran a single task configuration, 5 seeds per model.
- **No cross-leg significance testing** — the "no measurable difference" framing
  above is a description of what was observed (small samples, some legs not
  significance-tested at all), not a proven null result.
- **Gate 2's action-agreement clause is still only partially closed**: Gate 2 asked
  for "equal or better action agreement" as part of its own bar; §3.1's small
  append-only edge is suggestive but not tested for significance against Gate 2's
  cost finding.
- **§3.3's one-harness-per-model limitation**: the bootstrap/prompt setup was tuned
  empirically against the 0.6B model, then reused as-is for 1.7B. The 1.7B model's
  worse mechanical performance could reflect that harness mismatch rather than a
  clean capability comparison — see `phase3_ledger_summary.md`.

## Next

One path forward, not decided here: **investigate summarization fidelity directly**
(e.g. does `max_summary_tokens` or `retire_fraction` move retention/success more than
layout policy does?) — the mechanistic hypothesis above is specific and checkable,
not just a plausible story. The model-size question that was open in the previous
version of this document is now resolved (§3.3 retested on primary; still 0/5,
mechanically worse, not better) — this doesn't rule out a harness re-tuned for 1.7B
behaving differently, but that's a smaller, more speculative follow-up than the
original question was.

This doesn't block writing up Phases 0–3 as they stand: the negative results here are
real findings, not gaps, and this project's own stated ethos (§3.4, and the
Phase 2/Phase 3.2/Phase 3.3 summaries before this one) is to report them plainly
rather than keep searching for a flattering cut of the data.
