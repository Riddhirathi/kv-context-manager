# Phase 3.2 Summary — Retention Probes

Status: implementation complete, first full sweep run (§3.3 the verifiable ledger task
— the last leg Gate 3 needs — is not built yet).

## What was built

- `src/agentkv/bench/tasks/probes.py` — `generate_facts`, `inject_facts`,
  `build_query_prompt_ids`, `ANSWER_JSON_SCHEMA`, `check_retention`, `RetentionRecord`,
  `RetentionEventCollector`.
- `experiments/phase3_probes.py` — for each trajectory: injects 5 synthetic facts at
  controlled depths into the first 100 turns, replays that under both `naive` and
  `append_only` (each policy's own real compaction, exactly as `phase2_layout.py`/
  `phase3_agreement.py` do it), then queries each fact once against each policy's
  final compacted context.
- `results/phase3_probes.parquet` — one row per (fact, policy, trajectory) query.

## What spec §3.2 asks for

> "Inject synthetic facts at controlled depths in the trajectory ('the account ID is
> X', 'the user prefers Y'). At step 100, query for each. Produces a retention-vs-depth
> curve per policy. This is how you show precisely what each policy forgets."

Facts: 6 templates (account ID, user contact preference, ticket number, priority
level, escalation contact, resolution deadline), deterministic per (trajectory, seed).
5 are sampled per trajectory and injected at turns 10, 30, 50, 70, 90 of the first 100
turns, then queried once, all at the same point, right after replay finishes — depth
is defined as `100 - insertion_step`, so depth 90 is the oldest/deepest fact and depth
10 is the most recent.

## Elicitation: the same class of bug as Phase 3.1, twice

1. **Free-text QA, primed with a one-shot exemplar and `min_tokens`** (the same recipe
   that worked for 3.1's tool-call elicitation) mostly failed here too: the base
   trajectory context is entirely `<think>...</think>`-wrapped tool-call turns, so the
   model kept re-entering `<think>` mode instead of answering directly, exhausting the
   token budget before ever producing an answer (`0/4` retained in the first sanity
   check, including at depth 10 — implausibly low, a tell that this was an elicitation
   bug, not a real result). Fixed the same way 3.1 was: `guided_json`
   (`ANSWER_JSON_SCHEMA`, `{"answer": "..."}`) forces a direct answer from the first
   token.
2. **A second, more subtle bug survived that fix**: the one-shot exemplar's example
   answer ("The ticket number is TCK-1029.") happened to share a fact *kind*
   (`ticket_number`) with one of the six real fact templates. The model started
   literally copying the exemplar's fixed value regardless of what the real injected
   fact was — confirmed by inspecting raw answers, not inferred. `check_retention`
   correctly scored this as "not retained" (the copied value never matches the real
   random one), so it didn't corrupt the retention numbers below, but it meant the
   `ticket_number` fact kind's failures were partly measuring exemplar-copying, not
   compaction. Fixed by replacing the exemplar with an arithmetic example ("What is
   2 + 2?" / "The answer is 4.") that shares no vocabulary with any fact kind —
   structural prevention rather than a per-kind patch.

## Results (Qwen/Qwen3-0.6B fallback config, 14 trajectories, 5 facts each, n=14 per
## depth per policy)

| policy | depth | n | retention_rate |
|---|---|---|---|
| append_only | 10 | 14 | 1.000 |
| append_only | 30 | 14 | 0.286 |
| append_only | 50 | 14 | 0.000 |
| append_only | 70 | 14 | 0.071 |
| append_only | 90 | 14 | 0.071 |
| naive | 10 | 14 | 1.000 |
| naive | 30 | 14 | 0.429 |
| naive | 50 | 14 | 0.000 |
| naive | 70 | 14 | 0.000 |
| naive | 90 | 14 | 0.000 |

## What this shows

Both policies retain perfectly at depth 10 (the fact is still in `live`/unsummarized
context — trivial) and both collapse to near-zero by depth 50 and beyond. The falloff
is sharp, not gradual: full retention at depth 10, under half by depth 30, essentially
none from depth 50 on. This lines up with the ~60%-of-window compaction threshold
firing within the first 30–50 turns of these trajectories (consistent with Phase 2's
measured ~8 events per ~165-turn trajectory) — by depth 50, a fact planted that far
back has typically already been swept into at least one LLM-generated summary, and
the small model's summaries evidently don't reliably preserve exact injected values
(account IDs, ticket numbers) even when they preserve the general shape of the
conversation.

**`append_only` does not show a clear retention advantage over `naive` here** —
mixed and small: naive is *better* at depth 30 (0.429 vs 0.286), append_only is
*better* at depth 70/90 (0.071 vs 0.000), both are identically at floor by depth 50.
This is mechanistically plausible, not just noise: append-only's structural
contribution is *layout* (where cached text sits, monotonic prefix growth — confirmed
in Phase 2), not summarization *fidelity*. Both policies retire turns through the
same `LLMSummarizer` at the same threshold/retire_fraction; once a fact's turn is
swept into a summary, whether that summary later sits in a monotonically-growing
`frozen` list or gets re-swept by naive's dumber re-summarization doesn't change
whether the specific value survived the *first* summarization pass. Layout and
retention are answers to different questions, and this probe is evidence they
don't move together.

## Known limitations

- **n=14 per cell** — enough to see the shape of the falloff, not enough to treat the
  naive-vs-append_only differences at any single depth as significant. No test run yet.
- **Only 5 of 6 fact kinds sampled per trajectory** (one dropped per trajectory to fit
  5 insertion depths) — kind-specific retention differences (e.g. is a ticket number
  harder to retain than a priority level?) aren't broken out here.
- **`check_retention` is a strict substring match** on the exact generated value —
  correct for this project's purpose (did the *specific* injected value survive), but
  means a paraphrased-but-correct answer scores as not retained.
- **Depths only cover the first 100 turns** of trajectories that run ~161–169 turns —
  unlike Phase 3.1, this isn't a hard ceiling (naive/append_only contexts stay bounded
  by construction), just a scope choice matching spec's own "at step 100" language.

## Next: §3.3 the ledger task, or dig into the summarization-fidelity question first?

The mechanistic read above (retention is a summarization-quality question, not a
layout question) is a real, specific, checkable hypothesis this project hasn't tested
directly — e.g. does `max_summary_tokens` or `retire_fraction` change retention more
than layout policy does? Worth flagging as a candidate follow-up alongside proceeding
to §3.3, not a decision made here.
