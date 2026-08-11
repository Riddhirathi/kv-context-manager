# Rehearsal — the three-figure deck (AGENTKV_SPEC.md §6.2)

Figures: `deck_1_cliff.png`, `deck_2_fix.png`, `deck_3_pareto.png` (this
folder). All three use the fallback model (`Qwen/Qwen3-0.6B`) and, for
panels 1-2, the same demo trajectory (`traj-000`) — see `demo_deck.py`'s
docstring for why.

## 60-second version

Problem -> cliff figure -> one-line mechanism -> Pareto plot -> reproducibility line.

1. "Long-horizon agents keep re-sending their entire conversation history to
   the model on every single step. Nobody measures what that costs."
2. **[show `deck_1_cliff.png`]** "Here is a cost nobody is measuring. This is
   one ~164-step agent trajectory, naive compaction, cumulative tokens
   re-prefilled on the y-axis. Every dashed line is a compaction event —
   watch the staircase jump. By the end, this one trajectory has reprefilled
   93,000 tokens just to keep going."
3. "The fix is simple: don't treat every turn the same. Route by value —
   keep small turns verbatim, evict large tool dumps nobody re-reads,
   summarize the rest — and protect a verbatim tail so the prefix cache
   doesn't blow up right when the model needs its most recent context most."
4. **[show `deck_3_pareto.png`]** "Here is proof I didn't cheat. Five
   policies, five seeds each, cumulative prefill cost on the x-axis. My
   hybrid policy sits furthest left of the four complete policies — cheapest,
   not just different."
5. "And it's reproducible on an 8GB laptop."

## 5-minute version

Same shape, more room to show the work and the honesty.

1. **Problem (30s).** Same opener as the 60-second version.

2. **The cliff — `deck_1_cliff.png` (60s).** "This is naive compaction on one
   trajectory: summarize-and-replace whenever the window fills up. Notice the
   staircase isn't smooth — it's flat, flat, flat, then a vertical jump. Every
   jump is a compaction event where the prefix cache gets invalidated and the
   model has to reprefill from the divergence point forward. Across the
   project's full 14-trajectory sweep, naive spends a median of ~63% of all
   its prefill compute on these events alone, not on making progress."

3. **The mechanism (60s).** "Four real compaction strategies live in this
   repo: naive (LLM-summarize the middle when full), append-only (same, but
   never re-summarize what's already been summarized), KV-evict (drop a
   sliding window outright, no LLM call), and hybrid — mine. Hybrid routes
   *per segment*: small turns stay verbatim, large tool outputs get evicted
   outright since they're usually read once and never again, everything else
   gets summarized, and the most recent turns are always protected verbatim
   so the part of the cache the model actually leans on next never gets
   touched."

4. **The fix — `deck_2_fix.png` (45s).** "Same trajectory, all four policies
   overlaid. On this one trajectory: naive ends at 93,276 cumulative prefill
   tokens, append-only close behind at 92,872, KV-evict is *worse* at
   111,048 — evicting outright looks cheap per-event but forces far more
   total reprefill work — and hybrid ends at 80,116. That's not a fluke of
   one trajectory: across the full 14-trajectory, 5-seed sweep the median
   figures are naive 78,436, KV-evict 106,030, hybrid 70,026 — a paired
   Wilcoxon test confirms hybrid's reduction is significant."

5. **The Pareto frontier — `deck_3_pareto.png` (60s).** "Cost on the x-axis,
   task success on the y-axis, one point per policy. Hybrid sits furthest
   left among the four complete policies — cheapest. The fifth point,
   `none` — never compact, ever — is drawn hollow because it's incomplete:
   it blows through the model's context window before any trajectory
   finishes, on every single one. That's not swept under the rug; it's the
   point. No compaction can't even survive most of a long trajectory here."

6. **The honest part (45s).** "Task success is 0 out of 5 for *every*
   policy on this task/model — that's not a policy failure, it's a floor
   effect: the underlying model can't reliably finish this 50-transaction
   ledger task regardless of what context it's given, naive or hybrid. So
   this Pareto plot proves a cost *ranking*, not yet a cost/quality
   *tradeoff* — I'm not claiming hybrid buys better answers, only that it
   buys the same (currently zero) answers for less. What hybrid *does* show
   underneath that floor: it gets the model through 110 of its ~130 tool-call
   budget with only 0.6 ledger instructions left unprocessed on average,
   versus ~93 calls and ~9 left unprocessed for the other policies — real
   signal that better routing lets the model progress further, even though
   nothing here clears the success bar yet."

7. **Close (30s).** "Every number on these three figures comes from a
   committed parquet file, never hand-edited. This deck itself is one
   command — `python experiments/demo_deck.py`, no GPU needed, it just
   reassembles already-measured data. All of that data was measured on one
   $1,500 laptop GPU."
