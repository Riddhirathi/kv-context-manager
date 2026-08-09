# Phase 3.3 Summary — Synthetic Verifiable Ledger Task

Status: implementation complete, full sweeps run on both the fallback (0.6B) and
primary (1.7B) models — the last of Gate 3's three required legs (§3.1 agreement,
§3.2 retention probes, §3.3 this task).

## What was built

- `src/agentkv/bench/tasks/ledger.py` — `LedgerEnv` (a live, stateful multi-account
  ledger with a queued-instruction interface: `get_next_instruction`, `deposit`,
  `withdraw`, `transfer`, `get_balance`, `submit_final_report`), `action_schema()`
  (tool-name-enum-restricted `guided_json` schema), `LedgerRunRecord`,
  `LedgerRunCollector`.
- `experiments/phase3_ledger.py` — the only *live* agent loop in this project (every
  other experiment replays a recorded trajectory). At each step the measurement model
  picks its next tool call, the environment executes it for real, and the growing raw
  turn history is fed through `naive`/`append_only` exactly like every other
  experiment here.
- `results/phase3_ledger.parquet` (fallback/0.6B) and `results/phase3_ledger_primary.parquet`
  (primary/1.7B) — one row per (seed, policy) episode, same schema, separate files
  since the schema doesn't carry a model field.

## What spec §3.3 asks for

> "Build one long-horizon task a 1.7B model can actually complete, with programmatic
> verification: e.g. an inventory/ledger environment where the agent makes ~100 tool
> calls that mutate state, and the final reported balance is checked against ground
> truth. Binary reward. This is your real 'task success rate' number."

Two balance dicts are tracked deliberately: `_actual_balances` (mutated only by the
agent's own real tool calls) and `_reference_balances` (computed once by correctly
simulating the full queue). `verify()` checks the agent's *reported* final balances
against the reference, not against whatever it actually did — success means it
correctly executed and remembered the entire ~50-transaction queue, not merely that
it accurately reported its own mistakes.

## Getting the agent loop to actually work: three real, GPU-verified failures

This was the hardest of the three Phase 3 legs, because unlike 3.1/3.2 there is no
recorded trajectory giving the model *any* in-context precedent to imitate — the
model has to generate an entire ~100-step episode from nothing.

1. **Cold-start tool selection.** With zero in-context precedent, the model picked
   `submit_final_report` as its very first action regardless of what the system
   prompt said to do, with garbage args (a list of account names, not a balances
   dict) — confirmed by direct inspection of the raw completion. An explicit
   "Start now by calling get_next_instruction" instruction alone did not fix this.
2. **One example isn't enough to convey "keep repeating."** Bootstrapping the episode
   with one real, scripted fetch-then-apply cycle (executed by the driver, not the
   model, using the real environment) got the model to complete that one cycle
   correctly, then jump straight to `submit_final_report` instead of continuing.
   Fixed by bootstrapping **three** real cycles before handing control to the model,
   plus an explicit "do NOT call submit_final_report early" line in the system
   prompt — standard few-shot practice (one example rarely establishes a repeat
   pattern reliably), confirmed here empirically rather than assumed.
3. **Tool-name hallucination without an enum.** Reusing `bench/agreement.py`'s
   generic `ACTION_JSON_SCHEMA` (unconstrained `name` field) let the model
   occasionally invent tools that don't exist. `LedgerEnv.action_schema()` instead
   enums `name` to the environment's real tool list, so `guided_json` rules this out
   structurally, the same way it ruled out invalid JSON in 3.1/3.2.

With all three fixes, episodes run correctly: e.g. seed=0 processed all 50 queued
instructions in order before submitting a (still incorrect) final report.

## Infrastructure incident: WSL2 GPU passthrough hang

Mid-sweep, `vllm serve` started hanging indefinitely during CUDA initialization —
observed as a subprocess frozen at 0% GPU utilization with zero CPU-time growth,
reproduced four times in a row across both the primary (1.7B) and fallback (0.6B)
models. `dmesg` showed the actual cause: repeated
`misc dxg: dxgk: dxgkio_query_adapter_info: Ioctl failed: -22` — a WSL2
GPU-paravirtualization driver desync, not a code or model issue (likely triggered by
the many rapid vLLM server start/stop cycles across this session's debugging).
Fixed with `wsl --shutdown` from Windows (has to be run from the Windows side, not
from inside WSL) followed by a fresh WSL boot, which cleared the error and restored
normal GPU access.

## Model choice: both fallback (0.6B) and primary (1.7B) now tested

Spec explicitly asks for "a task a 1.7B model can actually complete," and
`configs/model.yaml`'s primary model *is* `Qwen/Qwen3-1.7B` — but it reproducibly hung
during boot (the incident above) every time it was first attempted for this specific
workload, including after the WSL restart resolved it for the 0.6B model. The first
full sweep below therefore ran on the fallback (0.6B) model instead — the same
pragmatic deviation Phase 2 already made and documented, here forced by an
infrastructure reliability problem rather than a context-window one. A second sweep
was then run on the primary (1.7B) model once the same `wsl --shutdown` fix was
confirmed to also resolve *its* boot hang, resolving the open question the first
version of this document left unanswered.

## Results: fallback (Qwen/Qwen3-0.6B, max_model_len=16384)

50 transactions, 5 accounts, 5 seeds, 130 tool-call budget, 3 bootstrap cycles.

| policy | n | success_rate | mean_tool_calls | stopped_reasons |
|---|---|---|---|---|
| naive | 5 | 0.000 | 93.4 | {'submitted': 5} |
| append_only | 5 | 0.000 | 91.6 | {'submitted': 5} |

**0/5 success for both policies.** Every episode ran to completion in the mechanical
sense — the agent always eventually called `submit_final_report` (`stopped_reason`
is `submitted` in all 10 runs, never a budget or context-limit failure) — but never
once reported the exactly-correct final balances.

| seed | policy | tool_calls | instructions_remaining at submit | success |
|---|---|---|---|---|
| 0 | naive | 120 | 0 | False |
| 0 | append_only | 120 | 0 | False |
| 1 | naive | 48 | 27 | False |
| 1 | append_only | 39 | 31 | False |
| 2 | naive | 95 | 3 | False |
| 2 | append_only | 95 | 3 | False |
| 3 | naive | 102 | 0 | False |
| 3 | append_only | 102 | 0 | False |
| 4 | naive | 102 | 0 | False |
| 4 | append_only | 102 | 0 | False |

`instructions_remaining=0` at submission (seeds 0, 3, 4) means the full 50-instruction
queue was processed before reporting — a real success at the *mechanical* task
(sustained, ordered tool use over ~100 steps), just not at the *arithmetic* one.
Seeds 1 and 2 submitted early despite the anti-early-stop instruction, showing that
fix reduced but did not eliminate premature submission.

## Results: primary (Qwen/Qwen3-1.7B, max_model_len=8192)

Identical config (`results/phase3_ledger_primary.parquet`, same seeds/transactions/
budget/bootstrap).

| policy | n | success_rate | mean_tool_calls | stopped_reasons |
|---|---|---|---|---|
| naive | 5 | 0.000 | 62.6 | {'submitted': 3, 'step_budget_exhausted': 2} |
| append_only | 5 | 0.000 | 57.0 | {'submitted': 4, 'step_budget_exhausted': 1} |

**Also 0/5 for both policies** — and, unexpectedly, the larger model did not even
match the smaller one's *mechanical* task-following:

| seed | policy | tool_calls | instructions_remaining at submit | stopped_reason | success |
|---|---|---|---|---|---|
| 0 | naive | 130 | 0 | step_budget_exhausted | False |
| 0 | append_only | 102 | 0 | submitted | False |
| 1 | naive | 13 | 44 | submitted | False |
| 1 | append_only | 13 | 44 | submitted | False |
| 2 | naive | 27 | 37 | submitted | False |
| 2 | append_only | 27 | 37 | submitted | False |
| 3 | naive | 13 | 44 | submitted | False |
| 3 | append_only | 13 | 44 | submitted | False |
| 4 | naive | 130 | 0 | step_budget_exhausted | False |
| 4 | append_only | 130 | 0 | step_budget_exhausted | False |

Three of five seeds (1, 2, 3) submitted a final report after only 13–27 tool calls —
i.e. right after the 3 bootstrapped cycles ended, with 37–44 of 50 instructions still
unprocessed — a *more* severe version of the premature-submission failure the
anti-early-stop instruction was written to prevent, worse than anything seen on the
0.6B model. Two seeds (0, 4 on naive; 4 on append_only) instead ran out the full
130-call budget without ever submitting, a failure mode the 0.6B sweep never hit at
all (`stopped_reason` there was `submitted` in all 10 runs). `mean_tool_calls` is
correspondingly *lower* for the 1.7B model (62.6/57.0 vs. 93.4/91.6) — it is doing
*less* productive work per episode, not more.

## What this shows

**The primary result resolves the open question directly: the 0% success rate is not
a 0.6B-specific ceiling.** The 1.7B model fails the task too, at the same 0/5 rate,
and its *failure mode is worse*, not better — more premature submissions, more
runaway non-terminating episodes, fewer instructions processed per episode on
average. This is a genuine, mildly counterintuitive finding: going from 0.6B to 1.7B
parameters did not help this specific long-horizon, mechanical, tool-heavy loop, and
by the crude proxy of "how much of the task did it actually attempt," it did
measurably worse. A plausible read: the exact one-shot-bootstrapped, JSON-only
elicitation harness built for this project (see "Getting the agent loop to actually
work" above) may fit the 0.6B model's response patterns better by coincidence, or the
1.7B model's stronger prior toward "wrapping up" a task readably fights the
"keep repeating" instruction harder than the smaller, less fluent model's does. Either
way, this is now data, not speculation: a 1.7B model does not straightforwardly
"actually complete" this task the way spec's phrasing hoped, at least not with this
harness.

Consistent with Phase 3.2's retention-probe finding (retention collapses to near-zero
by depth 50 for both policies, also on the 0.6B model), neither model can reliably
track exact numeric state across a ~100-step tool-calling episode, regardless of
compaction layout — `naive` and `append_only` are statistically indistinguishable on
both models (0/5 each, n far too small for a real test anyway). Task success here
depends on the model's raw arithmetic/tracking fidelity and its ability to sustain a
repetitive tool-use loop, neither of which layout policy touches.

## Known limitations

- **n=5 seeds per model** — a binomial 0/5 has a wide Wilson CI (roughly [0%, 43%] at
  95%); this rules out "usually succeeds," nothing more precise, for *either* model.
- **Only one harness configuration tested per model** — the 1.7B model's worse
  mechanical performance could be specific to this exact bootstrap/prompt setup
  (tuned empirically against the 0.6B model first) rather than a fair test of the
  1.7B model's ceiling; a harness tuned against 1.7B specifically wasn't attempted.
- **`success` is strict, all-or-nothing** — a report that's correct on 4 of 5 accounts
  scores identically to one that's correct on 0 of 5. A partial-credit metric (e.g.
  fraction of accounts exactly matched) would separate "close" from "not even trying"
  and wasn't computed for either model.
- **The bootstrap cycles are scripted, not elicited** — the first 3 of ~50
  instructions are always executed correctly by construction, slightly inflating
  `instructions_remaining=0` outcomes relative to a fully model-driven episode.
- **A single task configuration** (50 transactions, 5 accounts, one summarizer
  config) — no sweep over task difficulty was run on either model.

## Next

Both required legs of the model-choice question are now answered: neither model
succeeds, and the larger one is not obviously better at even attempting the task
mechanically. Worth flagging as a candidate follow-up, not started here: whether a
harness change (e.g. a shorter task, more bootstrap cycles, or a stronger anti-wrap-up
instruction tuned specifically against the 1.7B model's tendencies) changes this
picture, versus this being a genuine capability ceiling for both model sizes on a
task this mechanically demanding.
