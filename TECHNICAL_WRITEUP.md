# AgentKV: Cache-Aware Context Compaction for Long-Horizon LLM Agents

## The problem

Long-horizon agents — the kind that run dozens or hundreds of tool calls
per episode — accumulate context. Every serving stack eventually has to
decide what to do when that context outgrows its window, and the default
answer is almost always some form of "summarize the middle and carry on."
That answer has a cost nobody measures by default: modern inference engines
(vLLM among them) cache the KV tensors for a request's prompt and reuse
them on the next request if the new prompt shares a prefix with the old
one. The instant a summarization step rewrites anything in the middle of
that prefix, every token after the rewrite point stops matching, the cache
entry is worthless from that point forward, and the engine has to recompute
it from scratch. This project starts from one question — how big is that
cost, really, on hardware anyone might actually have — and spends most of
its effort trying to answer it honestly rather than trying to make the
mechanism look good.

The concrete engineering contribution is a hybrid, per-segment compaction
policy that routes each piece of conversation history by content — keep
cheap turns verbatim, evict large tool outputs outright, summarize
everything else, and always protect a verbatim recent tail so the part of
the cache an agent leans on next is never the part that just got rewritten.
Measured against four other policies on 14 real synthetic agent
trajectories, it is the cheapest, by a statistically significant margin.
That result, and the five negative or null results measured alongside it,
are the subject of this writeup.

## Phase 0: measurement infrastructure, validated before trusting it

Everything downstream depends on the harness being trustworthy, so Phase 0
existed only to prove that. Gate 0's bar — replay the same trajectory twice
under the same policy, TTFT medians within 5%, cache-hit sequence
bit-identical — passed at 1.90% drift, but not on the first attempt: the
first version of the check reused one warm server for both "independent"
runs, so the second run silently inherited the first run's cached prefix
blocks. That's a test-methodology bug, not a reproducibility problem, and
it was only caught because the check looked at the actual `cached_tokens`
sequence rather than trusting a passing TTFT comparison alone. Fixed by
booting two fully separate engine instances. Fourteen of fifteen recorded
trajectories (a synthetic on-call-incident task, recorded once from a
local `Qwen3-4B-AWQ` model, spec §0.4) are usable; spec asked for "at
minimum 30." That gap is recorded here deliberately, not glossed over —
Phase 4's statistical claims are exactly where more raw trajectories would
help most if this project were extended.

## Phase 1: quantifying the bug

Naive compaction — LLM-summarize the older half of the conversation
whenever the window fills up — costs a median 63.0% of a trajectory's
total prefill compute on compaction events alone, not on making progress
(`results/phase1/phase1_gate1_summary.md`, 14 trajectories). Every number
here is checked against a from-first-principles prediction before being
trusted: `bench/divergence.py` computes, from the previous and current
prompts' token-id lists alone, exactly how many cache blocks *should*
still be valid, and asserts that against what vLLM's own Prometheus
counter actually reports. It very nearly matches exactly — one
boundary-completing case (a block with 15/16 tokens already cached needing
just one more to complete) credits vLLM with one block "too many" versus
naive floor-division, confirmed as a real engine behavior rather than a
bug in the prediction by cross-checking with a non-LLM stub summarizer.
Building that check before building any policy is what makes every later
Phase 4 number defensible: a policy's measured prefill cost is grounded in
a formula that was verified against the engine's real behavior first.

## Phase 2: cache-preserving layout — a clean null result

The obvious first fix: never re-summarize what's already been summarized.
`append_only` keeps a monotonically growing list of frozen summary
segments instead of collapsing them back into one summarizer call each
time — confirmed structurally sound (14/14 trajectories show non-decreasing
divergence index across successive events, spec §2.2's own headline
property). It did not reduce cost. Median prefill-token reduction versus
naive: 0.7%, IQR [-0.9%, 1.9%], not significant (Wilcoxon p = 0.53,
`results/phase2/phase2_gate2_summary.md`). The theoretically clean
property (monotonic prefix growth) didn't translate into a measured win,
because both policies retire turns through the identical summarizer at the
identical threshold — layout changes *where* cached text sits, not how
much of it there is to begin with.

## Phase 3: quality — the honest bottom line

Gate 3 asks for a (cost, task-success) point per policy, with error bars.
It exists, and it is not flattering: 0/5 task success for both naive and
append_only, on both the fallback (0.6B) and primary (1.7B) measurement
models (`results/phase3/phase3_summary.md`). The 1.7B model's *mechanical*
task-following — how much of a scripted 50-instruction queue it attempted
before giving up — was measurably worse than the 0.6B model's, not better;
three of five 1.7B episodes submitted a final report almost immediately
after the bootstrap ended, with 37-44 of 50 instructions untouched. Model
size, in the range tested, did not rescue the task. Underneath that floor,
the three measured legs (action agreement, fact-retention probes, the
ledger task itself) agree on a mechanism: retention collapses to near-zero
by conversation depth 50 regardless of which layout policy is used,
because both policies retire turns through the same summarizer at the same
threshold — quality depends on whether the summarizer preserved a fact
when it first retired that turn, a summarization-*fidelity* question, not
a layout question. Phases 2 and 3, measured independently from two
different angles, land on the same conclusion.

## Phase 4.1: KV-space eviction — negative, and mechanistically explained

`kv_evict` (StreamingLLM-style: keep an anchor plus a sliding window,
delete everything older with no replacement text) is supposed to be
"near-free" — no summarizer call, no rewritten text. Measured, it costs
*more* than naive: median -19.8% (i.e., 19.8% worse), IQR [-27.2%, -7.5%],
significant (Wilcoxon p = 0.0017,
`results/phase4/phase4_kv_evict_summary.md`). The mechanism, confirmed
directly from the event log: kv_evict's fixed-size window drops its oldest
turn and re-fires on almost every subsequent step (150 events across the
sweep versus 115-116 for naive/append_only), and each firing invalidates
most of the cached window because the window's front boundary shifts every
time. This surfaced a real architectural fact before it became a bug:
"zero re-prefill" is a property of a live, persistent decode session where
old KV entries are freed mid-session — this project's harness sends each
agent step as an independent HTTP request, reusing cache only through
vLLM's hash-based prefix matching across requests. Genuine zero-reprefill
eviction needs live-session KV management this request-based harness
cannot express; approximating it with shorter text prompts doesn't just
fail to capture the benefit, it actively performs worse than doing
nothing.

## Phase 4.2: attention-score eviction — infeasible on this hardware

The spec-suggested signal for a smarter eviction policy is accumulated
attention mass. vLLM's serving kernels never materialize real per-token
attention weights, by design, for speed — the only path to real weights is
a *separate* eager-attention model instance outside the serving engine.
That path was built and verified correct at trivial scale, then measured
at realistic scale: 100 tokens cost 1,716 MB peak and 2.3s; 300 tokens
cost 1,867 MB and 8.3s, worse-than-linear, consistent with eager
attention's O(n²) cost. This project's actual contexts routinely reach
several thousand tokens before compaction fires, and kv_evict alone fired
150 times across one sweep — at these per-call costs, a full comparative
sweep would take minutes per call, repeated hundreds of times. A first
attempt at 6,000 tokens crashed the entire WSL VM outright. Given three
options — skip and document, substitute a cheap non-attention proxy, or
cap the scored window small enough to stay safe — the second would no
longer be a faithful "attention mass" implementation and the third would
exclude most of the context from ever being eligible, undermining any real
comparison. Skipped and documented
(`results/phase4/phase4_attn_evict_summary.md`) rather than either quietly
dropped or faked with a proxy relabeled as the real thing.

## Phase 4.3: trajectory-aware offload — correct, but counterproductive

The deepest engineering investment in the project: a real
`KVConnectorBase_V1` implementation (vLLM's supported extension point,
not a workaround) that classifies conversation content — anchor and
frozen-summary segments worth offloading to host RAM, large tool outputs
not worth it — and offloads selectively rather than indiscriminately.
Two real design errors were caught by testing, not assumed away: the
first version expected an external driver to push classification state
into a live connector, which turned out to be architecturally impossible
once `VLLMEngine`'s HTTP-only relationship to the server process was
examined properly; the second stored data in a plain instance dict that
turned out invisible across the separate connector instances vLLM creates
per role (scheduler vs. worker), diagnosed from vLLM's own boot log
showing the connector constructed twice. After both fixes, a forced-
eviction round-trip test confirmed the mechanism works exactly as
designed: content is correctly classified (5 of 106 blocks selected in a
mixed test case, matching hand-worked expectations), stored, evicted from
GPU, reloaded from host RAM on a repeat request, and the reloaded
completion is byte-identical to the original. And it is roughly 44x
slower in wall-clock terms than doing nothing, because the classification
step decodes the full prompt and binary-searches for span boundaries on
every uncached request, at a cost that grows with context length on every
single step of a trajectory. The storage/reload mechanism is proven
correct; the classification strategy, as implemented, is not viable
without caching classification results by content hash — identified, not
yet built.

## Phase 4.4-4.5: the hybrid policy and the Pareto plot

Hybrid routes each candidate turn to one of three actions instead of
applying one rule uniformly: turns under a small-size threshold are kept
verbatim (too cheap to bother compacting); large tool outputs are evicted
outright, reusing kv_evict's mechanism selectively rather than as a blanket
sliding window; everything else is summarized, exactly like append_only.
The event log explains the win directly
(`results/phase4/phase4_hybrid_summary.md`): hybrid fires fewer
compaction events than any other policy (97 versus 115-150) because
keep-verbatim routing removes small turns from ever needing compaction,
and each firing is individually cheaper — a reprefill-to-tokens-saved
ratio of 0.71 (each event saves more than it costs to reprefill later),
versus roughly break-even for naive/append_only (~1.12-1.13) and a
lopsided 11.93 for kv_evict, whose sliding window saves the least per
event while reprefilling the most.

Gate 4 asks for a Pareto plot with at least five policies and at least
five seeds each. This project has four real `CompactionPolicy`
implementations with prefill-cost data; the fifth point is `NoOpPolicy` —
never compact, ever — added as an honest ceiling reference after
presenting the gap as a genuine three-way fork rather than picking
silently. Final measured medians across 14 trajectories: hybrid 70,026
tokens, naive 78,436, append_only 85,496, kv_evict 106,030 — hybrid's
reduction versus naive is a median 9.8% (IQR [6.4%, 13.2%]), Wilcoxon
p = 0.0011. `none`'s point is real but partial: every one of its 14
trajectories exhausted the model's context window before finishing,
drawn as a hollow marker rather than averaged in as if complete. The
quality axis is flat at 0/5 for all five policies — the Pareto plot proves
a cost *ranking*, not a cost/quality *tradeoff*, and says so plainly
rather than implying more than it shows. One real signal survives
underneath that floor: hybrid's episodes use substantially more of their
tool-call budget (110.6 mean calls versus ~93-95 for the other four) and
leave almost nothing of the ledger task's instruction queue unprocessed
(0.6 versus ~9) before failing — the *reported* final answer is still
wrong every time, so this is not a success-rate win, but it is a real,
measured sign that better routing lets the model progress further before
whatever mechanism caps this task's success rate takes over.

## Phase 5: descoped, not attempted

Spec gates KV re-basing and selective recompute behind "only attempt after
Phases 0-4 are written up and committed" and calls it "research-grade, may
fail." Given the time already invested in thoroughly measuring and writing
up Phases 0-4 and 6, this was a deliberate scope decision, made and stated
directly rather than silently dropped. There is no layer-depth error curve
to report, because the RoPE re-basing experiment that would produce one was
never run — a different thing from having run it and failed.

## Phase 6: demo, cost model, reproducibility

The live dashboard (`viz/dashboard.py`) and three-figure deck
(`experiments/demo_deck.py`) both replay already-committed measurements
rather than driving vLLM live, a deliberate choice given this project's own
repeated GPU/WSL fragility (documented below) — every number they show was
already measured once, correctly, with the theory-match check already
described. The context-map strip's colors are derived purely from
`prompt_tokens`/`cached_tokens`/`block_size`, the same quantities the
divergence check already validates, so the visual introduces no new
accounting.

The analytical cost model (`serving/cost_model.py`) is worth describing in
some detail because its first version was wrong in an instructive way.
Fitting one achieved/peak-FLOPS efficiency constant across all 9,905
measured steps validated badly — 91.9% median error against real TTFT.
Stratifying by step size explained why: efficiency rises monotonically
from ~5% for tiny steps to ~81% for large ones, because fixed per-request
overhead (HTTP round-trip, engine scheduling) dominates small requests and
amortizes away for large ones. Since compaction's real savings come from
large reprefill-after-compaction events, not tiny single-turn appends,
recalibrating on steps above 200 tokens dropped the median error to 9.1%.
Extrapolating naive-versus-hybrid's real, measured per-step token
sequences to a 70B model on an H100 — not scaling one summary number, but
replaying the same measured shapes through a larger FLOPs formula — gives
a 10.2% FLOPs reduction, consistent with Gate 4's independently-measured
9.8% token reduction, a real cross-check between two differently-derived
numbers. Reported with a fitted point estimate, IQR sensitivity bounds, and
an independent industry-MFU cross-check, not as a bare projected dollar
figure.

Building the reproducibility pipeline (`experiments/run_all.py`) surfaced
one more real bug: `VLLMEngine.__init__` raising on a failed server boot
meant its context-manager cleanup never ran, leaking the server subprocess
and several GB of GPU memory — confirmed concretely when a timed-out boot
attempt left a live process that then sabotaged the next boot attempt.
Fixed by wrapping the risky startup call in a try/except that tears down
the subprocess before re-raising. A full live `--smoke` run of the
resulting pipeline was not completed in this session: the command plan is
verified correct by inspection and dry-run, but live execution hit a
WSL2 I/O stall (evidenced by a plain, GPU-free `pytest` run hanging in
uninterruptible sleep) that this project's own history had already flagged
as an occasional, environment-level failure mode, not a defect in the new
orchestration code.

## What didn't work, collected

`append_only`'s layout change (null on both cost and quality).
`kv_evict`'s sliding window (costs more, not less, than doing nothing
smart). `attn_evict` (infeasible on this hardware at the scale needed).
Trajectory-aware offload (correct and verified, ~44x slower as
implemented). The task-success floor (0/5 for every one of five policies,
on two model sizes). A full live reproduction run (built, dry-run verified,
not completed end-to-end this session). Each of these is reported with its
mechanism, not just its outcome, because the mechanism is what makes a
negative result reusable by someone else.

## Limitations

The fullest version of this section lives in `README.md`, which this
writeup does not repeat verbatim. In short: the Pareto plot's headline
claim is a cost ranking, not yet a validated cost/quality tradeoff; `none`'s
Pareto point is partial; every policy uses one fixed configuration, unswept;
the 70B/H100 cost projection carries real efficiency-transfer assumptions,
stated with sensitivity bounds rather than hidden; all measurements come
from one synthetic task on 14-15 trajectories, not real production traffic;
and the reproducibility pipeline's live verification is incomplete.

## What's next

The single highest-leverage follow-up Phase 3 already points at directly:
investigate summarization *fidelity* (does `max_summary_tokens` or
`retire_fraction` move retention or task success more than layout or
routing policy does?) rather than continuing to vary how compacted content
is arranged. Behind that: implement offload's suggested classification-
caching fix and re-measure; sweep each policy's threshold/window
configuration instead of reporting one point each; find a task/model
combination that clears the 0% success floor so the Pareto plot's quality
axis can say something beyond "flat"; and, if revisited, a genuine
live-session KV-management path for eviction, which Phase 4.1 identified
as the real prerequisite "zero re-prefill" eviction needs and this
project's request-based harness cannot itself provide.
