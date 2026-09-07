# AgentKV — Full Project History & Interview Prep

This document is not part of the spec deliverables (`README.md`, `TECHNICAL_WRITEUP.md`
cover that). It's a working memory dump: how the project was actually built, phase by
phase, every architectural decision and why, every bug and how it was found and fixed,
and a bank of likely interview questions with the honest answers. Written for your own
prep, not for a reader who hasn't seen the code.

---

## 1. The one-paragraph pitch

Long-horizon LLM agents periodically summarize/compact their growing context so it fits
the model's context window. The industry-standard way to do this — rewrite the middle of
the prompt in place — silently invalidates most of the KV cache every time it fires,
because inference engines like vLLM cache prefixes by exact token-id match: change one
token early in the prompt and everything after it must be recomputed. Nobody measures
this. This project (1) proves the cost is real and large (63% of all prefill compute on
a naive baseline), (2) tries several fixes of increasing sophistication, (3) finds that
the obvious fix (never rewrite, only append) doesn't actually help, and a "smarter"
theoretical fix (KV-space sliding-window eviction) actively makes things worse, and (4)
lands on a **content-aware hybrid router** (keep small turns verbatim / KV-evict large
tool outputs / summarize everything else) that gives a statistically significant, clean
9.8% prefill-cost reduction on every single trajectory tested — the only policy in the
project with a real win. It also finds, honestly, that none of this moves *task success*
on the one hard benchmark task built (a 50-transaction ledger, 0/5 for every policy) —
a floor effect in the underlying small models, not something layout policy touches.

Built entirely on a single RTX 4060 Laptop (8GB VRAM) under WSL2 — no cloud GPU, no API
keys, fully local (vLLM 0.8.5 + Qwen3 0.6B/1.7B).

---

## 2. How it was built — phase by phase

### Phase 0 — Measurement harness
**Goal:** trustworthy instrumentation before anything else. Gate: replaying the same
trajectory twice gives TTFT within 5% and *bit-identical* cache-hit counts.

- Served Qwen3 models locally via `vllm.entrypoints.openai.api_server` (not the offline
  batch `LLM.generate()` API — that doesn't populate per-request cache metrics in vLLM
  0.8.5). Read cache hits from vLLM's own Prometheus counter
  (`vllm:gpu_prefix_cache_hits_total`), not a hand-rolled estimate.
- Recorded 14 valid synthetic agent trajectories (~165 turns each, an incident-
  investigation tool-use environment) using a separate, stronger local model
  (Qwen3-4B-AWQ, 4-bit quantized — the unquantized 4B didn't fit in 8GB).
- **First real bug, caught by the gate itself, not the code:** the first Gate 0 attempt
  reused one warm server for "two runs" — run 2 silently inherited run 1's cache, so the
  "reproducibility" check was measuring nothing. Fixed by booting two fully independent
  server processes per check. This is the pattern that repeats throughout the project:
  the measurement methodology itself is the first thing that has to be proven correct.
- Deviated from spec's "≥30 trajectories" down to 14-15, explicitly logged as a scope
  cut under hardware/time constraints, not hidden.

### Phase 1 — Quantify the bug (Gate 1)
**Goal:** put a number on the cost naive compaction pays. Gate: state X extra prefill
tokens / Y% of compute / Z% of wall clock, with error bars.

- Built the `CompactionPolicy` interface (spec's own rule: adding a new policy touches
  exactly one new file) and `NaivePolicy`: once the prompt exceeds 60% of the context
  window, replace the oldest 50% of turns with one fresh LLM-generated summary in place.
- Built `bench/divergence.py`: a pure, GPU-free block-math model that predicts, from two
  consecutive steps' token-id sequences, how many cache blocks *should* be reused —
  independently cross-checked every step against vLLM's real reported number.
- **Result: naive compaction spends a median 63.0% of all prefill compute re-doing work
  forced by its own mid-context rewrites** (36,611 extra tokens per 100-step trajectory,
  IQR 34,693–40,516, n=14). This is the headline justification for the whole project.
- **Five real bugs found and fixed while getting this number**, all environment/
  measurement bugs, not logic bugs:
  1. GPU clock baseline captured while idle (~210MHz) instead of loaded (~2250MHz) —
     every drift check failed by ~1000%. Fixed by warming up the engine before baseline.
  2. Single-sample clock noise (one reading caught a 2535MHz transient spike). Fixed
     with a rolling-median comparison.
  3. vLLM's Prometheus cache-hit counter lagged after large requests — reading it once
     could attribute a stale update to the next request. Fixed by polling until stable.
  4. A genuine ±1-block accounting quirk in vLLM 0.8.5 (completing an almost-full
     trailing block credits the whole block) — bounded and non-compounding, so the
     theory check now tolerates ±1 block.
  5. Per-trajectory state reset wrongly cleared cross-trajectory cache bookkeeping,
     making legitimate reuse (the shared anchor prompt still resident from the prior
     trajectory) look like a theory violation.
- Also documented, not silently dropped: the spec's clock-drift threshold (5%) proved
  unenforceable on this bursty WSL2/laptop workload (legit clock swung 885–2550MHz) —
  kept the spec's number, logs loud warnings on breach, and records full GPU telemetry
  per step so it's auditable rather than quietly loosening the check.

### Phase 2 — Cache-preserving context layout (Gate 2 — **failed**, informatively)
**Goal:** append-only layout (never rewrite, only append new summaries) should reduce
prefill cost vs. naive at equal-or-better agreement.

- Built `ContextState`/`Segment` (`frozen` is a type-level append-only guarantee, not a
  convention), `AppendOnlyPolicy`, block-alignment padding, and a paired Wilcoxon
  signed-rank test (`bench/stats.py`, no scipy dependency).
- **Result: median 0.7% reduction, IQR [-0.9%, 1.9%], p=0.53 — not significant.** Roughly
  half the trajectories were actually *worse* under append-only.
- **Why, mechanistically (the real finding):** the design hypothesis was right about one
  thing — append-only's divergence point creeps monotonically later every event (14/14
  trajectories, confirmed) — but wrong about the net effect, because `frozen` grows
  *unboundedly*. A bigger *cached* prefix and a bigger *total* prompt grow together, so
  the actual number of tokens that need reprefilling stays roughly the same. Traced
  concretely on traj-000: by event 9, append-only's divergence point (830) was 12x later
  than naive's (68) — but the real reprefilled-token count was only ~8% lower, because
  append-only's post-compaction context was itself larger.
- **A subtle, important bug-hunt:** the first theory-mismatch found here was misdiagnosed
  as a metric-settling-timing bug, "fixed," and appeared to work — until a real (non-stub)
  full sweep reproduced the exact same mismatch byte-for-byte, proving the fix did
  nothing. Root cause turned out to be real: at temperature=0, the small model often
  quotes retired text verbatim into its own summaries, and that exact text was still
  resident in vLLM's *global* cross-request cache pool from a non-adjacent earlier
  request — genuine reuse the pairwise theory model couldn't see. Fixed by making the
  theory check one-directional (only flag cache reuse falling *short* of prediction, not
  exceeding it).
- Also found and killed zombie multi-day-old GPU processes silently blocking a fresh
  server boot (`nvidia-smi --query-compute-apps` is the fast diagnostic).
- Forced a config split: `AppendOnlyPolicy.frozen`'s unbounded growth overflowed the
  1.7B primary model's tight 8192-token window on one trajectory — from here on, most of
  the project runs on the fallback model (Qwen3-0.6B, 16384 context, 3.56x KV headroom)
  rather than the primary (1.7B, 8192, only 1.15x headroom).

### Phase 3 — Quality measurement (Gate 3 — met, unflatteringly)
**Goal:** for every policy, produce a (prefill cost, task success) point. Three
independent legs, spec explicitly warns against reporting only the flattering one.

- **§3.1 Next-action agreement:** at each decision point, compare the model's action
  given full uncompacted context vs. compacted context. Needed grammar-constrained
  decoding (`guided_json`) to get the small model to reliably emit valid, on-schema JSON
  at all — two rounds of failure first (free-text rambling, then an exemplar that primed
  immediate empty-stop). Result: naive 0.779/0.572 (tool/args match) vs. append-only
  0.800/0.593 — a small directional edge, untested for significance, and only measurable
  over the first ~30-40% of each trajectory (the *uncompacted* counterfactual itself hits
  the context ceiling).
- **§3.2 Retention probes:** inject synthetic facts at controlled depths, query at step
  100. Both policies retain perfectly at depth 10, collapse to ~0% by depth 50. No
  consistent append-only advantage (mixed: naive better at depth 30, append-only better
  at 70/90). Diagnosis: retention depends on whether the *summarizer* preserved a fact
  when it first retired that turn — a summarization-fidelity question, not a layout one.
  Two elicitation bugs found here too: the model kept re-entering `<think>` mode instead
  of answering (fixed with `guided_json`), and a one-shot exemplar's example value
  happened to share a fact *kind* with a real fact template, so the model literally
  copied the exemplar's value regardless of the real injected one (fixed by using an
  arithmetic exemplar sharing no vocabulary with any fact kind).
- **§3.3 Verifiable ledger task (the real task-success number):** a live, stateful
  50-transaction multi-account ledger; the model calls tools to execute transactions,
  final reported balances checked against ground truth. **0/5 success on both the 0.6B
  and 1.7B model.** The 1.7B model's failure mode was *worse*, not better — more
  premature `submit_final_report` calls, less of the queue actually processed per
  episode (62.6/57.0 mean tool calls vs. 93.4/91.6 for 0.6B). Needed three real fixes to
  even get a coherent agent loop: forcing a "start now" instruction (didn't fully fix
  cold-start tool selection alone), bootstrapping 3 scripted correct cycles before
  handing control to the model (1 example wasn't enough to establish "keep repeating"),
  and enum-restricting the tool-name field in the JSON schema (prevented hallucinated
  tool names). Also hit a real WSL2 GPU-paravirtualization driver desync
  (`dxgk: dxgkio_query_adapter_info: Ioctl failed`) mid-sweep, fixed with `wsl --shutdown`
  from the Windows side.
- **Bottom line: layout policy (naive vs. append-only) does not move quality on any of
  the three measured axes.** Phases 2 and 3 together say the same thing from two angles:
  this specific append-only implementation doesn't demonstrably win on cost or quality.

### Phase 4 — KV-space eviction and the Pareto frontier (Gate 4)
Five sub-parts. This is where the project's actual positive result comes from.

**§4.1 `kv_evict` — StreamingLLM-style sliding-window eviction.**
Before writing code: confirmed by reading vLLM's own connector source that "zero
re-prefill" is a property of a *live, persistent decode session* — this project's
harness sends each step as an independent HTTP request, so there's no live session to
evict from; the only lever available is sending a *shorter text prompt*. Built anyway,
measured honestly. **Result: -19.8% vs. naive (IQR [-27.2%, -7.5%], p=0.0017) —
significantly worse, the opposite of its "near-free" framing.** Mechanism confirmed from
the event log: a fixed-size sliding window re-fires on almost every step (150 events vs.
115 for naive) because its front boundary shifts every time, invalidating most of the
window's cache each time, and each firing is also individually more expensive (6,177 vs.
5,160 tokens reprefilled/event). Conclusion: real KV-space eviction needs actual
in-engine session persistence this project's request-based harness can't express;
approximating it with shorter text prompts doesn't just fail to help, it actively hurts.

**§4.2 `attn_evict` — attention-score eviction. Not built; a real infeasibility finding.**
vLLM's serving kernels never materialize real attention weights (by design, for speed).
The only path outside vLLM internals is a separate eager-attention `transformers` model
instance. Verified the mechanism works at 8 tokens. Then verified it doesn't survive
scale: a first attempt at 6000 tokens **crashed the entire WSL VM**. A follow-up attempt
to cap memory with `RLIMIT_AS` failed for an unrelated reason (`safetensors`' mmap-based
loading trips a virtual-address-space cap even when resident memory is modest). Retested
at genuinely small, safe sizes: memory and time both blow up far worse than the naive
"one layer's tensor" estimate — 300 tokens already costs 1,867MB and 8.3s, worse-than-
linear (O(n²)-consistent) scaling that makes this project's actual multi-thousand-token
working set impractical (minutes per scoring call, needed 100+ times per sweep). Reported
as a genuine infeasibility on this specific hardware with this specific
out-of-vLLM-internals constraint, not evidence attention-score eviction is a bad idea.

**§4.3 `offload` — trajectory-aware CPU KV offload. Correct, but net negative.**
The only component that runs *inside* vLLM itself, via the real, documented
`KVConnectorBase_V1` extension point (adapted from vLLM's reference disk-backed
connector to an in-memory host-RAM version, storing only turns worth keeping — anchor,
frozen-summary segments — while excluding large tool outputs). **Two real design bugs
caught by testing, not assumed away:**
1. First version needed an external driver to call `set_current_state()` on a live
   connector — impossible, since the connector lives inside a separate vLLM server
   process this project only talks to over HTTP. Fixed by making the connector
   self-contained: it decodes and classifies from `request.prompt_token_ids` itself.
2. Storage worked (`13/13 blocks worth offloading` logged) but every lookup — even for
   the same request — reported 0 keys. Root cause: vLLM creates *separate connector
   instances per role* (SCHEDULER does lookups, WORKER does storage); a plain instance
   dict was invisible across them. Fixed with a module-level shared dict (confirmed safe
   only because this single-GPU non-distributed setup has no real second process).
Round-trip verified with a forced-eviction test: byte-identical completion after reload
from the external cache — the mechanism is provably correct. But **comparative
measurement showed offload ~44x slower** (1,477.5ms baseline vs 64,832.2ms with offload
on one trajectory, cut short by a WSL crash) — the classification step decodes the full
prompt and binary-searches token boundaries on every uncached request, and that cost
grows with context length every step. A real, well-measured negative result: the
mechanism works, the specific classification strategy doesn't scale as implemented.

**§4.4 `hybrid` — the project's actual win.**
Per-turn routing into three actions: **keep** verbatim (small turns, too cheap to
touch), **evict** (large tool outputs, dropped with no replacement — same mechanism as
kv_evict but *selective*), or **summarize** (everything else, appended as a new frozen
segment — same mechanism as append-only). Spec's suggested routing signal ("predicted
attention mass") is exactly what §4.2 proved infeasible — used spec's own documented
fallback instead: a hand-tuned heuristic keyed on turn size/type (large tool output vs.
small turn vs. everything else). **Result: median 16.6% reduction vs. naive in this
sweep (IQR [15.5%, 20.1%], p=0.0011) — wins on every single trajectory, not just on
median.** Mechanism, confirmed from the event log: hybrid fires *fewer* compaction
events (97 vs. 115-150) because keep/evict routing removes turns from ever needing
compaction, and each firing is *individually cheaper* (mean 3,799 tokens reprefilled vs
5,092-6,177 for others) while saving *more* per event (5,692 vs 3,676-4,756) — giving it
the only reprefill:saved ratio below 1.0 (0.71) of any policy tested. Naive/append-only
roughly break even (~1.12-1.13); kv_evict loses badly (11.93).

**§4.5 The Pareto plot / Gate 4 itself.**
Added a 5th policy, `none` (never compacts), specifically to hit Gate 4's "≥5 policies"
bar honestly — `attn_evict` was never built and `offload` measures a different axis
(wall-clock, not prefill tokens), so neither was a natural 5th point; `none` is a real,
cheap, honestly-labeled ceiling reference instead. Its cost is real but incomplete (it
blows past the context window on every trajectory, drawn as a hollow marker, never
averaged into the other four's comparison). In this specific 5-way interleaved sweep
(a different rotation than §4.1/§4.4's 3-way/4-way ones, so the numbers shift slightly —
expected variance from cross-trajectory cache-warming order, not a bug), the numbers
that became the project's headline: **hybrid 70,026 vs naive 78,436 median prefill
tokens — 9.8% reduction, p=0.0011.** Task success: **0/5 for every single policy** —
the quality axis is completely flat, a genuine floor effect in the small models' ability
to track exact numeric state over ~100 tool calls, not a policy effect (every policy
shares the identical underlying model). A secondary, partial quality signal survives
underneath the floor though: hybrid's episodes used far more of their tool-call budget
(110.6 vs ~93-95) and left almost nothing of the 50-item queue unprocessed (0.6 vs ~9) —
hybrid got the model through nearly the whole mechanical task before failing only at the
final arithmetic step, unlike the other four which failed both mechanically and
arithmetically. Three real bugs caught before they touched committed data here too:
`none` crashing the harness on context overflow (unique to `none`, since every other
policy actively avoids overflow by design), an accidental first run on the wrong
(primary, tight-context) model reproducing a known overflow, and — the recurring
project-wide footgun — `RecordCollector.flush()`'s append-not-overwrite behavior
silently mixing a failed run's partial data with a rerun's, caught and fixed by deleting
and rerunning clean.

### Phase 5 — KV re-basing / selective recompute
**Explicitly skipped, by your decision** ("too complicated"). Spec itself labels this
"research-grade, may fail" and a stretch goal. Documented as a deliberate scope decision
in the writeup and README, not a failed attempt — no fabricated results were produced
for it.

### Phase 6 — Cost model, packaging, demo, writeup
**§6.1 Analytical cost model (`serving/cost_model.py`).** FLOPs-based prefill cost
formula (`2 × N_params × prefill_tokens` plus attention terms). **First attempt failed
validation badly** (91.9% median error) — root-caused via a stratified breakdown by step
size, which showed measured "efficiency" (achieved/peak FLOPS) rises monotonically from
5% to 81% with step size, because fixed per-request overhead (HTTP, scheduling, kernel
launch) dominates small requests and isn't part of a pure-FLOPs model. **Fixed** by
calibrating only on steps >200 tokens (the compute-bound regime where compaction's real
savings actually live), dropping error to 9.1% median. Cross-checked the 70B/H100
extrapolation's 10.2% FLOPs-reduction figure against Gate 4's independently-measured
9.8% token-reduction figure — close agreement, a real consistency check between two
differently-derived numbers. Also ran (and reported honestly, not hidden) an aggregate
same-hardware sanity check that *didn't* validate cleanly (predicted a time saving; real
wall-clock showed a small loss) — investigated via SM clock (ruled out simple thermal
throttling) and flagged the likely-but-unconfirmed explanation: hybrid's own per-turn
CPU-side routing logic costs more client-side time than naive's single threshold check,
which a pure-GPU-FLOPs model was never built to capture.

**§6.2 Reproducibility (`experiments/run_all.py`, `make reproduce`).** 15 pipeline steps
in dependency order, `--smoke`/`--dry-run`/`--from`/`--only` flags. Building this
surfaced a real, previously-latent bug: `VLLMEngine.__init__` could leak its subprocess
(and ~6.5GB of GPU memory) if `_wait_until_ready()` raised on timeout, because raising
inside `__init__` means `__enter__`/`__exit__` never run, so `shutdown()` never fires.
Fixed by wrapping the wait in try/except and calling `shutdown()` before re-raising.
Full live `--smoke` verification was **not** completed — repeated vLLM boot timeouts and
a WSL2 I/O stall (even plain `pytest -q` hung in uninterruptible D-state) ate the
available time. Verified instead, honestly, that this didn't corrupt anything: backed up
`results/` before attempting the run, and confirmed via `diff -rq` afterward that zero
files were actually changed (every failed step died before writing output). This gap is
the explicit headline caveat in `results/phase6/phase6_reproduce_summary.md`, cross-
referenced in README's limitations — not hidden.

**§6.3 README restructure.** Spec's exact required order: cliff figure → one-sentence
result → Pareto figure/table → reproduce instructions → limitations (written
prominently, per spec's own framing that this is "the single strongest credibility
signal in a portfolio repo").

**§6.4 Technical writeup.** 2,740 words, phase-by-phase, explicitly including what
didn't work (kv_evict's regression, attn_evict's infeasibility, offload's 44x slowdown)
and explicitly *not* fabricating a Phase 5 result it never attempted.

**Demo design (dashboard + deck).** Two artifacts, both built to replay already-
committed data rather than drive live inference — a deliberate decision given how much
real GPU/WSL fragility this project hit (thermal throttling, a process stall needing
`kill -9`, the VLLMEngine leak, repeated boot hangs): "must run reliably during an
interview on a laptop with no internet" (spec's own words) is incompatible with live GPU
dependency on this hardware. `viz/dashboard.py` is a `rich` terminal TUI (no new web
framework — `rich` was already a pinned dependency, and the spec explicitly says not to
over-engineer this); it needed a genuinely new precompute step (hybrid's next-action
agreement had never been measured anywhere else in the project — only naive/append-only
had). The three-figure deck (cliff → fix → Pareto) regenerates its first two panels from
the fallback-model data rather than reusing Phase 1's existing cliff figure, specifically
because Phase 1 predates the primary→fallback model switch and reusing it would have
silently put two different models on adjacent panels. One nuance caught by checking the
real data before writing any claim about it: the spec's "yours reddens only the tail"
framing isn't quite what happens — the *raw count* of invalidated blocks per event is
actually similar between naive and hybrid; what differs is hybrid's reused (green)
*prefix* growing across events while naive's stays flat, so hybrid's red *fraction*
shrinks over time rather than each event's red segment being small. Reported the honest
version.

---

## 3. Architectural decisions and the reasoning behind them

- **HTTP completions API, not the offline batch API.** The batch API doesn't expose
  per-request cache metrics in this vLLM version — a hard measurement requirement, not
  a style choice.
- **One `CompactionPolicy` ABC, one file per policy.** Spec's own convention; paid off
  concretely every time a new policy (kv_evict, hybrid, none) was added with minimal
  surface area and no cross-cutting changes.
- **`frozen` as a type-level append-only guarantee**, not a runtime convention —
  structural correctness over trust.
- **Fallback model (Qwen3-0.6B, 16k context) as the default measurement model** for
  Phases 2 onward, not the primary (1.7B, 8k context) — the primary's KV headroom
  (1.15x) proved too tight once policies like append-only's unbounded `frozen` list
  entered the picture. This is a real, disclosed limitation: most of the project's
  numbers are on the smaller model, with the cost model's extrapolation section
  explicitly built to address "does this scale."
- **Interleaved A/B(/C/D/E) ordering, always**, never a naive "run policy A on all
  trajectories, then policy B" — controls for any trajectory-order or thermal-drift
  confound, at the cost of the numbers shifting slightly between differently-sized
  interleavings (documented explicitly in Phase 4.5).
- **`RecordCollector.flush()` appends, never overwrites** — a deliberate design (so a
  long sweep can be safely resumed after a crash without losing earlier trajectories),
  but its sharp edge (a rerun with the same output filename silently mixes with a bad
  prior attempt) bit the project twice and is now a standing, documented hazard: every
  new experiment script is written to use a fresh output filename per sweep rather than
  reuse another script's.
- **`none` added as the honest 5th Pareto policy** rather than stretching `attn_evict`
  or `offload` to fill that slot — a ceiling reference, not a compaction strategy,
  drawn distinctly (hollow marker) rather than presented as comparable.
- **Demo replays committed data, never drives live inference** — reliability under
  interview conditions outweighs the marginal impressiveness of "look, it's really
  running the GPU right now," especially given this project's own documented track
  record of WSL/GPU fragility.
- **No magic numbers in code** — every threshold, model spec, and hardware constant
  lives in `configs/*.yaml`, not hardcoded (spec §10, followed consistently, e.g.
  `configs/cost_model.yaml` for the Phase 6 cost model).

---

## 4. Full bug/issue log (chronological, cross-phase)

| # | Phase | Bug | Root cause | Fix |
|---|---|---|---|---|
| 1 | 0 | Gate 0 "reproducibility" check passed for the wrong reason | Reused one warm server for both "independent" runs — run 2 inherited run 1's cache | Boot two fully separate server instances per check |
| 2 | 1 | Clock-drift check failed by ~1000% | Baseline clock sample taken while GPU was idle (~210MHz), not loaded (~2250MHz) | Warm up the engine with throwaway requests before sampling baseline |
| 3 | 1 | Clock-drift check still noisy post-fix | Single clock sample can land on a transient spike | Rolling-median comparison + median-of-several-samples baseline |
| 4 | 1 | Cache-hit counter occasionally misattributed | vLLM's Prometheus counter lags briefly after large requests | Poll until the value stops changing before reading it |
| 5 | 1 | ±1 block "mismatch" on some steps | Genuine vLLM 0.8.5 accounting quirk when a step completes an almost-full trailing block | Tolerate ±1 block in the theory check (bounded, non-compounding) |
| 6 | 1 | Spurious theory violation at trajectory boundaries | Per-trajectory bookkeeping reset to empty while the shared server/cache wasn't | Thread last-sent prompt state across trajectory boundaries |
| 7 | 2 | A "fix" that didn't fix anything | Diagnosed a real mismatch as a metric-settling-timing bug; confirmed via a stub-summarizer test that never exercised the real code path | Real cause: cross-request global cache reuse (verbatim-quoted retired text landing in a non-adjacent request). Made the theory check one-directional |
| 8 | 2 | Fresh server sat at 0% GPU util for minutes | Days-old orphaned zombie GPU processes holding unresolvable CUDA contexts | Killed via PIDs found with `nvidia-smi --query-compute-apps` |
| 9 | 2 | HTTP 400 context-length error on one trajectory | Append-only's `frozen` grows unboundedly; overflowed the primary model's 8192 window | Split `max_model_len` per model; ran on fallback (16384) instead |
| 10 | 3.1 | Model produced unparseable rambling / instant empty stop | Nothing forced valid JSON; a naive priming exemplar broke differently than free text did | Grammar-constrained decoding (`guided_json`) forces on-schema output from token 1 |
| 11 | 3.2 | Model looped `<think>` mode instead of answering | Base trajectory context is all `<think>`-wrapped tool-call turns | Same `guided_json` fix |
| 12 | 3.2 | Silent exemplar-copying corrupting one fact kind's results | One-shot exemplar's example answer shared a fact *kind* with a real fact template | Replaced exemplar with an arithmetic example sharing no vocabulary with any fact kind |
| 13 | 3.1 | Full-context comparison crashed mid-sweep | "Uncompacted" counterfactual prompt exceeded `max_model_len` — the exact cliff this project exists to study, showing up in a different experiment | Check length before each call; stop that trajectory's agreement measurement there, log it explicitly |
| 14 | 3.3 | Model always picked `submit_final_report` first, with garbage args | Zero in-context precedent for a fresh live episode | Explicit "start now" instruction (partial fix) |
| 15 | 3.3 | Model did one correct cycle then wrapped up immediately | One example doesn't establish "keep repeating" | Bootstrap 3 scripted correct cycles before handing control to the model |
| 16 | 3.3 | Model occasionally invented nonexistent tool names | Unconstrained `name` field in the JSON schema | Enum-restrict `name` to the environment's real tool list |
| 17 | 3.3 | `vllm serve` hung indefinitely at CUDA init | WSL2 GPU-paravirtualization driver desync (`dxgk` ioctl failure), likely from many rapid server restarts | `wsl --shutdown` from Windows, then fresh boot |
| 18 | 4.2 | WSL VM crashed entirely | Eager-attention capture at 6000 tokens far exceeded the ~7.6GB WSL memory budget | Investigated at safe small sizes instead; concluded genuinely infeasible on this hardware, didn't force it |
| 19 | 4.2 | Model loading failed under a memory cap | `RLIMIT_AS` caps virtual address space, which `safetensors`' mmap-based loading trips even at modest real memory use | Abandoned the artificial cap; measured real costs directly instead |
| 20 | 4.3 | No way for an external driver to set connector state | Connector instances live inside a separate vLLM server process; no in-process call was reachable | Made the connector self-contained — it decodes and classifies from the request's own token ids |
| 21 | 4.3 | Stored data never found on lookup, even for the same request | vLLM creates separate connector instances per role (SCHEDULER vs WORKER); a plain instance dict was invisible across them | Module-level shared dict (safe here only because this is a single-process, non-distributed setup) |
| 22 | 4.5 | `none` policy crashed the sweep | Every other policy actively avoids overflow by design; `none` never compacts, so it's the first policy to actually exceed `max_model_len` | Proactive length check before each call; stop and flag (`context_exceeded=True`) rather than crash |
| 23 | 4.5 | First full sweep produced impossible numbers | Accidentally run on the primary (tight-context) model instead of the established fallback convention | Reproduced the same crash on unmodified old code to confirm it wasn't new-code-specific; added `--use-fallback` |
| 24 | 4.5 | Committed parquet had ~2x the expected trajectory count, mixed bad data | `RecordCollector.flush()` appends, not overwrites; a failed run's partial output plus a rerun's full output landed in the same file | Deleted the contaminated parquet, reran clean |
| 25 | 6.1 | Cost model validated at 91.9% median error | A single blended efficiency constant conflates tiny-step (overhead-dominated) and large-step (compute-bound) regimes | Stratified the fit by step size; calibrated only on steps >200 tokens; error dropped to 9.1% |
| 26 | 6.2 | `gate0_check` step timed out during `--smoke` testing | Orphaned vLLM worker process from an earlier failed boot was still holding ~6.5GB GPU memory | Root cause: `VLLMEngine.__init__` never ran `shutdown()` on a startup timeout because the exception happened inside `__init__`, before `__enter__` could ever be reached. Fixed with a try/except around the wait, calling `shutdown()` before re-raising |
| 27 | 6.2 | Repeated vLLM boot hangs, and a plain `pytest -q` hanging in uninterruptible D-state | WSL2 environment-level instability (recurring pattern across the whole session, not isolated to this step) | Killed stuck processes, confirmed GPU returned to idle, verified via backup diff that no committed data was corrupted, and documented the unresolved live-verification gap honestly rather than either hiding it or forcing through an unstable environment |

**The pattern worth naming out loud:** almost none of these were "I wrote the wrong
logic" bugs. They were measurement-methodology bugs (1, 7, 10-13), environment/
infrastructure bugs (8, 17-19, 26-27), or genuine mechanism discoveries that looked like
bugs until investigated (7, 9 in spirit). The project's actual engineering discipline
was less "write correct code the first time" and more "don't trust a number until you've
tried to break the thing that produced it."

---

## 5. Headline numbers (for quick recall)

- **63.0%** — median share of naive compaction's total prefill compute spent purely on
  compaction-triggered re-prefill (Gate 1).
- **0.7%, p=0.53** — append-only vs naive prefill reduction; not significant (Gate 2,
  the first negative result).
- **-19.8%, p=0.0017** — kv_evict vs naive; significantly *worse* (§4.1).
- **~44x slower** — CPU offload's measured wall-clock overhead vs no offload (§4.3),
  despite the store/reload mechanism itself being proven byte-correct.
- **9.8%, p=0.0011** — hybrid vs naive prefill reduction, the project's headline result
  (Gate 4, 14 trajectories, wins on every single one).
- **0/5** — task success rate, every one of 5 policies, on the 50-transaction ledger
  task (Gate 4's quality axis — a floor effect in the model, not a policy effect).
- **9.1%** — cost model's median validation error after correcting for the small-step
  overhead bias (down from 91.9% uncorrected).
- **10.2% vs 9.8%** — cost model's independent 70B/H100 FLOPs-reduction extrapolation
  vs. Gate 4's directly-measured token-reduction figure — the cross-check that gives
  the extrapolation credibility.
- **~$0.02/trajectory → ~$159/day → ~$58k/year** — illustrative dollar scaling of the
  hybrid-vs-naive saving at 100,000 trajectories/day, H100 pricing, point-estimate
  efficiency. Explicitly labeled illustrative, not a real production estimate.

---

## 6. Likely interview questions and how to answer them

**"Walk me through the project in two minutes."**
Use §1 above almost verbatim. Lead with the mechanism (cache invalidation from
mid-context rewrites), not the tech stack.

**"What's the actual contribution — what's new here?"**
Not a new eviction algorithm — the KV-space eviction idea (StreamingLLM-style) is
well-known and this project shows it doesn't transfer cleanly to a request-based serving
harness. The contribution is: (1) a rigorous *measurement* of a cost nobody quantifies,
(2) an honest demonstration that the "obvious" fix (append-only) doesn't work and a
"principled" fix (attention-score/KV-space eviction) is either infeasible or actively
harmful in this setting, and (3) a hybrid content-aware router that actually works,
with the mechanism explained from real event-log data, not asserted.

**"Why didn't append-only work? Isn't 'never rewrite the cache' obviously better?"**
Because reducing reprefill-per-event isn't the same as reducing total reprefill.
Append-only's divergence point does creep later every event (confirmed, monotonic,
14/14) — but its own context keeps growing right along with its cached prefix, because
`frozen` never shrinks. The bigger-cached-prefix and bigger-total-context effects nearly
cancel out in absolute reprefilled-token terms. This is the single most important
mechanistic finding of the project and worth explaining with the concrete traj-000
numbers (§2, Phase 2 section above) if asked to go deep.

**"Why does kv_evict get worse instead of better?"**
Because "near-free, zero re-prefill" is a property of live in-engine session management
where old KV blocks are freed and attention remaps around them — something this
project's request-based (independent HTTP call per step) harness architecturally can't
express. The only lever available is sending a shorter *text* prompt, and a fixed-size
sliding window's front boundary shifts every single step, so it invalidates the cache
almost every step (150 events vs 115 for naive) instead of rarely. It's evidence that
implementing the *idea* without the *mechanism* it depends on can be actively harmful,
not neutral.

**"Why does the hybrid policy actually work — what's the mechanism, not just the number?"**
Two independent effects, both confirmed from the event log, not asserted: it fires fewer
compaction events in total (97 vs 115-150, because keep/evict routing removes turns from
ever needing compaction), and each event that does fire is individually cheaper to
reprefill (3,799 tokens vs 5,092-6,177) while saving more (5,692 vs 3,676-4,756). Its
reprefill:saved ratio (0.71) is the only one below 1.0 of any policy tested. In plain
terms: stop treating all context uniformly — a giant tool dump you'll never reference
again and a two-line status update shouldn't be compacted the same way.

**"Your quality metric was 0/5 for everything — doesn't that undermine the whole
project?"**
No, and this is worth being direct about rather than defensive. It's a real finding: a
0.6B/1.7B model cannot reliably track exact numeric state over ~100 tool calls,
*regardless of compaction policy* — every policy shares the identical underlying model,
so this is a model-capability floor, not something layout touches. Going from 0.6B to
1.7B made mechanical task-following *worse*, not better, which is itself a genuine,
slightly counterintuitive result reported plainly. What *does* survive underneath the
floor: hybrid's episodes got through nearly the entire 50-item queue before failing
(110.6 tool calls, 0.6 items left unprocessed) vs the other policies failing both
mechanically and arithmetically (~93-95 calls, ~9 left) — real evidence hybrid's
keep-verbatim routing protects exactly the short numeric turns this task depends on,
even though it didn't flip any episode to a full success.

**"How do you know your measurements are trustworthy? What stops this from being GPU
noise?"**
Every step is cross-checked against a pure, GPU-free block-math prediction
(`bench/divergence.py`) derived from comparing consecutive steps' token-id sequences —
if measured cache reuse ever falls short of that prediction, it raises loudly rather
than being silently accepted. Every comparison is interleaved A/B (never "all of policy
A, then all of policy B") specifically to control for trajectory-order and thermal
drift. Every claimed improvement gets a paired Wilcoxon significance test across seeds,
not just a difference in medians. And multiple points in this project (Phase 2's
settle-timing misdiagnosis, Gate 0's warm-server bug) are examples of catching the
measurement methodology itself being wrong before trusting a number.

**"What would you do with a bigger GPU / more time?"**
Directly answered by the cost model's extrapolation section: validate the same
efficiency-fit methodology on real mid-size hardware, get quality above the 0/5 floor
(bigger model, easier task, or a harness re-tuned specifically for the larger model
rather than reused as-is from the smaller one), sweep hybrid's thresholds
(`large_tool_output_tokens`, `protect_recent_turns`) instead of the single hand-picked
configuration tested, and revisit offload with the classification-caching fix identified
but not implemented (cache the worth-offloading decision by content hash instead of
reclassifying the full prompt from scratch on every request).

**"What was the hardest engineering problem, not the hardest research problem?"**
Getting a small local model to reliably emit valid, on-schema tool calls at all, for
elicitation experiments with no in-context precedent to imitate (Phase 3's three legs).
Free-text failed in three different ways before landing on grammar-constrained decoding
plus a carefully-chosen exemplar. Close second: the WSL2/GPU environment itself —
thermal throttling, a full VM crash, a paravirtualization driver desync, and a leaked
subprocess bug that only showed up once reproducibility testing forced a real failure
path to execute.

**"If you started over, what would you do differently?"**
Record more trajectories at the start (30 per spec, not 14) and pick a task the small
models can actually make partial progress on, so the quality axis isn't degenerate —
both would have let later phases (Gate 4's Pareto plot, the cost model's aggregate
sanity check) say more. Would also standardize output filenames per sweep from day one
instead of hitting the append-not-overwrite footgun twice.

**"Explain the KV cache / prefix caching mechanism itself, for someone who doesn't know
vLLM."**
Inference engines split the prompt into fixed-size token blocks and cache each block's
computed key/value tensors by a hash of its exact token content. On the next request, any
prefix of blocks whose hash matches a previous request's is reused without recomputation
— that's why keeping a prompt's *prefix bytes* unchanged is what actually matters, not
just "using fewer tokens." Change one token anywhere in the prefix (e.g. rewriting a
summary in place) and every block from that point onward gets a different hash and must
be recomputed from scratch, even if 99% of the content after it is unchanged.

**"What's not done / what are the honest weaknesses of this project?"**
Say this plainly, it reads well: Phase 5 (KV re-basing/selective recompute) was
deliberately descoped, not attempted. The quality axis never got above a 0/0 floor on
any policy on this task/model combination. `run_all.py`'s full live `--smoke` pipeline
was never verified end-to-end in one sitting (environment instability, not a code
defect) — verified instead that it didn't corrupt any committed data. The 70B/H100
numbers are a FLOPs-based projection, never a real measurement on that hardware. All of
this is written into the README's limitations section and the technical writeup on
purpose, not omitted.

---

## 7. Anything else worth knowing before an interview

- **Hardware/environment, if asked "how did you even run this":** single RTX 4060
  Laptop, 8GB VRAM, WSL2 Ubuntu 24.04 on Windows 11. vLLM 0.8.5 V1 engine. Qwen3-0.6B
  (fallback/default measurement model, 16384 context, ~29,150-token KV capacity) and
  Qwen3-1.7B (primary, 8192 context, ~9,456-token KV capacity, used in Phase 1 and
  Phase 3.3's primary-model rerun). No API keys, no cloud spend, everything local.
- **Every figure/number in the repo comes from committed parquet data, generated by a
  script, never hand-edited** — spec's own non-negotiable rule (§7), followed
  throughout, including for the demo deck (which regenerates from committed data rather
  than reusing a stale, model-mismatched Phase 1 figure).
- **Statistical rigor throughout:** ≥5 seeds/trajectories per configuration reporting
  median + IQR, interleaved ordering always, paired Wilcoxon tests for every claimed
  improvement, Wilson confidence intervals for the binomial task-success rate.
- **You (the user) made the real judgment calls at every genuine fork** — this project's
  own summaries repeatedly present forks explicitly (e.g. §4.1's "measure honestly vs
  invest in live session management vs analytical-only," §4.5's "how to fill the 5th
  Pareto slot," §6.1's dashboard precompute-vs-placeholder-vs-drop choice) rather than
  silently picking a direction. Worth mentioning if asked how you approached ambiguous
  engineering decisions: presented options with tradeoffs, decided together, moved on.
- **This document plus `TECHNICAL_WRITEUP.md` and `README.md` are not fully redundant:**
  README is the pitch, the writeup is the 2,000-3,000-word narrative for a reader who
  wants the story once, this document is the exhaustive reference for *your* prep —
  it's fine (expected) for this one to be longer and more granular than either.
