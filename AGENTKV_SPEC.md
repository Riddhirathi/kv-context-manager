# AgentKV — Cache-Aware Context Compaction for Long-Horizon LLM Agents

**Project spec / requirements document. Written for an AI coding agent (Claude Code) plus a human owner.**

Hardware target: single NVIDIA RTX 4060 Laptop (8 GB VRAM, Ada / SM 8.9, PCIe 4.0 x8), 16–32 GB system RAM, Linux preferred (WSL2 acceptable with caveats noted in §9).

---

## 1. One-paragraph problem statement

Long-horizon LLM agents append to their context on every step, which makes them cheap to serve: the KV cache from step *N* is a valid prefix for step *N+1*, so only the newly appended tokens are prefilled. **Compaction breaks this.** When an agent summarizes or prunes its history, it rewrites tokens in the *middle* of the context. Because causal attention makes every KV entry depend on all preceding tokens, the prefix cache becomes invalid from the first changed token onward, forcing a full re-prefill of everything after it. A 100-step agent that compacts eight times pays eight large re-prefills that nobody is currently measuring, because the entire compaction literature treats this as a prompting problem. It is a serving problem.

**Thesis to prove or refute:** a context manager that is aware of the KV cache's block structure can preserve most of that cache across compaction events, and a hybrid of token-space and KV-space compaction dominates both pure strategies on the (prefill cost × task quality) Pareto frontier.

---

## 2. Goals and non-goals

### Goals

- G1. **Measure** the cache-invalidation cost of standard compaction, precisely, with reproducible numbers.
- G2. **Reduce** it via a cache-aware context layout, without changing the model.
- G3. **Compare** token-space compaction against KV-space eviction on one common axis.
- G4. **Prove no quality regression** using programmatic verification, not vibes.
- G5. Produce an **analytical cost model** that extrapolates single-GPU measurements to production model sizes.

### Non-goals

- Not training or fine-tuning any model. Inference only.
- Not building a new serving engine. Build *on top of* and *inside* vLLM.
- Not claiming state-of-the-art agent accuracy. The agent's absolute task performance is irrelevant; only the *delta* between compaction policies matters.
- Not multi-GPU, not distributed, not production-hardened.

### Explicit success criteria for the repo

The project is "done" when a reader can run `make reproduce` on a single consumer GPU and regenerate every figure in the report from scratch in under 6 hours.

---

## 3. Key concepts the implementation depends on

(Plain-language versions of all of these live in `CONCEPTS_PRIMER.md`. This section is the precise version.)

| Term                         | Operational definition used in this repo                                                                                                                                                    |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Prefill**            | The forward pass over all prompt tokens that populates the KV cache. Compute-bound.                                                                                                         |
| **Decode**             | Autoregressive generation, one token per forward pass. Memory-bandwidth-bound.                                                                                                              |
| **KV cache**           | Per-layer Key/Value tensors for every past token. Bytes per token =`2 × n_layers × n_kv_heads × head_dim × dtype_bytes`.                                                              |
| **Block / page**       | vLLM stores KV in fixed-size blocks (default`block_size = 16` tokens). Cache reuse is granular to blocks, not tokens.                                                                     |
| **Prefix cache (APC)** | vLLM hashes block contents; a new request reuses cached blocks for its longest matching*prefix*. Divergence at token *i* invalidates every block from `floor(i / block_size)` onward. |
| **Cache hit rate**     | `num_cached_tokens / num_prompt_tokens`, reported by vLLM per request.                                                                                                                    |
| **Anchor**             | The immutable head of the context: system prompt + tool schemas + original task. Never rewritten.                                                                                           |
| **Frozen segment**     | A summary chunk, written exactly once and never edited afterwards. Segments accumulate append-only.                                                                                         |
| **Live window**        | The most recent K turns, kept verbatim.                                                                                                                                                     |
| **Compaction event**   | Any operation that reduces context length. Two families:*token-space* (rewrite text) and *KV-space* (drop KV entries, text unchanged).                                                  |
| **Divergence point**   | The first token index where the new context differs from the previously cached one. Determines re-prefill cost.                                                                             |
| **Action agreement**   | Whether the model, given a compacted context, emits the same next tool call as it would given the full context. Primary quality proxy.                                                      |

---

## 4. System architecture

Three layers, deliberately ordered from lowest to highest implementation risk.

```
┌─────────────────────────────────────────────────────────────┐
│  L3  KV-space layer  (INSIDE vLLM — highest risk, Phase 5+) │
│      block eviction · RoPE re-basing · selective recompute  │
├─────────────────────────────────────────────────────────────┤
│  L2  Serving layer   (vLLM config + KV connector, Phase 4)  │
│      prefix caching · CPU offload · eviction policy          │
├─────────────────────────────────────────────────────────────┤
│  L1  Context layer   (OUTSIDE vLLM — lowest risk, Phase 2)  │
│      anchor / frozen segments / live window · block align   │
└─────────────────────────────────────────────────────────────┘
```

**Critical design note for the implementing agent:** L1 requires *zero* modification to vLLM. It works purely by controlling which token sequence gets sent. This means Phases 0–3 produce publishable results even if Phase 5 never lands. Do not attempt L3 before L1 numbers are locked in.

### Repository layout

```
agentkv/
├── README.md
├── Makefile                     # make setup / bench / reproduce / demo
├── pyproject.toml
├── configs/
│   ├── model.yaml               # model, dtype, block_size, gpu_mem_util
│   ├── policies/*.yaml          # one file per compaction policy
│   └── experiments/*.yaml       # experiment matrices
├── src/agentkv/
│   ├── context/
│   │   ├── segments.py          # Segment, ContextState dataclasses
│   │   ├── layout.py            # renders ContextState -> token ids
│   │   └── alignment.py         # block-boundary padding/alignment
│   ├── policies/
│   │   ├── base.py              # CompactionPolicy ABC
│   │   ├── naive.py             # baseline: rewrite the middle
│   │   ├── append_only.py       # anchor + frozen segments + window
│   │   ├── kv_evict.py          # StreamingLLM sinks + window
│   │   ├── attn_evict.py        # attention-score eviction (H2O/SnapKV-lite)
│   │   └── hybrid.py            # value-based per-segment routing
│   ├── serving/
│   │   ├── engine.py            # vLLM wrapper, exposes cache stats
│   │   ├── harness.py           # minimal HF+custom paged cache, for L3 dev
│   │   └── cost_model.py        # FLOPs + $ extrapolation
│   ├── kv/
│   │   ├── rebase.py            # RoPE re-basing of shifted blocks
│   │   ├── recompute.py         # CacheBlend-style selective recompute
│   │   └── offload.py           # CPU/NVMe block offload + eviction
│   ├── bench/
│   │   ├── replay.py            # deterministic trajectory replayer
│   │   ├── tasks/
│   │   │   ├── ledger.py        # synthetic verifiable long-horizon task
│   │   │   └── probes.py        # needle-at-depth retention probes
│   │   └── agreement.py         # next-action agreement scorer
│   ├── metrics/
│   │   ├── collector.py         # per-step records -> parquet
│   │   └── rigor.py             # clock lock, warmup, interleaved A/B
│   └── viz/
│       ├── dashboard.py         # live demo UI
│       └── figures.py           # all paper figures
├── experiments/                 # runnable scripts, one per figure
├── results/                     # committed parquet + PNG, small files only
├── trajectories/                # recorded agent traces (committed)
└── tests/
```

---

## 5. Phased implementation plan

Each phase has a **deliverable**, an **acceptance test**, and an **exit gate**. Do not start phase N+1 until phase N's gate passes.

---

### Phase 0 — Measurement harness (target: 5–7 days)

**0.1 Environment**

- vLLM (V1 engine) on the 4060. Model: `Qwen/Qwen3-1.7B` primary, `Qwen/Qwen3-0.6B` fallback if VRAM-tight. Both support tool calling.
- Set `gpu_memory_utilization` empirically — start at 0.85, back off on OOM. Log the resulting KV cache capacity in tokens; this number goes in the README.
- Confirm prefix caching is enabled and that `num_cached_tokens` is exposed per request.

**0.2 Benchmark rigor module (`metrics/rigor.py`)**
This is not optional polish. On a thermally-throttled laptop, sloppy measurement produces fake results.

- Lock GPU clocks via `nvidia-smi -lgc` where permitted; if not permitted, detect and record clock drift per run and fail loudly if it exceeds 5%.
- Warmup: discard the first N=3 trajectories.
- **Interleaved A/B**: never run all of policy A then all of policy B. Alternate them, so thermal drift affects both equally. This is a hard requirement.
- Report median + IQR across ≥5 seeds, never a bare mean.
- Record: GPU temp, clock, power, per-step, into the metrics parquet.

**0.3 Metrics collector**
Per agent step, record: `step_idx, prompt_tokens, cached_tokens, prefill_tokens, ttft_ms, decode_ms, output_tokens, gpu_mem_bytes, policy, seed, temp_c, sm_clock_mhz`.

**0.4 Trajectory replayer (`bench/replay.py`)**
Deterministic replay is the backbone of the whole project. Record real agent trajectories **once** using a strong model via a free API tier (Groq / Cerebras / Gemini free tier all work). Store as JSONL: list of `{role, content, tool_calls, tool_results}`. Then replay them locally against the small model. This decouples "is the agent good?" (irrelevant) from "what does compaction cost?" (the actual question).

Record at minimum:

- 30 trajectories of ≥80 steps each
- Mixed tool-output sizes, including some very large ones (log dumps, file contents) — these are what compaction actually targets
- Store raw, uncompacted, so every policy sees identical input

> **Gate 0:** Running the same trajectory twice under the same policy produces TTFT medians within 5% of each other, and cache hit rate is bit-identical. If you can't reproduce your own numbers, nothing downstream is meaningful.

---

### Phase 1 — Quantify the bug (target: 4–5 days)

**1.1 Implement `policies/naive.py`** — the industry-standard baseline. When context exceeds a threshold (e.g. 60% of window), replace the oldest 50% of turns with an LLM-generated summary placed at that same position. This is what everyone ships today.

**1.2 The cliff figure.** Plot cumulative prefill tokens vs. agent step, for a full 100-step trajectory. Expect a staircase: flat-ish growth punctuated by vertical jumps at each compaction event. **This single figure is the entire justification for the project.** Make it beautiful.

**1.3 Instrument the divergence point.** For every compaction event, log: divergence token index, blocks invalidated, tokens re-prefilled, wall-clock penalty, and the ratio `re-prefilled tokens / tokens actually saved by compacting`.

**1.4 Sanity check against theory.** Predicted invalidated blocks = `n_blocks_total - floor(divergence_idx / block_size)`. Measured must match predicted exactly. If it doesn't, your understanding of the cache is wrong — stop and fix that first.

> **Gate 1:** You can state, with error bars, "standard compaction costs X extra prefill tokens per 100-step trajectory, which is Y% of total prefill compute and Z% of wall clock." If X is small, the project premise is wrong — say so publicly and pivot. A well-measured negative result is still a good writeup.

---

### Phase 2 — Cache-preserving context layout (target: 8–10 days)

The core L1 contribution. No vLLM internals touched.

**2.1 `ContextState` model**

```
ContextState:
  anchor:          Segment            # system + tools + task, immutable
  frozen:          list[Segment]      # append-only summaries, never edited
  live:            list[Turn]         # verbatim recent turns
```

**2.2 Append-only compaction (`policies/append_only.py`)**
On a compaction event at step *t*:

1. Select the oldest turns in `live` for retirement.
2. Summarize them into a new `Segment` S_t.
3. **Append** S_t to `frozen`. Never modify S_1..S_{t-1}.
4. Drop the retired turns from `live`.

Resulting divergence point = end of `frozen[:-1]`, i.e. everything before the newest summary stays cached. Contrast with naive, where divergence = start of the summarized region.

**Key property to verify and highlight:** the preserved prefix *grows monotonically over the trajectory*. Later compactions are cheaper than earlier ones. This is the opposite of naive behavior and is the headline systems result.

**2.3 Block alignment (`context/alignment.py`)**
Cache reuse is block-granular. A divergence one token before a block boundary wastes an entire block. Pad segment boundaries to `block_size` multiples using a neutral filler that does not perturb model behavior (validate this claim empirically — measure action agreement with and without padding). Report the marginal gain from alignment separately; it will be small but it demonstrates that you understand the paging model.

**2.4 Anchor hygiene**
Tool schemas often dominate the anchor. Verify the anchor is byte-stable across steps — a single timestamp or reordered JSON key in the system prompt destroys 100% of the cache. Add a regression test that hashes the anchor every step and fails on change. *This bug is extremely common in real agent frameworks and is worth calling out in the writeup.*

> **Gate 2:** Append-only layout reduces cumulative prefill tokens per 100-step trajectory by a measurable, statistically significant margin vs. naive, at equal or better action agreement.

---

### Phase 3 — Quality measurement (target: 5–6 days)

Runs in parallel with Phase 2 conceptually, but gate it separately because it is what makes the results credible.

**3.1 Next-action agreement (`bench/agreement.py`)**
For each step in a replayed trajectory, compute the model's next action given (a) full uncompacted context and (b) compacted context. Score exact tool-name match, argument match, and a semantic similarity fallback. Report agreement rate per policy.

**3.2 Retention probes (`bench/tasks/probes.py`)**
Inject synthetic facts at controlled depths in the trajectory ("the account ID is X", "the user prefers Y"). At step 100, query for each. Produces a *retention-vs-depth curve* per policy. This is how you show precisely what each policy forgets.

**3.3 Synthetic verifiable task (`bench/tasks/ledger.py`)**
Build one long-horizon task a 1.7B model can actually complete, with programmatic verification: e.g. an inventory/ledger environment where the agent makes ~100 tool calls that mutate state, and the final reported balance is checked against ground truth. Binary reward. This is your real "task success rate" number.

**3.4 Report all three.** Agreement is cheap and dense; probes are diagnostic; task success is the honest bottom line. Never report only the flattering one.

> **Gate 3:** For every policy, you can produce a (prefill cost, task success) point with error bars.

---

### Phase 4 — KV-space eviction and the Pareto frontier (target: 8–10 days)

**4.1 `policies/kv_evict.py`** — StreamingLLM-style: retain the first few "attention sink" tokens plus a sliding window; discard the middle *in KV space*. Text is never rewritten, so **there is zero re-prefill**. Cost is near-free; the question is what it costs in quality.

**4.2 `policies/attn_evict.py`** — score-based eviction. Retain KV entries with the highest accumulated attention mass (H2O/SnapKV family). More expensive to compute, should retain more.

**4.3 CPU offload (`kv/offload.py`)** — instead of discarding evicted blocks, page them to host RAM via vLLM's KV connector interface (LMCache is the reference implementation). Implement a **trajectory-aware** eviction policy that beats LRU by exploiting agent structure: the anchor is never evictable; frozen segments are re-referenced predictably; large tool outputs are usually referenced once and never again.

> Your PCIe 4.0 x8 link (~16 GB/s) is a *feature* for this experiment. On an NVLink server the transfer is nearly free and policy quality is invisible. On your laptop, a bad policy is measurably bad. Say this explicitly in the writeup.

**4.4 `policies/hybrid.py`** — the contribution. Per segment, choose one of three actions:

- keep verbatim (cheap, no quality loss, costs context length)
- KV-evict (free prefill, lossy, unrecoverable)
- textually summarize into a frozen segment (costs re-prefill from divergence, semantically smart)

Route by estimated value: predicted future attention mass × segment size × recompute cost. Start with a hand-tuned heuristic. A learned policy is a stretch goal, not a requirement.

**4.5 The Pareto plot.** X = cumulative prefill tokens (or $/trajectory), Y = task success rate. One point per policy per configuration. **This is the money figure of the whole project.** The claim to test: hybrid sits below-and-right of both pure strategies.

> **Gate 4:** A Pareto plot exists with ≥5 policies and ≥5 seeds each, and you can defend every point on it.

---

### Phase 5 — Stretch: KV re-basing and selective recompute (research-grade, may fail)

**Only attempt after Phases 0–4 are written up and committed.** This is genuinely hard and might not work. That is acceptable; a documented negative result here is still strong material.

**5.1 The observation.** After compaction, surviving turns are textually *identical* but sit at *different positions*. Their KV is not reusable by vLLM's hash-based APC because the prefix diverged — but it's "nearly right."

**5.2 RoPE re-basing (`kv/rebase.py`).** RoPE applies a position-dependent rotation to K. Shifting a block from position p to p' is a cheap elementwise rotation by (p' − p) — no attention, no MLP. **At layer 0 this is exact**, because layer-0 K/V depend only on the token embedding and its position.

**5.3 Depth-dependent error.** At layer ≥1, K/V depend on attention outputs over *removed* tokens, so re-basing is approximate and error compounds with depth. Measure this: per-layer cosine similarity between re-based and freshly-computed KV. Publish the curve. It is interesting regardless of outcome.

**5.4 Selective recompute (`kv/recompute.py`).** CacheBlend's insight: recompute only the top-k% of tokens with highest KV deviation, reuse the rest. Sweep k from 0 to 100 and find the knee. If you can hit full-recompute quality at k ≈ 15%, that is a ~6× prefill reduction on top of everything in Phase 2.

**5.5 Honest reporting.** State clearly that this is approximate and validate against task success, not just cosine similarity. Approximation methods that look great on similarity metrics and quietly destroy agent behavior are the standard failure mode here.

---

### Phase 6 — Cost model, packaging, demo, writeup (target: 6–8 days)

**6.1 Analytical cost model (`serving/cost_model.py`)**
Your measurements are on a 1.7B model. Show the effect scales. Prefill FLOPs ≈ `2 × N_params × n_tokens` plus attention terms; derive tokens-saved → FLOPs-saved → GPU-seconds-saved → dollars, parameterized by model size and hardware. **Validate the model against your own measurements first**, then extrapolate to 70B on an H100 and state the assumptions and the error bars. Do not present extrapolated numbers as measurements.

**6.2 Reproducibility.** `make reproduce` regenerates every figure. Pin all versions. Commit the trajectories. Document the exact `gpu_memory_utilization` and resulting KV capacity.

**6.3 README structure.** Figure first (the cliff plot), then the one-sentence result, then the Pareto plot, then how to reproduce, then the limitations section. **Write the limitations section honestly and prominently** — it is the single strongest credibility signal in a portfolio repo.

**6.4 Technical writeup.** 2,000–3,000 words. Include what didn't work. Include the layer-depth error curve even if Phase 5 failed.

---

## 6. Demo design

Two artifacts. Build both.

### 6.1 Live side-by-side dashboard (`viz/dashboard.py`)

The 90-second version. Two panes, same trajectory, same model, different policy:

- **Left:** naive compaction. **Right:** your hybrid manager.
- **Context map strip** — a horizontal bar of blocks, one cell per KV block, colored green (cache hit) / red (invalidated this step) / grey (offloaded to CPU). At a compaction event the naive pane goes red from the middle to the end in one frame; yours reddens only the tail. **This is the visual that sells the entire project in three seconds.**
- **Live counters:** cumulative prefill tokens, TTFT for the current step, elapsed wall clock, estimated $ at production API rates.
- **A running quality readout:** action agreement so far, so nobody can accuse you of buying speed with accuracy.

Use whatever renders fast and reliably — a simple web UI or a terminal TUI both work. Do not over-engineer this; it must run reliably during an interview on a laptop with no internet.

### 6.2 Three-figure deck

1. **The cliff.** Cumulative prefill vs. step, naive policy. "Here is a cost nobody is measuring."
2. **The fix.** Same axes, all policies overlaid. "Here is the cost removed."
3. **The Pareto frontier.** Cost vs. task success. "Here is proof I didn't cheat."

Rehearse a 60-second and a 5-minute version. The 60-second version is: problem → cliff figure → one-line mechanism → Pareto plot → "and it's reproducible on an 8 GB laptop."

---

## 7. Experimental protocol (non-negotiable)

- ≥5 seeds per configuration, report median and IQR.
- Interleaved A/B ordering, always.
- Every claimed improvement needs a paired statistical test across seeds.
- Ablate individually: layout alone, alignment alone, offload alone, hybrid routing alone. A single number for the whole system is not defensible in an interview.
- Log every run to parquet; figures are generated from committed data, never hand-edited.

---

## 8. Risk register

| Risk                                                     | Likelihood | Mitigation                                                                                                               |
| -------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------ |
| The cache-invalidation cost is smaller than hypothesized | Medium     | Gate 1 catches it early. Pivot to "here is why the intuition is wrong," which is still a publishable measurement.        |
| vLLM internals shift and break L3 work                   | High       | Pin the vLLM version. Keep L1/L2 contributions independent of internals.                                                 |
| 8 GB is too tight for a useful context length            | Medium     | Fall back to Qwen3-0.6B; the*relative* effect is model-size-independent, and the cost model handles extrapolation.     |
| Thermal throttling corrupts timings                      | High       | `metrics/rigor.py` is a Phase 0 requirement precisely for this.                                                        |
| Phase 5 doesn't work                                     | High       | It's explicitly a stretch. Ship Phases 0–4 as a complete project first.                                                 |
| Small model can't complete any agent task                | Medium     | The synthetic ledger task in 3.3 is designed to be within a 1.7B model's reach. Action agreement is the backstop metric. |

---

## 9. Environment notes

- Linux native strongly preferred. Under WSL2, `nvidia-smi -lgc` is typically unavailable and CPU-offload bandwidth measurements are unreliable — if you must use WSL2, document it and treat offload numbers as indicative only.
- Pin: vLLM version, PyTorch, CUDA, transformers. Record in `README.md`.
- Prefer bf16. Ada supports it natively.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation on 8 GB.

## 10. Coding standards for the implementing agent

- Type hints everywhere; `mypy` clean.
- Every policy implements the same `CompactionPolicy` ABC. Adding a policy must require touching exactly one new file.
- No magic numbers in code — everything in `configs/`.
- Tests for: block-alignment math, divergence-point calculation, and the cost model. These three are where silent bugs will invalidate results.
- Deterministic seeding across numpy / torch / the sampler.
- Do **not** write results-generating code that depends on network access at run time. Trajectories are recorded once and committed.
