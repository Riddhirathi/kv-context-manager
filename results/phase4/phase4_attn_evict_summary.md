# Phase 4.2 Summary — Attention-Score Eviction (Not Built: Infeasibility Finding)

Status: **not implemented**. Investigated to the point of an empirically-demonstrated
infeasibility finding on this hardware, not abandoned early — this document is that
finding, not a placeholder for future work assumed to be easy.

## What spec §4.2 asks for

> "`policies/attn_evict.py` — score-based eviction. Retain KV entries with the
> highest accumulated attention mass (H2O/SnapKV family). More expensive to compute,
> should retain more."

## Why this needed investigation before writing a policy

Phase 4.1 (`kv_evict.py`) already established that vLLM's serving kernels
(FlashAttention-based) never materialize real per-token attention weights — by
design, for speed (see `results/phase4/phase4_kv_evict_summary.md`). The same is true here:
"accumulated attention mass" requires real attention weights, which vLLM's serving
stack cannot provide. The one path that stays outside vLLM internals (this project's
consistent approach for every L1/L2 policy) is a *separate* eager-attention model
instance via plain `transformers` — genuinely obtainable, not a proxy, just not from
the serving engine.

## What was verified to work

- `AutoModelForCausalLM.from_pretrained(..., attn_implementation="eager")` on
  `Qwen/Qwen3-0.6B` (28 layers, 16 heads, head_dim 128) returns real per-layer
  `(hidden_states, attn_weights)` tuples from each `Qwen3Attention` module —
  confirmed by direct inspection with a forward hook on every layer.
- A hook that captures the target layer's attention weights, reduces them to a
  per-token-position "mass" (sum over heads and query positions), and then replaces
  the layer's returned attention weights with `None` — preventing the outer decoder
  loop from retaining that layer's tensor — works correctly and produces the
  expected shape at trivially small inputs (8 tokens).

## What was verified NOT to work: real memory and timing costs on this hardware

The theoretical plan was that bounding capture to one layer, with sequential
layer execution, would bound peak memory to roughly one layer's attention tensor
(~1-2 GB estimated at a realistic multi-thousand-token context). This was wrong in
practice, discovered incrementally rather than assumed:

1. **A first test at 6000 tokens crashed the entire WSL VM** (`Wsl/Service/
   E_UNEXPECTED`), not just the Python process — recovered via `wsl --shutdown`
   from Windows. WSL's actual memory budget here is 7.6 GB total, far below what a
   naive full-28-layer materialization would need (tens of GB), and apparently
   also below what even the intended one-layer-bounded version needed at that
   sequence length.
2. **A follow-up attempt to contain the risk with `RLIMIT_AS` failed for an
   unrelated reason**: `RLIMIT_AS` caps virtual address space, which `safetensors`'
   `mmap`-based model loading trips even when real (resident) memory usage would be
   modest — model loading itself failed under a 5 GB cap before any scoring
   computation even began.
3. **Retested with no artificial cap, at genuinely small sequence lengths, isolating
   real costs**: model loads cleanly (bf16, 544 MB peak). But a single forward pass
   already costs far more than the attention-tensor size alone would suggest —
   100 tokens: 1,716 MB peak, 2.3s. 200 tokens: 1,783 MB, 4.7s. 300 tokens: 1,867 MB,
   8.3s. Both memory (a large, roughly-constant ~1.2 GB jump from baseline even at
   100 tokens — likely reference-implementation overhead in HF's "eager" attention
   path, not just the nominal weights tensor) and time (worse than linear scaling,
   consistent with the expected O(n²) eager-attention cost) make realistic
   multi-thousand-token contexts — this project's actual working set, since the
   compaction threshold is 60% of a 16k-token window — impractical: likely minutes
   per single scoring call, needed potentially 100+ times across a full sweep
   (`kv_evict` alone fired 150 times across 14 trajectories in §4.1).

## The decision

Presented this reversal to the user directly rather than silently downgrading the
approach: the "real eager-mode scores" path they'd approved was now demonstrated
infeasible, not just risky, with real cost already paid (one WSL crash). Given three
options (skip and document / cheap non-attention proxy / cap the scored window
small enough to stay safe), **chosen: skip §4.2, document this finding, proceed to
§4.3** — a cheap proxy would no longer be a faithful "attention mass" implementation
(spec's own "more expensive to compute, should retain more" framing wouldn't apply),
and a small capped window would exclude most of the context from ever being
selectable for retention, undermining any real comparison against naive/
append_only/kv_evict.

## What this shows

Genuine H2O/SnapKV-style attention-score eviction needs either (a) a GPU/host with
meaningfully more memory headroom than this 8 GB laptop card's ~7.6 GB WSL budget,
or (b) a fundamentally different scoring strategy that doesn't require full
`output_attentions=True` eager materialization (e.g. approximate attention-mass
estimators, or scoring only at KV-cache-write time inside a custom attention kernel
— genuinely L3-internals territory, which spec explicitly defers past Phase 4). This
is not evidence that attention-score eviction is a bad *idea* — only that this
specific hardware and this specific "stay outside vLLM" implementation strategy
can't produce it. Worth stating plainly rather than either quietly skipping it or
overclaiming a proxy as the real thing.

## Known limitations of this finding itself

- **Tested only up to 300 tokens** on the safe path — the real ceiling could be
  somewhat higher (maybe 500-1000 tokens) before becoming truly impractical, but
  this project's contexts routinely reach several thousand tokens before
  compaction fires, so the qualitative conclusion (infeasible at the scale needed)
  is unlikely to change.
- **Single-layer scoring was the design** (last layer, following common H2O-lite
  practice) — a different layer or a coarser scoring method (e.g. only every Nth
  token) wasn't tried, though neither addresses the fundamental O(n²) eager-attention
  cost driving both the memory and timing problems.
- **No GPU placement was tried for the scorer** — CPU was chosen specifically to
  avoid VRAM contention with vLLM's concurrently-running server (the risk flagged
  before starting this investigation); a GPU-placed scorer might have different
  (possibly better, possibly worse given the 8GB card's own tightness) memory
  characteristics, not measured here.

## Next

§4.3 (`kv/offload.py`, via vLLM's real, confirmed `KVConnectorBase_V1` interface)
does not have this class of problem — it's a documented, supported extension point,
not something requiring a workaround outside vLLM. Proceeding there next.
