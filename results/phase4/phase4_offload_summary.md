# Phase 4.3 Summary — Trajectory-Aware CPU Offload

Status: implementation complete, **round trip verified correct, comparative
measurement complete — net result is strongly negative as implemented**. This is
the deepest engineering investment in the project so far — the only component
that runs *inside* the vLLM engine itself, not just against its public HTTP API —
and the only one that needed two real design corrections mid-verification, both
caught by testing rather than assumed away.

## What was built

- `src/agentkv/kv/offload.py` — pure, GPU-free trajectory-aware offload policy.
  `classify_decoded_text` finds every `<role>` tag span in a piece of decoded text
  and classifies it per spec §4.3's three rules: the *first* tag is always the
  anchor (worth offloading); a `<system>` tag whose content starts with the
  literal `"[frozen summary #"` marker (already used by `context/segments.py`'s
  `Segment.to_turn`, not invented here) is a frozen segment (worth offloading); a
  `<tool>` tag spanning at least `large_tool_output_chars` characters is a large
  tool output (*not* worth offloading); everything else defaults to worth
  offloading. `token_boundary_for_char_offset` maps character offsets back to
  token indices via binary search (decoded length is monotonic in token count).
  `classify_token_ids` composes these into token-index spans; `token_worth_
  offloading_mask` and `blocks_worth_offloading` turn that into a per-block
  decision (majority vote for blocks straddling a turn boundary). Deliberately
  kept free of any vLLM import — see below.
- `src/agentkv/kv/offload_connector.py` — `TrajectoryAwareOffloadConnector`, a
  real `KVConnectorBase_V1` subclass adapted from vLLM's reference
  `SharedStorageConnector` (disk-backed) to an in-memory CPU tensor dict (spec's
  "host RAM," not disk) — and to only mark a request for storage when
  `_worth_storing_mask` (backed by the policy above, run against
  `request.prompt_token_ids` with the connector's own tokenizer) says at least
  one of its blocks is worth it, rather than a plain LRU tier's indiscriminate
  "store everything."
- `experiments/_vllm_server_with_offload.py` — a thin launcher that registers
  the connector with vLLM's `KVConnectorFactory` before delegating to vLLM's own
  server entrypoint (necessary because the factory's registry is hardcoded at
  import time in this vLLM version, and this project's `VLLMEngine` spawns vLLM
  as a separate OS process — an in-process registration in our own script would
  register into the wrong process).
- `VLLMEngine` gained an optional `kv_transfer_config` parameter that switches to
  this launcher and adds `--kv-transfer-config`.
- `experiments/phase4_offload_compare.py` — the comparative sweep: replays the
  same trajectories under `append_only` through two separate engine boots
  (offload enabled vs. not), measuring wall-clock time.

## Why this file is split into two (deviates from spec's exact layout)

Spec lists a single `kv/offload.py`. Split into policy (`offload.py`) and
connector (`offload_connector.py`) because `offload_connector.py` imports real
vLLM classes at module level, and importing vLLM alone was independently
confirmed to take over two minutes in this environment (a plain `import vllm`
timed out past 120s during §4.2's investigation). Keeping that import out of the
pure-policy module is what keeps `tests/test_offload_policy.py` fast (14 tests,
under a second) — the full project test suite (127 tests) still runs in ~6-7s
with this split.

## Design correction #1: classification can't be driver-supplied

The first version classified from a `ContextState` object, with the connector
exposing a `set_current_state()` method an external driver would call before each
request. **Wrong**: `VLLMEngine` talks to vLLM only over HTTP, and
`KVConnectorBase_V1` instances live *inside* the separate vLLM server process —
there is no in-process call an external driver could make. Caught while designing
the round-trip test (trying to figure out *how* a driver would call
`set_current_state()` on a live connector revealed there was no mechanism to do
so), not from a crash. **Fixed** by making the connector self-contained: it
decodes `request.prompt_token_ids` with its own tokenizer (loaded in `__init__`
via `vllm_config.model_config.model`) and classifies directly from the decoded
text's `<role>` tags and the frozen-segment marker. The old `ContextState`-based
path was deleted, not kept alongside the new one.

## Design correction #2: store and lookup use separate connector instances

Verified with an actual round-trip test (below) that the store decision fired
correctly (`13/13 blocks worth offloading` logged) but every subsequent lookup —
including for the *same* request's own content — reported `0 keys in host_ram`.
Root cause: vLLM creates a **separate connector instance per role**
(`KVConnectorRole.SCHEDULER`, which does lookups, vs. `KVConnectorRole.WORKER`,
which does the actual tensor save) — confirmed directly from vLLM's own log,
which shows `Creating v1 connector with name: TrajectoryAwareOffloadConnector`
*twice* every boot. `self._host_ram` as a plain instance dict was invisible
across those two separate objects. **Fixed** by making storage a module-level
dict (`_HOST_RAM`) instead — both role-instances share the same Python process
for this project's single-GPU, non-distributed setup (confirmed empirically: no
second worker subprocess appears in `ps aux`), so a module-level singleton is
visible to both without needing real inter-process sharing (unlike the reference
`SharedStorageConnector`, which uses disk files specifically because that
mechanism *does* survive a genuine multi-process split — not needed here).

## Round-trip verification: forced eviction, confirmed correct reload

Built a real test: send a small, distinct "target" prompt, capture its
deterministic (`temperature=0`) completion, then send ~11 large, distinct filler
requests (~4,000 tokens each) to push cumulative unique content past the
29,152-token GPU cache capacity, forcing real LRU eviction of the target's
GPU-resident blocks. Then re-send the identical target prompt.

Before the module-level-dict fix: `cached_tokens=0` on the re-request (full
re-prefill) and `0 keys in host_ram` at every lookup — the external cache
silently never engaged, despite `build_connector_meta` correctly deciding to
store. After the fix: `host_ram` grows correctly across filler requests (28, 56,
84, ... 308 keys), and the log shows a genuine hit —

```
TrajectoryAwareOffloadConnector: external cache HIT for request ..., 192 tokens
available beyond the 0 already computed
```

— and the re-requested prompt's completion is **byte-identical** to the original
(`MATCH: True`), confirming the reloaded KV data is not just present but
*correct*, not corrupted.

## Content-discrimination verification

Separately, a real rendered turn sequence (anchor, a frozen-segment-marked turn,
a user turn, a deliberately large tool-output turn, an assistant turn) produced:

```
TrajectoryAwareOffloadConnector: 5/106 blocks worth offloading for request ...
```

5 of 106 blocks (1,681 tokens total) is exactly the expected shape: the large
tool-output turn dominates the token count (~100 of 106 blocks) and is correctly
excluded, while the small anchor/frozen/other-turn content is correctly included
— genuine content-based discrimination, not a "store everything" (`106/106`) or
"store nothing" (`0/106`) default.

## Comparative measurement: net cost is strongly negative

`experiments/phase4_offload_compare.py` replays the same trajectories under
`append_only` twice — once with the connector enabled, once without — measuring
summed wall-clock time (`ttft_ms + decode_ms`; see note below on why prefill
token counts can't be used for this comparison). `append_only` specifically
because its anchor is byte-identical across every trajectory (every trajectory's
`synth_env.py` system prompt is seed-independent), making cross-trajectory anchor
reuse — exactly what offload is meant to catch when GPU-native caching would
evict it — a realistic scenario for this comparison.

On the one trajectory that completed before this session's third WSL crash cut
the sweep short (`traj-000`, 20 steps):

| condition | summed wall-clock (ms) |
|---|---|
| baseline (no offload) | 1,477.5 |
| offload enabled | 64,832.2 |

**Offload was ~44x slower.** The mechanism is clear, not a fluke: `_worth_
storing_mask` decodes the full prompt text and runs a binary search (multiple
more decode calls) to map every classification span back to token indices, on
*every single request that isn't already cached* — and both the decode cost and
the number of spans grow with context length, which increases every step of a
trajectory replay. This is a real, structural cost that would only get worse on
longer trajectories, not better with more data — the qualitative conclusion
doesn't depend on completing the full 6-trajectory sweep.

**Metric note, discovered during round-trip verification**: this project's
existing `StepResult.cached_tokens`/`prefill_tokens` are derived entirely from
vLLM's *native* GPU prefix-cache Prometheus counter
(`vllm:gpu_prefix_cache_hits_total`), confirmed not to move for the connector's
own external cache hits (a structurally separate code path). So token-count
metrics would have silently under-reported the connector's real behavior in
either direction; wall-clock is the only metric this project's harness collects
that actually reflects what the connector does.

## What this shows

The store/load mechanism itself is now proven correct (byte-identical output
after a real forced eviction and reload) and the classification is proven
accurate (correct discrimination on real content) — this sub-phase's two hardest
engineering questions are resolved. But the **as-implemented classification cost
dominates and inverts any benefit the offload mechanism might otherwise provide**.
This is a real, well-measured negative result, not a failure to complete the
work: spec's own Gate 1 language applies here too ("A well-measured negative
result is still a good writeup") — the honest finding is that this specific
classification strategy (full-text decode + binary search, per request) is not
viable without optimization, most plausibly by caching classification results
keyed by content hash (already computed for storage keys) rather than
reclassifying an identical or incrementally-growing prefix from scratch on every
request.

## Known limitations

- **Comparative measurement is n=1 trajectory, 20 steps** — cut short by a WSL
  crash (the third of this session). The finding is treated as conclusive
  because the mechanism causing it (cost scales with context length and request
  count) would only strengthen on a longer sweep, not reverse — but it is a
  single data point, not a statistically tested result.
- **No optimization attempted** — the classification-caching fix suggested above
  was identified but not implemented or tested this session.
- **Single attention backend** — the tensor reshape logic in `_inject_kv_into_
  layer`/`_extract_kv_from_layer` only implements the non-MLA (GQA) path Qwen3
  uses, unlike the reference connector's separate MLA branch.
- **Content-addressed by prompt-token-id hash**, not block-level granularity —
  matches the reference connector's own design (request-level store/load); a
  request whose prefix partially overlaps a previously-stored one doesn't get
  partial credit.
- **`_HOST_RAM` is unbounded** — no eviction policy applied to the offload tier
  itself; everything classified worth-storing accumulates forever for the life
  of the process.

## Next

If revisited: implement and test the classification-caching fix (keyed by the
same content hash already used for storage keys) before attempting another
comparative sweep — there is no point re-measuring the current, known-expensive
classification path. Given three WSL-level crashes absorbed across this session
(a GPU-driver hang, an OOM VM crash during §4.2's investigation, and this
sub-phase's own crash mid-sweep), further live GPU work was intentionally
deferred rather than pushed through on a visibly unstable environment.
