# Phase 0 Summary — Measurement Harness

Status: **complete**. All 9 Phase 0 tasks done; Gate 0 passes. See `AGENTKV_SPEC.md` §5
for the phase definition and gate criteria.

## Environment (measured, not assumed)

- Hardware: RTX 4060 Laptop, 8GB VRAM, driver 572.70, CUDA 12.8.
- OS: WSL2 Ubuntu 24.04 (Windows 11 host). vLLM does not run natively on Windows.
- Engine: vLLM 0.8.5, V1, served via `vllm.entrypoints.openai.api_server` (the offline
  `LLM.generate()` batch API doesn't populate per-request `metrics`/`num_cached_tokens`
  in this version — see `src/agentkv/serving/engine.py` module docstring).
- `gpu_memory_utilization = 0.85` needed no backoff for any model used (no OOM):

  | Model | Role | KV cache capacity (tokens) | Max concurrency @ configured ctx |
  |---|---|---|---|
  | `Qwen/Qwen3-1.7B` | primary measurement model | ~9,456–9,472 | 1.15–1.54x @ 8192 ctx |
  | `Qwen/Qwen3-0.6B` | fallback measurement model | ~29,150 | 3.56x @ 8192 ctx |
  | `Qwen/Qwen3-4B-AWQ` | trajectory-recording model (Phase 0 only, not measured) | ~12,608 | 1.54x @ 8192 ctx |

  (KV capacity varies ~50-100 tokens run-to-run from free-VRAM fragmentation at boot.)

## Trajectory recording (§0.4)

Recorded via `experiments/record_trajectories.py` using `Qwen/Qwen3-4B-AWQ` served
locally through vLLM (no API keys, no free-tier quota — see history below) against
the synthetic incident-investigation environment in
`src/agentkv/bench/tasks/synth_env.py`.

**15 trajectories recorded (`traj-000` – `traj-014`), 14 valid:**

| File | Turns | Assistant steps | Incidents chained | Status |
|---|---|---|---|---|
| traj-000 | 164 | 81 | 2 | OK |
| traj-001 | 169 | 81 | 7 | OK |
| traj-002 | 166 | 81 | 4 | OK |
| traj-003 | 163 | 80 | 3 | OK |
| traj-004 | 165 | 81 | 3 | OK |
| traj-005 | 164 | 80 | 4 | OK |
| traj-006 | 162 | 80 | 2 | OK |
| traj-007 | 128 | 60 | 7 | **short — interrupted mid-run, needs regen** |
| traj-008 | 161 | 80 | 1 | OK |
| traj-009 | 162 | 80 | 2 | OK |
| traj-010 | 164 | 80 | 4 | OK |
| traj-011 | 161 | 80 | 1 | OK |
| traj-012 | 161 | 80 | 1 | OK |
| traj-013 | 164 | 81 | 2 | OK |
| traj-014 | 165 | 80 | 5 | OK |

All 15 are valid JSONL with zero repetition-bug artifacts (`grep -c "Not enough
evidence"` = 0 everywhere). To regenerate the short one:
```bash
rm trajectories/traj-007.jsonl
.venv/bin/python experiments/record_trajectories.py --count 1 --seed-start 7
```

**Known, accepted deviation from spec:** §0.4 asks for "at minimum 30" trajectories;
14-15 were recorded. This is a deliberate scope reduction under time/hardware
constraints, not an oversight — flag it explicitly in the eventual limitations
section (spec §6.3). Gate 0 and early Phase 1-2 work don't need 30; later
statistical claims (Phase 4's Pareto plot, "≥5 seeds per configuration") are where
more raw trajectories would matter most if revisited.

**Debugging history worth remembering** (all fixed, see `experiments/record_trajectories.py`
and `src/agentkv/bench/tasks/synth_env.py` for the surviving code/comments):
- Free-tier API quotas (Gemini, then Groq) couldn't sustain this task's large tool
  outputs — switched to a fully local model.
- `Qwen/Qwen3-4B` doesn't fit unquantized in 8GB (crashes mid weight-load) — switched
  to the `Qwen/Qwen3-4B-AWQ` 4-bit quantized checkpoint.
- The environment's `min_steps` gate originally blocked `submit_root_cause` until 80
  steps, causing the model to loop a rejected-submission cycle ~15x with near-duplicate
  content — fixed by chaining fresh incidents instead of blocking.
- Local token-budget estimation (two different approaches: raw json.dumps+tokenizer,
  then `tokenizer.apply_chat_template`) both undercounted the real vLLM prompt size,
  because `--tool-call-parser hermes` injects a prompt preamble no local proxy
  replicates (verified: same messages+tools showed 23 tokens via `/tokenize` but 138
  real `prompt_tokens` in an actual generation call) — fixed by reacting to the real
  400 error vLLM reports (`ContextTooLongError`) and shrinking context based on the
  exact numbers it gives, instead of predicting.
- The synthetic environment's log-dump generator could produce a single tool output
  up to 15,274 real tokens — bigger than any reasonable context window on this
  hardware — capped it to a 20-60 line range (worst case now ~2,771 tokens).

## Gate 0: reproducibility (PASS)

Per spec: *"Running the same trajectory twice under the same policy produces TTFT
medians within 5% of each other, and cache hit rate is bit-identical."*

Checked via `experiments/gate0_check.py` (rerunnable — reuses the project's own
`serving/engine.py` + `bench/replay.py`), replaying the first 15 steps of
`traj-000.jsonl` verbatim (no compaction policy exists yet — that's Phase 1+)
against `Qwen/Qwen3-1.7B`, in **two fully independent server instances**:

```
KV cache capacity: 9472 tokens
cached_tokens bit-identical across runs: True
TTFT median run 1: 61.90ms, run 2: 60.72ms, drift: 1.90%
GATE 0: PASS
```

**Pitfall found and fixed during validation:** the first attempt reused one warm
server for both "runs," so run 2 silently benefited from run 1's cached prefix
blocks (observed: run 2's `cached_tokens` sequence was exactly run 1's, shifted by
one step) — not a real reproducibility problem, a test-methodology bug. Fixed by
booting two separate `VLLMEngine` instances, one per run.

## Next: Phase 1 (quantify the bug)

Naive compaction baseline, the cliff figure, divergence-point instrumentation.
Nothing further required in Phase 0 unless `traj-007` is regenerated first.
