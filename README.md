# AgentKV

Cache-aware context compaction for long-horizon LLM agents.

**Status:** Phase 0 (measurement harness) in progress. No figures yet — this section
will be replaced per the structure in `AGENTKV_SPEC.md` §6.3 (cliff figure, one-line
result, Pareto plot, reproduction steps, limitations) once Phase 1+ lands.

See `AGENTKV_SPEC.md` for the full project spec and phased plan.

## Environment

- Hardware: NVIDIA RTX 4060 Laptop, 8GB VRAM, driver 572.70, CUDA 12.8.
- OS: WSL2 Ubuntu 24.04 (Windows 11 host). vLLM does not run natively on Windows;
  GPU passthrough into WSL2 confirmed working.
- Engine: vLLM 0.8.5, V1 engine, served via `vllm.entrypoints.openai.api_server`
  (not the offline `LLM.generate()` batch API — verified that it doesn't populate
  per-request `metrics`/`num_cached_tokens` in this version; see
  `src/agentkv/serving/engine.py`'s module docstring for the full rationale).
- `gpu_memory_utilization = 0.85` needed no backoff for either model — confirmed
  no OOM at `max_model_len = 8192` for both:

  | Model | KV cache capacity (tokens) | Max concurrency @ 8192 ctx |
  |---|---|---|
  | `Qwen/Qwen3-1.7B` (primary) | ~9,456 | 1.15x — tight, little headroom |
  | `Qwen/Qwen3-0.6B` (fallback) | ~29,150 | 3.56x |

  (KV capacity varies ~50-100 tokens run-to-run from free-VRAM fragmentation at
  boot; see `configs/model.yaml`'s `measured` section for the source numbers.)
- `max_model_len` is configured per model in `configs/model.yaml` (`primary: 8192`,
  `fallback: 16384`), not one shared value: Phase 2's append_only policy never
  prunes `frozen` (unlike naive, which replaces it every event), so a full
  trajectory can need more total context than naive ever does. The primary
  model's headroom is too tight (1.15x) to raise its ceiling safely, so
  Phase 2 experiments run with `--use-fallback` (`Qwen/Qwen3-0.6B`, 3.56x
  headroom) instead — spec §8's documented mitigation for "8GB is too tight
  for a useful context length," and its own claim is that the *relative*
  naive-vs-append_only effect is model-size-independent.
- Pinned versions: see `pyproject.toml`. Installed and verified: torch 2.6.0+cu124,
  vLLM 0.8.5, Python 3.12.3.

## Setup (inside WSL2 Ubuntu)

```bash
make setup      # creates .venv, installs agentkv + deps
make test        # runs unit tests (no GPU required)
make bench        # Phase 0 smoke benchmark
make reproduce   # regenerate every figure from scratch
make demo        # live side-by-side dashboard
```

## Repository layout

See `AGENTKV_SPEC.md` §4 for the full annotated layout.

## Limitations

To be written honestly and prominently per spec §6.3, once there are results to be
honest about.
