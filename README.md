# AgentKV

Cache-aware context compaction for long-horizon LLM agents.

**Status:** Phase 0 (measurement harness) in progress. No figures yet — this section
will be replaced per the structure in `AGENTKV_SPEC.md` §6.3 (cliff figure, one-line
result, Pareto plot, reproduction steps, limitations) once Phase 1+ lands.

See `AGENTKV_SPEC.md` for the full project spec and phased plan.

## Environment

- Hardware target: single NVIDIA RTX 4060 Laptop (8 GB VRAM, Ada / SM 8.9).
- OS: Linux via WSL2 Ubuntu (Windows host). vLLM does not run natively on Windows.
- Model: `Qwen/Qwen3-1.7B` primary, `Qwen/Qwen3-0.6B` fallback if VRAM-tight.
- `gpu_memory_utilization`: TBD empirically (see `configs/model.yaml`) — will be
  recorded here along with the resulting KV cache capacity in tokens once measured.
- Pinned versions: see `pyproject.toml` (vLLM, torch, transformers). Exact versions
  will be locked via `pip freeze` after first successful `make setup` and reported here.

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
