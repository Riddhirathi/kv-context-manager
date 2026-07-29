"""Thin wrapper around vLLM's LLM (V1 engine) exposing per-request cache stats.

vLLM does not run on native Windows (spec §9) — the `vllm` import below is
deferred so this module stays importable for type-checking and for the parts
of the test suite that don't need a live engine. `ModelConfig` itself has no
vLLM dependency and is fully testable without a GPU.

NOTE for whoever runs this first inside WSL2 (Task: set up WSL2 environment):
the exact attribute names for per-request cache/timing stats
(`RequestOutput.num_cached_tokens`, `RequestOutput.metrics.*`) are version-
dependent in vLLM's V1 engine. Verify them against the pinned version in
pyproject.toml on first real run and correct `_extract_step_result` if they've
moved — don't trust this file's field names over the installed library's.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    dtype: str
    enable_prefix_caching: bool
    block_size: int
    gpu_memory_utilization: float
    max_model_len: int
    seed: int

    @classmethod
    def from_yaml(cls, path: Path, *, use_fallback: bool = False) -> ModelConfig:
        data = yaml.safe_load(path.read_text())
        model_key = "fallback" if use_fallback else "primary"
        return cls(
            model_name=data["model"][model_key],
            dtype=data["model"]["dtype"],
            enable_prefix_caching=data["engine"]["enable_prefix_caching"],
            block_size=data["engine"]["block_size"],
            gpu_memory_utilization=data["engine"]["gpu_memory_utilization"],
            max_model_len=data["engine"]["max_model_len"],
            seed=data["engine"]["seed"],
        )


@dataclass(frozen=True)
class StepResult:
    """Per-request stats consumed by metrics.collector.StepRecord."""

    prompt_tokens: int
    cached_tokens: int
    prefill_tokens: int
    output_tokens: int
    ttft_ms: float
    decode_ms: float


class VLLMEngine:
    """Wraps vllm.LLM, forcing prefix caching on and surfacing cache-hit stats."""

    def __init__(self, config: ModelConfig) -> None:
        from vllm import LLM  # deferred: vLLM is Linux/WSL2-only

        self._config = config
        self._llm: Any = LLM(
            model=config.model_name,
            dtype=config.dtype,
            enable_prefix_caching=config.enable_prefix_caching,
            block_size=config.block_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            seed=config.seed,
        )

    def kv_cache_capacity_tokens(self) -> int:
        """Reads the KV cache capacity vLLM actually allocated, in tokens.

        Spec §0.1: "Log the resulting KV cache capacity in tokens; this number
        goes in the README." Verify this accessor against the installed V1
        engine's stats API — see module docstring.
        """
        cache_config = self._llm.llm_engine.cache_config
        return int(cache_config.num_gpu_blocks * self._config.block_size)

    def generate_step(self, prompt_token_ids: list[int], *, max_tokens: int) -> StepResult:
        from vllm import SamplingParams

        params = SamplingParams(max_tokens=max_tokens, seed=self._config.seed)
        outputs = self._llm.generate(
            prompt_token_ids=[prompt_token_ids], sampling_params=params, use_tqdm=False
        )
        output = outputs[0]
        metrics = output.metrics
        cached_tokens = int(getattr(output, "num_cached_tokens", 0))
        prompt_tokens = len(prompt_token_ids)
        ttft_s = metrics.first_token_time - metrics.arrival_time
        decode_s = metrics.finished_time - metrics.first_token_time
        return StepResult(
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            prefill_tokens=prompt_tokens - cached_tokens,
            output_tokens=len(output.outputs[0].token_ids),
            ttft_ms=ttft_s * 1000,
            decode_ms=decode_s * 1000,
        )
