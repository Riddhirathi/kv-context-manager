"""Wraps vLLM's OpenAI-compatible server, exposing per-request cache stats.

vLLM does not run on native Windows (spec §9) — the `vllm`/subprocess-spawn
codepaths below only execute when `VLLMEngine` is actually instantiated, so
this module stays importable for type-checking on any platform. `ModelConfig`
itself has no vLLM dependency and is fully testable without a GPU.

Why the server, not the offline `vllm.LLM` batch API: verified on the real
4060 against vLLM 0.8.5 that `LLM.generate()`'s `RequestOutput.metrics` and
`RequestOutput.num_cached_tokens` are simply `None` in this version — V1's
offline path doesn't populate per-request timing/cache stats. The OpenAI
server does, via two independent, verified channels:

1. Streaming `/v1/completions` gives real TTFT (wall-clock to first SSE
   chunk) and decode time (first chunk to stream end), plus `usage` in the
   final chunk for prompt/completion token counts.
2. The Prometheus `/metrics` endpoint exposes a cumulative
   `vllm:gpu_prefix_cache_hits_total` counter, in *blocks*. Verified
   empirically: it increases by exactly N when a request reuses N full
   blocks of a previously-cached prefix, and by 0 on a cold request. Per-
   request cached_tokens = block-counter delta × block_size.

Neither of these is documented as a stable public contract — both were
confirmed by directly booting the server and diffing real responses, not
assumed from vLLM's docs. If a future vLLM version renames the metric or
drops delta-style counters, `generate_step`/`_prefix_cache_hit_blocks` will
raise clearly rather than silently return wrong numbers.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

_KV_CACHE_LOG_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
_PREFIX_CACHE_HITS_METRIC = "vllm:gpu_prefix_cache_hits_total"


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
            # Per-model ceiling, not one shared value — the two models measured
            # very different KV cache headroom on this 8GB card (see
            # configs/model.yaml's `measured` section), so each gets its own.
            max_model_len=data["engine"]["max_model_len"][model_key],
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
    """Boots `vllm serve` as a subprocess and drives it over HTTP.

    Use as a context manager so the server subprocess is always cleaned up:

        with VLLMEngine(config) as engine:
            result = engine.generate_step(token_ids, max_tokens=32)
    """

    def __init__(
        self, config: ModelConfig, *, port: int = 8901, startup_timeout_s: float = 300.0
    ) -> None:
        self._config = config
        self._base_url = f"http://127.0.0.1:{port}"
        self._client = httpx.Client(timeout=120.0)
        self._kv_cache_capacity_tokens: int | None = None
        self._log_lines: list[str] = []

        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            config.model_name,
            "--dtype",
            config.dtype,
            "--block-size",
            str(config.block_size),
            "--gpu-memory-utilization",
            str(config.gpu_memory_utilization),
            "--max-model-len",
            str(config.max_model_len),
            "--seed",
            str(config.seed),
            "--port",
            str(port),
        ]
        if config.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")

        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        self._log_thread = threading.Thread(target=self._drain_log, daemon=True)
        self._log_thread.start()
        self._wait_until_ready(startup_timeout_s)

    def _drain_log(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            self._log_lines.append(line)
            print(line, end="")  # don't silently swallow vLLM's server log
            match = _KV_CACHE_LOG_RE.search(line)
            if match is not None:
                self._kv_cache_capacity_tokens = int(match.group(1).replace(",", ""))

    def _wait_until_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    "vLLM server exited during startup:\n" + "".join(self._log_lines)
                )
            try:
                if self._client.get(f"{self._base_url}/health", timeout=2.0).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(2.0)
        else:
            raise RuntimeError(f"vLLM server did not become healthy within {timeout_s}s.")

        if self._kv_cache_capacity_tokens is None:
            # Give the log-draining thread a moment to catch up with the health check.
            time.sleep(1.0)
        if self._kv_cache_capacity_tokens is None:
            raise RuntimeError(
                "vLLM server is healthy but never logged 'GPU KV cache size: N tokens'; "
                "its log format may have changed — update _KV_CACHE_LOG_RE in this module."
            )

    def kv_cache_capacity_tokens(self) -> int:
        """The KV cache capacity vLLM actually allocated, in tokens.

        Spec §0.1: "Log the resulting KV cache capacity in tokens; this number
        goes in the README."
        """
        assert self._kv_cache_capacity_tokens is not None
        return self._kv_cache_capacity_tokens

    def _prefix_cache_hit_blocks(self) -> float:
        text = self._client.get(f"{self._base_url}/metrics", timeout=10.0).text
        for line in text.splitlines():
            if line.startswith(_PREFIX_CACHE_HITS_METRIC):
                return float(line.rsplit(" ", 1)[1])
        raise RuntimeError(f"{_PREFIX_CACHE_HITS_METRIC} not found in /metrics output.")

    def _stable_prefix_cache_hit_blocks(
        self, *, settle_timeout_s: float = 2.0, poll_interval_s: float = 0.05
    ) -> float:
        """Polls `/metrics` until the counter stops changing between reads.

        Verified empirically: after a large chunked-prefill request (a
        compaction event's re-prefill, which vLLM splits across multiple
        scheduler steps per `--max-num-batched-tokens`), the counter can still
        be ticking up for a moment after the HTTP response has already
        finished streaming. Reading it exactly once right at response
        completion silently hands part of that still-arriving increment to
        whichever request's "before" snapshot happens to poll next,
        inflating that *next* request's apparent cache-hit delta beyond what
        it actually reused.
        """
        value = self._prefix_cache_hit_blocks()
        deadline = time.monotonic() + settle_timeout_s
        while time.monotonic() < deadline:
            time.sleep(poll_interval_s)
            next_value = self._prefix_cache_hit_blocks()
            if next_value == value:
                return value
            value = next_value
        return value

    def generate_step(self, prompt_token_ids: list[int], *, max_tokens: int) -> StepResult:
        hits_before = self._stable_prefix_cache_hit_blocks()
        t0 = time.monotonic()
        first_chunk_t: float | None = None
        usage: dict[str, int] | None = None
        raw_lines: list[str] = []

        with self._client.stream(
            "POST",
            f"{self._base_url}/v1/completions",
            json={
                "model": self._config.model_name,
                "prompt": prompt_token_ids,
                "max_tokens": max_tokens,
                "seed": self._config.seed,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as response:
            if response.status_code != 200:
                # A non-200 response (e.g. 400 because prompt_token_ids exceeds
                # max_model_len) still lands here rather than raising, since
                # streaming responses don't raise_for_status automatically —
                # read the body explicitly so the failure is diagnosable
                # instead of silently falling through to "no usage chunks".
                body = response.read().decode(errors="replace")
                raise RuntimeError(
                    f"vLLM /v1/completions returned HTTP {response.status_code} for a "
                    f"{len(prompt_token_ids)}-token prompt (max_model_len="
                    f"{self._config.max_model_len}): {body}"
                )
            for line in response.iter_lines():
                raw_lines.append(line)
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: ") :]
                if payload.strip() == "[DONE]":
                    break
                if first_chunk_t is None:
                    first_chunk_t = time.monotonic()
                chunk_usage = json.loads(payload).get("usage")
                if chunk_usage:
                    usage = chunk_usage
        t_end = time.monotonic()

        if usage is None or first_chunk_t is None:
            preview = " | ".join(raw_lines[:5]) if raw_lines else "(empty response body)"
            raise RuntimeError(
                f"vLLM completions stream returned no usage/content chunks for a "
                f"{len(prompt_token_ids)}-token prompt (max_model_len="
                f"{self._config.max_model_len}). Raw response: {preview}"
            )

        hits_after = self._stable_prefix_cache_hit_blocks()
        cached_tokens = int((hits_after - hits_before) * self._config.block_size)

        return StepResult(
            prompt_tokens=usage["prompt_tokens"],
            cached_tokens=cached_tokens,
            prefill_tokens=usage["prompt_tokens"] - cached_tokens,
            output_tokens=usage["completion_tokens"],
            ttft_ms=(first_chunk_t - t0) * 1000,
            decode_ms=(t_end - first_chunk_t) * 1000,
        )

    def complete_text(
        self,
        prompt_token_ids: list[int],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        guided_json: dict[str, object] | None = None,
        min_tokens: int = 0,
    ) -> str:
        """Plain-text completion, not routed through the cache-hit-delta bookkeeping
        in `generate_step` — used for summarization calls (policies/naive.py) that
        are real generation work but not one of the trajectory steps being measured.
        `temperature=0.0` keeps summaries deterministic given the same input text,
        which compaction-event reproducibility depends on.

        `guided_json`, when given, is forwarded as vLLM's grammar-constrained
        decoding schema (no server restart or extra flags needed — vLLM's
        OpenAI-compatible server accepts this per-request on the plain
        `/v1/completions` endpoint). `bench/agreement.py` uses this: verified
        empirically that this measurement model, asked in free text to
        produce a `{"name": ..., "args": {...}}` action, would either emit an
        immediate end-of-sequence token (nothing generated) or write prose
        reasoning that never actually closes a valid JSON call — grammar
        constraints force a well-formed object every time instead.

        `min_tokens`, when given, blocks end-of-sequence for that many tokens.
        Verified empirically (Phase 3.1 debugging): a primed prompt can lead
        this model to emit an immediate EOS (`finish_reason="stop"`, 1
        completion token, empty text) rather than answering at all —
        `bench/tasks/probes.py`'s free-text QA elicitation isn't wrapped in
        `guided_json` (an open-ended answer isn't a fixed schema), so it needs
        this instead to guarantee a non-empty attempt.
        """
        payload: dict[str, object] = {
            "model": self._config.model_name,
            "prompt": prompt_token_ids,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": self._config.seed,
        }
        if guided_json is not None:
            payload["guided_json"] = guided_json
        if min_tokens:
            payload["min_tokens"] = min_tokens
        response = self._client.post(f"{self._base_url}/v1/completions", json=payload)
        response.raise_for_status()
        text: str = response.json()["choices"][0]["text"]

        # Force the global prefix-cache-hits counter to settle before handing
        # control back to the caller. Root cause of a real Phase 2 bug: this
        # method (unlike generate_step) never touched the counter at all, so a
        # large summarization prefill's own cache-hit accounting could still
        # be trickling in (same lagging-counter behavior documented on
        # `_stable_prefix_cache_hit_blocks`) when the *next* call reads
        # `hits_before` for an unrelated trajectory step — silently crediting
        # that step with part of the summarizer's own reuse. Reproduced: naive's
        # first compaction event on a 69-turn retirement (a large summarization
        # prefill) measured 592 more "reused" tokens on the next trajectory
        # step than the block-math theory allowed for (a 33-block mismatch,
        # far past the documented +/-1 tolerance) - a timing race, not
        # deterministic, so Phase 1's ~280 firing events not hitting it
        # doesn't mean it couldn't happen there too; a larger retirement just
        # widens the window to lose the race.
        self._stable_prefix_cache_hit_blocks()
        return text

    def shutdown(self) -> None:
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._client.close()

    def __enter__(self) -> VLLMEngine:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()
