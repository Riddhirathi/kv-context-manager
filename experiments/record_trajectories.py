#!/usr/bin/env python3
"""One-time trajectory recorder (AGENTKV_SPEC.md §0.4).

Runs a strong model (Qwen/Qwen3-4B-AWQ, served locally via vLLM on the
project's own GPU — no network calls, no free-tier API keys) against the
synthetic long-horizon incident-investigation environment in
`agentkv.bench.tasks.synth_env`, and commits the resulting transcripts as
JSONL under trajectories/.

Historical note: earlier attempts used Groq's free-tier hosted API (and,
before that, Gemini's). Both hit free-tier quota walls with this task's large
tool outputs. This version launches Qwen3-4B-AWQ locally instead.

Context-length handling is REACTIVE, not predictive — two different local
estimates were tried and both were wrong:
  1. A json.dumps + tokenizer.encode heuristic undercounted the real prompt
     by ~650 tokens (no chat-template role markers accounted for).
  2. tokenizer.apply_chat_template(tools=...) still undercounted by ~550
     tokens, because vLLM's --tool-call-parser hermes injects its own
     tool-list preamble into the prompt that neither apply_chat_template nor
     vLLM's own /tokenize endpoint replicate (empirically verified: the same
     messages+tools tokenized 23 tokens via /tokenize but 138 prompt_tokens
     in the real /v1/chat/completions response — /tokenize silently ignores
     `tools` in this vLLM version).
No local proxy matches the real hermes-formatted prompt, so instead of
guessing, `LocalClient.generate` catches the real 400 ("maximum context
length is N... requested M") and record_one_trajectory drops the oldest
history units and retries, converging on the exact number vLLM reports.

Resumable: re-running skips any trajectory file that already exists, so it's
safe to interrupt (crash, laptop sleep) and restart.

Usage (inside the WSL venv):
    python experiments/record_trajectories.py --count 30 --min-steps 80
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentkv.bench.tasks.synth_env import SyntheticIncidentEnv  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-4B-AWQ"
# 8192 (not 4096) for headroom above the largest single tool output — measured
# empirically up to ~15k tokens before the synth_env.py log-dump size was
# capped; even capped, some margin above a single unit + anchor is worth
# keeping. Confirmed the AWQ model's KV cache pool (~12.6k tokens at this
# gpu_memory_utilization) comfortably covers this.
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_MAX_OUTPUT_TOKENS = 512

_KV_CACHE_LOG_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
_CONTEXT_LIMIT_RE = re.compile(
    r"maximum context length is (\d+) tokens\. However, you requested (\d+) tokens"
)


def to_openai_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"type": "function", "function": schema} for schema in schemas]


def _group_into_units(history: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Groups messages after the anchor into [assistant, its tool/user followups...]
    units, so trimming never separates a tool_call from its tool response."""
    units: list[list[dict[str, Any]]] = []
    for msg in history:
        if msg["role"] == "assistant" or not units:
            units.append([msg])
        else:
            units[-1].append(msg)
    return units


def _select_units(
    anchor: list[dict[str, Any]], units: list[list[dict[str, Any]]], keep_n: int
) -> list[dict[str, Any]]:
    """Keeps the anchor plus the most recent `keep_n` history units. What gets
    SENT to the server is independent of what gets RECORDED to JSONL; the
    full untruncated history is always written to the trajectory file."""
    kept_units = units[-keep_n:] if keep_n > 0 else []
    return anchor + [msg for unit in kept_units for msg in unit]


class ContextTooLongError(RuntimeError):
    """Raised when vLLM rejects a request as exceeding max_model_len. Carries
    the exact requested/limit token counts from the real error message —
    waiting doesn't help, only sending fewer tokens does."""

    def __init__(self, requested: int, limit: int) -> None:
        self.requested = requested
        self.limit = limit
        super().__init__(f"context too long: requested {requested} tokens, limit {limit}")


class LocalVLLMServer:
    """Boots `vllm.entrypoints.openai.api_server` as a subprocess and waits
    for it to become healthy. Same pattern as serving/engine.py's
    VLLMEngine, trimmed down to just what recording needs (no per-request
    cache-stat probing)."""

    def __init__(
        self,
        model: str,
        *,
        port: int = 8901,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization: float = 0.85,
        startup_timeout_s: float = 600.0,
    ) -> None:
        self.base_url = f"http://127.0.0.1:{port}"
        self._client = httpx.Client(timeout=120.0)
        self._log_lines: list[str] = []
        self._kv_cache_capacity_tokens: int | None = None

        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--quantization",
            "awq",
            "--enable-prefix-caching",
            "--block-size",
            "16",
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--max-model-len",
            str(max_model_len),
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--port",
            str(port),
        ]
        print(f"Launching local vLLM server: {' '.join(cmd)}")
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
            print(line, end="")
            match = _KV_CACHE_LOG_RE.search(line)
            if match is not None:
                self._kv_cache_capacity_tokens = int(match.group(1).replace(",", ""))

    def _wait_until_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    "vLLM server exited during startup:\n" + "".join(self._log_lines[-200:])
                )
            try:
                if self._client.get(f"{self.base_url}/health", timeout=2.0).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2.0)
        raise RuntimeError(f"vLLM server did not become healthy within {timeout_s}s.")

    def shutdown(self) -> None:
        self._process.terminate()
        try:
            self._process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._client.close()

    def __enter__(self) -> LocalVLLMServer:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.shutdown()


class LocalClient:
    """Talks to a local OpenAI-compatible vLLM server. No rate limiting since
    there's no per-minute token quota, but context-length errors are real
    (see module docstring) and surfaced as ContextTooLongError."""

    def __init__(self, base_url: str, model: str, max_output_tokens: int) -> None:
        self._base_url = base_url
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._client = httpx.Client(timeout=300.0)

    def generate(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        body = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": self._max_output_tokens,
        }
        backoff = 2.0
        max_attempts = 5
        for attempt in range(max_attempts):
            response = self._client.post(f"{self._base_url}/v1/chat/completions", json=body)
            if response.status_code == 200:
                return response.json()  # type: ignore[no-any-return]
            if response.status_code == 400:
                match = _CONTEXT_LIMIT_RE.search(response.text)
                if match is not None:
                    raise ContextTooLongError(
                        requested=int(match.group(2)), limit=int(match.group(1))
                    )
            retryable = response.status_code in (429, 500, 502, 503)
            if retryable and attempt < max_attempts - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise RuntimeError(
                f"vLLM server error {response.status_code} on attempt {attempt + 1}: "
                f"{response.text[:500]}"
            )
        raise RuntimeError("unreachable")


def extract_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    raw_calls = message.get("tool_calls") or []
    calls = []
    for call in raw_calls:
        fn = call["function"]
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        calls.append({"id": call["id"], "name": fn["name"], "args": args})
    return calls


def _generate_with_shrink(
    client: LocalClient,
    anchor: list[dict[str, Any]],
    history: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    keep_n: int,
) -> tuple[dict[str, Any], int]:
    """Tries `keep_n` most-recent history units, shrinking on a real
    ContextTooLongError until the request fits. Returns the response and the
    keep_n that worked, so the caller can reuse it as next step's starting
    guess (context only grows, so it's a safe upper bound to retry from)."""
    units = _group_into_units(history)
    keep_n = min(keep_n, len(units))
    while True:
        request_messages = _select_units(anchor, units, keep_n)
        try:
            return client.generate(messages=request_messages, tools=tools), keep_n
        except ContextTooLongError as e:
            if keep_n <= 1:
                raise RuntimeError(
                    "Even the anchor plus a single history unit exceeds max_model_len "
                    f"(requested {e.requested}, limit {e.limit}) — a single tool output "
                    "is larger than the context window; raise --max-model-len."
                ) from e
            # Shrink proportionally to the real reported overage rather than by one
            # unit at a time, so this converges in a couple of retries, not dozens.
            keep_n = max(1, int(keep_n / (e.requested / e.limit)) - 1)


def record_one_trajectory(
    client: LocalClient,
    seed: int,
    min_steps: int,
    max_steps: int,
    out_path: Path,
) -> None:
    """Chains independent incidents within one trajectory until min_steps is
    reached, rather than gating submit_root_cause until then. The synthetic
    environment is shallow enough that a competent model resolves one
    incident in ~5-15 turns — gating submission just forced the model to
    loop the same rejected-submission cycle for the remaining ~70 turns
    (observed empirically: 15 identical "not enough evidence" rejections in
    one trajectory, most of the recorded content near-duplicate). Real
    long-horizon agents work through many tasks back-to-back; this mirrors
    that instead of manufacturing repetition."""
    episode = 0
    env = SyntheticIncidentEnv(seed=seed)
    tools = to_openai_tools(env.tool_schemas())

    anchor: list[dict[str, Any]] = [
        {"role": "system", "content": env.system_prompt()},
        {"role": "user", "content": "Begin your investigation."},
    ]
    messages: list[dict[str, Any]] = list(anchor)
    keep_n = 10_000  # effectively "all" to start; shrinks reactively as needed

    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"role": "system", "content": env.system_prompt()}) + "\n")
        f.write(json.dumps({"role": "user", "content": "Begin your investigation."}) + "\n")

        step = 0
        while step < max_steps:
            response, keep_n = _generate_with_shrink(
                client, anchor, messages[2:], tools, keep_n
            )
            message = response["choices"][0]["message"]
            text = message.get("content")
            calls = extract_calls(message)

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": text}
            if calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": json.dumps(c["args"])},
                    }
                    for c in calls
                ]
            messages.append(assistant_msg)

            assistant_turn = {
                "role": "assistant",
                "content": text,
                "tool_calls": [{"name": c["name"], "args": c["args"]} for c in calls] or None,
            }
            f.write(json.dumps(assistant_turn) + "\n")
            step += 1
            keep_n += 1  # the turn just added grows history by one unit

            if not calls:
                if step < min_steps:
                    nudge = "Keep investigating using the available tools."
                    messages.append({"role": "user", "content": nudge})
                    f.write(json.dumps({"role": "user", "content": nudge}) + "\n")
                    continue
                break

            tool_results = []
            incident_resolved = False
            for call in calls:
                result_text = env.call(call["name"], call["args"]).output
                if call["name"] == "submit_root_cause":
                    incident_resolved = True
                tool_results.append({"name": call["name"], "output": result_text})
                messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": result_text}
                )

            tool_turn = {"role": "tool", "tool_results": tool_results}
            f.write(json.dumps(tool_turn) + "\n")

            if incident_resolved:
                if step >= min_steps:
                    break
                episode += 1
                env = SyntheticIncidentEnv(seed=seed * 1000 + episode)
                tools = to_openai_tools(env.tool_schemas())
                nudge = (
                    "Root cause accepted, incident closed. A new incident has just come "
                    "in — begin investigating it using the tools."
                )
                messages.append({"role": "user", "content": nudge})
                f.write(json.dumps({"role": "user", "content": nudge}) + "\n")

    n_lines = sum(1 for _ in out_path.open(encoding="utf-8"))
    print(f"  seed {seed}: {n_lines} turns, {episode + 1} incident(s) -> {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--min-steps", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=130)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=8901)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument(
        "--base-url",
        default=None,
        help="Connect to an already-running vLLM server instead of launching one "
        "(useful for iterating without a reboot each time).",
    )
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "trajectories"))
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one 5-step trajectory to validate the server/model before the full batch.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def run_batch(base_url: str) -> None:
        client = LocalClient(base_url, args.model, args.max_output_tokens)

        if args.smoke_test:
            print(f"Smoke test against model={args.model!r} at {base_url} ...")
            record_one_trajectory(
                client,
                seed=999_999,
                min_steps=3,
                max_steps=5,
                out_path=out_dir / "_smoke_test.jsonl",
            )
            print("Smoke test OK.")
            (out_dir / "_smoke_test.jsonl").unlink(missing_ok=True)
            return

        for seed in range(args.seed_start, args.seed_start + args.count):
            out_path = out_dir / f"traj-{seed:03d}.jsonl"
            if out_path.exists():
                print(f"  seed {seed}: already recorded, skipping.")
                continue
            record_one_trajectory(client, seed, args.min_steps, args.max_steps, out_path)

    if args.base_url:
        run_batch(args.base_url)
    else:
        with LocalVLLMServer(
            args.model, port=args.port, max_model_len=args.max_model_len
        ) as server:
            run_batch(server.base_url)


if __name__ == "__main__":
    main()
