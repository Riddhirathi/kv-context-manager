#!/usr/bin/env python3
"""One-time trajectory recorder (AGENTKV_SPEC.md §0.4).

Runs a strong model (Groq free tier, OpenAI-compatible API) against the
synthetic long-horizon incident-investigation environment in
`agentkv.bench.tasks.synth_env`, and commits the resulting transcripts as
JSONL under trajectories/. This is the ONLY place in the project allowed to
make a network call — the rest of the pipeline (replay, policies, serving)
must never depend on network access at run time (spec §10).

Resumable: re-running skips any trajectory file that already exists, so it's
safe to interrupt (rate limits, laptop sleep) and restart.

Usage (inside the WSL venv):
    python experiments/record_trajectories.py --count 30 --min-steps 80
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentkv.bench.tasks.synth_env import SyntheticIncidentEnv  # noqa: E402

DEFAULT_MODEL = "llama-3.3-70b-versatile"
API_URL = "https://api.groq.com/openai/v1/chat/completions"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def to_openai_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"type": "function", "function": schema} for schema in schemas]


class GroqClient:
    def __init__(self, api_key: str, model: str, min_interval_s: float) -> None:
        self._api_key = api_key
        self._model = model
        self._min_interval_s = min_interval_s
        self._last_call = 0.0
        self._client = httpx.Client(timeout=60.0)

    def generate(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval_s:
            time.sleep(self._min_interval_s - elapsed)

        body = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        backoff = 2.0
        max_attempts = 8
        for attempt in range(max_attempts):
            self._last_call = time.monotonic()
            response = self._client.post(API_URL, headers=headers, json=body)
            if response.status_code == 200:
                return response.json()  # type: ignore[no-any-return]
            retryable = response.status_code in (429, 500, 502, 503) or (
                response.status_code == 400 and "tool_use_failed" in response.text
            )
            if retryable and attempt < max_attempts - 1:
                # tool_use_failed is Groq/Llama occasionally emitting malformed
                # function-call markup — observed to be transient on resend.
                time.sleep(backoff)
                backoff *= 2
                continue
            raise RuntimeError(
                f"Groq API error {response.status_code} on attempt {attempt + 1}: "
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


def record_one_trajectory(
    client: GroqClient, seed: int, min_steps: int, max_steps: int, out_path: Path
) -> None:
    env = SyntheticIncidentEnv(seed=seed)
    tools = to_openai_tools(env.tool_schemas())

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": env.system_prompt()},
        {"role": "user", "content": "Begin your investigation."},
    ]

    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"role": "system", "content": env.system_prompt()}) + "\n")
        f.write(json.dumps({"role": "user", "content": "Begin your investigation."}) + "\n")

        step = 0
        while step < max_steps:
            response = client.generate(messages=messages, tools=tools)
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

            if not calls:
                if step < min_steps:
                    nudge = "Keep investigating using the available tools."
                    messages.append({"role": "user", "content": nudge})
                    f.write(json.dumps({"role": "user", "content": nudge}) + "\n")
                    continue
                break

            tool_results = []
            done = False
            for call in calls:
                if call["name"] == "submit_root_cause" and step < min_steps:
                    result_text = (
                        "Not enough evidence yet — keep investigating logs and files "
                        "before submitting a root cause."
                    )
                else:
                    result_text = env.call(call["name"], call["args"]).output
                    if call["name"] == "submit_root_cause":
                        done = True
                tool_results.append({"name": call["name"], "output": result_text})
                messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": result_text}
                )

            tool_turn = {"role": "tool", "tool_results": tool_results}
            f.write(json.dumps(tool_turn) + "\n")

            if done:
                break

    n_lines = sum(1 for _ in out_path.open(encoding="utf-8"))
    print(f"  seed {seed}: {n_lines} turns -> {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--min-steps", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=130)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--model", default=os.environ.get("GROQ_MODEL", DEFAULT_MODEL))
    parser.add_argument("--rpm", type=float, default=float(os.environ.get("GROQ_RPM", "25")))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "trajectories"))
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one 5-step trajectory to validate the API key/model before the full batch.",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY not set (expected in .env or the environment).")

    client = GroqClient(api_key, args.model, min_interval_s=60.0 / args.rpm)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        print(f"Smoke test against model={args.model!r} ...")
        record_one_trajectory(
            client, seed=999_999, min_steps=3, max_steps=5, out_path=out_dir / "_smoke_test.jsonl"
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


if __name__ == "__main__":
    main()
