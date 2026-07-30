"""Synthetic long-horizon tool environment for recording Phase 0 trajectories.

Not the verifiable ledger task from spec §3.3 — that's a later phase. This
exists purely to give a real model (via free API tier) something long and
tool-call-heavy to do, with the mixed small/large tool-output distribution
spec §0.4 asks for ("mixed tool-output sizes, including some very large ones
— log dumps, file contents — these are what compaction actually targets").
Fully deterministic given a seed so recorded trajectories are reproducible.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

_SERVICES = ["checkout", "auth", "billing", "search", "notifications", "inventory"]
_ERROR_KINDS = [
    "ConnectionTimeoutError",
    "NullPointerException",
    "DeadlockDetected",
    "OutOfMemoryError",
    "RateLimitExceeded",
    "SerializationError",
]
_LOG_LEVELS = ["INFO", "WARN", "ERROR", "DEBUG"]


@dataclass(frozen=True)
class ToolCallResult:
    output: str
    is_error: bool = False


class SyntheticIncidentEnv:
    """A fake production-incident filesystem/log environment.

    Tools: list_files, read_file, search_logs, grep, get_recent_errors.
    Everything is generated deterministically from `seed` the first time it's
    touched, then cached, so repeated calls with the same args are stable
    within one environment instance (mirrors how a real filesystem behaves).
    """

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._incident_service = self._rng.choice(_SERVICES)
        self._root_cause = self._rng.choice(_ERROR_KINDS)
        self._files: dict[str, str] = {}
        self._logs: dict[str, str] = {}
        self._file_list = self._make_file_list()

    def _make_file_list(self) -> list[str]:
        return [
            f"src/{self._incident_service}/handler.py",
            f"src/{self._incident_service}/client.py",
            f"src/{self._incident_service}/config.yaml",
            f"logs/{self._incident_service}-2026-07-29.log",
            f"logs/{self._incident_service}-2026-07-28.log",
            "src/common/retry.py",
            "src/common/db_pool.py",
            "infra/deploy.yaml",
        ]

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "list_files",
                "description": "List files in the incident workspace.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "read_file",
                "description": "Read the full contents of a file by path.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "search_logs",
                "description": "Search log files for a substring, returns matching lines.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "get_recent_errors",
                "description": "Dump the most recent ERROR-level log lines across all services.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "submit_root_cause",
                "description": "Submit the final root-cause diagnosis to close the incident.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string"},
                        "error_kind": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                    "required": ["service", "error_kind", "summary"],
                },
            },
        ]

    def call(self, name: str, args: dict[str, Any]) -> ToolCallResult:
        if name == "list_files":
            return ToolCallResult(output="\n".join(self._file_list))
        if name == "read_file":
            return self._read_file(str(args.get("path", "")))
        if name == "search_logs":
            return ToolCallResult(output=self._search_logs(str(args.get("query", ""))))
        if name == "get_recent_errors":
            return ToolCallResult(output=self._recent_errors())
        if name == "submit_root_cause":
            return ToolCallResult(output="Root cause recorded. Incident closed.")
        return ToolCallResult(output=f"Unknown tool: {name}", is_error=True)

    def _read_file(self, path: str) -> ToolCallResult:
        if path not in self._file_list:
            return ToolCallResult(output=f"No such file: {path}", is_error=True)
        if path not in self._files:
            if path.startswith("logs/"):
                self._files[path] = self._gen_log_dump(path, n_lines=self._rng.randint(80, 400))
            else:
                self._files[path] = self._gen_source_file(path)
        return ToolCallResult(output=self._files[path])

    def _gen_source_file(self, path: str) -> str:
        n_lines = self._rng.randint(15, 60)
        lines = [f"# {path}", f"# service: {self._incident_service}", ""]
        for i in range(n_lines):
            lines.append(f"def handler_{i}(request):")
            lines.append(f"    result = call_downstream_{i % 5}(request)")
            lines.append("    return result")
            lines.append("")
        return "\n".join(lines)

    def _gen_log_dump(self, path: str, n_lines: int) -> str:
        if path not in self._logs:
            lines = []
            error_at = self._rng.randint(n_lines // 3, n_lines - 5)
            for i in range(n_lines):
                level = "ERROR" if i == error_at else self._rng.choice(_LOG_LEVELS)
                if level == "ERROR":
                    lines.append(
                        f"2026-07-29T12:{i % 60:02d}:00Z {level} {self._incident_service}: "
                        f"{self._root_cause}: downstream call failed after 3 retries"
                    )
                else:
                    lines.append(
                        f"2026-07-29T12:{i % 60:02d}:00Z {level} {self._incident_service}: "
                        f"request_id={self._rng.randint(10000, 99999)} handled in "
                        f"{self._rng.randint(5, 400)}ms"
                    )
            self._logs[path] = "\n".join(lines)
        return self._logs[path]

    def _search_logs(self, query: str) -> str:
        matches: list[str] = []
        for path in self._file_list:
            if not path.startswith("logs/"):
                continue
            content = self._read_file(path).output
            matches.extend(line for line in content.splitlines() if query.lower() in line.lower())
        if not matches:
            return f"No log lines matched '{query}'."
        return "\n".join(matches[:200])

    def _recent_errors(self) -> str:
        return self._search_logs("ERROR")

    def system_prompt(self) -> str:
        return (
            "You are an autonomous on-call engineer investigating a production incident. "
            "Use the available tools to explore logs and source files, narrow down the root "
            "cause, and call submit_root_cause once you are confident. Investigate thoroughly "
            "across many tool calls before concluding — do not guess prematurely."
        )
