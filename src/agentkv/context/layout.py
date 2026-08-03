"""Renders a turn sequence to token ids (AGENTKV_SPEC.md repo layout: context/layout.py).

Phase 0/1 have no `ContextState` (anchor/frozen/live) yet — that lands in
Phase 2 (spec §2.1). Until then this is a minimal, deterministic
flatten-to-text-then-tokenize used by both the Gate 0 replay check and every
compaction policy, so all of them measure cache behavior against the exact
same rendering. Token ids (not text) are sent straight to vLLM's completions
endpoint, so any tokenizer quirk at a turn boundary is captured by the ids
themselves rather than hidden behind a re-tokenization step on the server.
"""
from __future__ import annotations

import json
from typing import Protocol

from agentkv.bench.replay import Turn


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


def render_turns_to_token_ids(turns: list[Turn], tokenizer: Tokenizer) -> list[int]:
    parts: list[str] = []
    for t in turns:
        parts.append(f"<{t.role}>")
        if t.content:
            parts.append(t.content)
        if t.tool_calls:
            parts.append(json.dumps(t.tool_calls))
        if t.tool_results:
            parts.append(json.dumps(t.tool_results))
    text = "\n".join(parts)
    return tokenizer.encode(text)


def render_turns_to_text(turns: list[Turn], *, header: str | None = None) -> str:
    """Text-only rendering, for feeding a summarization prompt (never sent as the
    measured trajectory prompt itself — only `render_turns_to_token_ids` is)."""
    parts: list[str] = [header] if header else []
    for t in turns:
        parts.append(f"<{t.role}>")
        if t.content:
            parts.append(t.content)
        if t.tool_calls:
            parts.append(json.dumps(t.tool_calls))
        if t.tool_results:
            parts.append(json.dumps(t.tool_results))
    return "\n".join(p for p in parts if p is not None)
