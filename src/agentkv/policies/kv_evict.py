"""KV-space eviction baseline (AGENTKV_SPEC.md §4.1): "StreamingLLM-style: retain
the first few 'attention sink' tokens plus a sliding window; discard the middle
*in KV space*. Text is never rewritten, so there is zero re-prefill. Cost is
near-free; the question is what it costs in quality."

**Measurement caveat, decided deliberately, not discovered as a bug**: "zero
re-prefill" is a claim about a *live, persistent* decode session, where old KV
entries are freed from GPU memory mid-session and attention skips them via
position-ID remapping. This project's entire harness (Phases 0-3) instead sends
each agent step as an independent HTTP completions request to vLLM, with reuse
coming from vLLM's hash-based prefix cache across those requests — there is no
"keep this session's KV cache alive and evict from it" primitive at that layer
(vLLM's real extension point for KV work, `KVConnectorBase_V1`, is a per-request
save/load interface, not that). Sending a shorter *text* prompt each step (sink +
sliding window, literally omitting the evicted middle) does not reproduce
StreamingLLM's zero-re-prefill property under hash-based prefix caching: once the
window is full, its front boundary shifts every step, which invalidates most of
the cached window on every single step — plausibly *worse* than naive/append_only's
occasional compaction events, not better. This is measured through the exact same
harness as every other policy anyway (see `experiments/phase4_kv_evict.py`), and
whatever the real numbers show is reported as-is, consistent with how this
project has always treated a well-measured negative result (spec's own Gate 1
language: "A well-measured negative result is still a good writeup").

What *is* implemented faithfully to spec, independent of the re-prefill question:
this policy's "sink" is this project's existing `anchor` convention (system
prompt + tool schemas + task — already always kept, by every policy, at position
0), and its "sliding window" is the newest `window_turns` raw turns, kept
verbatim. Everything older than the window is **deleted with no replacement
text** — the one substantive difference from `naive.py`/`append_only.py`, both of
which synthesize a summary for retired turns. No LLM summarizer call is needed
here at all, which is itself a real, measurable efficiency property (spec's "cost
is near-free" — true at minimum for the *compaction step itself*, whatever the
re-prefill numbers turn out to say).
"""
from __future__ import annotations

from agentkv.bench.replay import Turn
from agentkv.context.layout import Tokenizer, render_turns_to_token_ids
from agentkv.policies.base import CompactionPolicy


class KVEvictPolicy(CompactionPolicy):
    name = "kv_evict"

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        threshold_tokens: int,
        window_turns: int,
    ) -> None:
        if window_turns < 1:
            raise ValueError(f"window_turns must be >= 1, got {window_turns}")
        self._tokenizer = tokenizer
        self._threshold_tokens = threshold_tokens
        self._window_turns = window_turns

    def maybe_compact(self, turns: list[Turn], step_idx: int) -> tuple[list[Turn], bool]:
        prompt_len = len(render_turns_to_token_ids(turns, self._tokenizer))
        if prompt_len < self._threshold_tokens or len(turns) < 2:
            return turns, False

        # turns[0] is the anchor (spec §2.4 convention, shared by every policy
        # in this project) — doubling here as the "attention sink."
        anchor, rest = turns[:1], turns[1:]
        window = rest[-self._window_turns :]
        if len(window) == len(rest):
            # Threshold crossed on token count, but nothing yet extends past
            # the window in turn count — nothing to evict this step.
            return turns, False
        return anchor + window, True
