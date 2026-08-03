"""Naive compaction baseline (AGENTKV_SPEC.md §1.1): "the industry-standard
baseline. When context exceeds a threshold (e.g. 60% of window), replace the
oldest 50% of turns with an LLM-generated summary placed at that same
position. This is what everyone ships today."

Deliberately dumb: unlike Phase 2's append-only policy, this does not treat a
previous summary specially — if the current turns already include one from an
earlier compaction, it gets swept up into the next round's "oldest 50%" and
re-summarized from scratch. That repeated mid-context rewriting, and the
resulting repeated cache invalidation near the same spot, is exactly the
behavior this baseline exists to quantify.
"""
from __future__ import annotations

from typing import Protocol

from agentkv.bench.replay import Turn
from agentkv.context.layout import Tokenizer, render_turns_to_text, render_turns_to_token_ids
from agentkv.policies.base import CompactionPolicy
from agentkv.serving.engine import VLLMEngine


class Summarizer(Protocol):
    def summarize(self, retired_turns: list[Turn]) -> str: ...


class LLMSummarizer:
    """Real summarization via the same local vLLM server used for measurement.

    A separate method (`VLLMEngine.complete_text`), not `generate_step`, so
    summarization calls don't get counted as measured trajectory steps — but
    they do share the physical KV cache pool, which is realistic: a self-hosted
    agent + summarizer sharing one server is a common real deployment shape.
    """

    def __init__(
        self, engine: VLLMEngine, tokenizer: Tokenizer, *, max_summary_tokens: int
    ) -> None:
        self._engine = engine
        self._tokenizer = tokenizer
        self._max_summary_tokens = max_summary_tokens

    def summarize(self, retired_turns: list[Turn]) -> str:
        prompt_text = render_turns_to_text(
            retired_turns,
            header=(
                "Summarize the following agent trajectory turns concisely. "
                "Preserve key facts, tool outputs, and conclusions an agent "
                "would still need later.\n"
            ),
        )
        prompt_ids = self._tokenizer.encode(prompt_text)
        return self._engine.complete_text(prompt_ids, max_tokens=self._max_summary_tokens).strip()


class NaivePolicy(CompactionPolicy):
    name = "naive"

    def __init__(
        self,
        tokenizer: Tokenizer,
        summarizer: Summarizer,
        *,
        threshold_tokens: int,
        retire_fraction: float = 0.5,
    ) -> None:
        if not 0.0 < retire_fraction < 1.0:
            raise ValueError(f"retire_fraction must be in (0, 1), got {retire_fraction}")
        self._tokenizer = tokenizer
        self._summarizer = summarizer
        self._threshold_tokens = threshold_tokens
        self._retire_fraction = retire_fraction

    def maybe_compact(self, turns: list[Turn], step_idx: int) -> tuple[list[Turn], bool]:
        prompt_len = len(render_turns_to_token_ids(turns, self._tokenizer))
        if prompt_len < self._threshold_tokens:
            return turns, False
        if len(turns) < 2:
            return turns, False

        # turns[0] is the anchor (system prompt) — every real agent framework
        # keeps it fixed; Phase 2 formalizes this as `ContextState.anchor`.
        anchor, rest = turns[:1], turns[1:]
        cutoff = max(1, int(len(rest) * self._retire_fraction))
        retired, kept = rest[:cutoff], rest[cutoff:]

        summary_text = self._summarizer.summarize(retired)
        summary_turn = Turn(
            role="system",
            content=f"[compacted summary of {len(retired)} earlier turns]\n{summary_text}",
        )
        return anchor + [summary_turn] + kept, True
