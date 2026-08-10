"""Hybrid per-segment routing (AGENTKV_SPEC.md §4.4): "the contribution. Per
segment, choose one of three actions: keep verbatim (cheap, no quality loss,
costs context length); KV-evict (free prefill, lossy, unrecoverable); textually
summarize into a frozen segment (costs re-prefill from divergence, semantically
smart). Route by estimated value: predicted future attention mass × segment
size × recompute cost. Start with a hand-tuned heuristic. A learned policy is a
stretch goal, not a requirement."

**On "predicted future attention mass"**: spec's own suggested routing signal
is exactly what §4.2 (`attn_evict.py`) found infeasible to compute affordably on
this hardware — see `results/phase4/phase4_attn_evict_summary.md`. Spec's own
fallback bar ("start with a hand-tuned heuristic") is used instead: a `tool`
turn's rendered size stands in for "predicted future value," reusing the exact
signal already validated twice in this project (`kv_evict.py`'s spirit and
`kv/offload.py`'s `classify_decoded_text`, both keyed on spec's own repeated
observation that "large tool outputs are usually referenced once and never
again"). Not learned, not attention-based — a real, hand-tuned heuristic, which
is spec's stated minimum bar for this sub-phase.

The routing rule, applied to every `live` turn old enough to not be in the
protected recent window:

- a `tool` turn at least `large_tool_output_tokens` long -> **KV-evict**: dropped
  outright, no replacement text (spec: "free prefill, lossy, unrecoverable" —
  same mechanism as `kv_evict.py`, just selective rather than blanket-windowed).
- a turn under `small_turn_tokens` -> **keep verbatim**: too cheap to be worth
  the overhead of summarizing or losing outright.
- everything else -> **textually summarize**: batched with any other
  same-round "summarize" turns into one new frozen `Segment`, exactly like
  `append_only.py`'s mechanism.

Reuses `ContextState`/`Segment`/`AnchorHygieneGuard` from `context/segments.py`
and the same statefulness contract `append_only.py` documents: a fresh instance
per trajectory, `turns[-1]` is always exactly the one new raw turn since the
last call.
"""
from __future__ import annotations

from agentkv.bench.replay import Turn
from agentkv.context.layout import Tokenizer, render_turns_to_token_ids
from agentkv.context.segments import AnchorHygieneGuard, ContextState, Segment
from agentkv.policies.base import CompactionPolicy
from agentkv.policies.naive import Summarizer


class HybridPolicy(CompactionPolicy):
    name = "hybrid"

    def __init__(
        self,
        tokenizer: Tokenizer,
        summarizer: Summarizer,
        *,
        threshold_tokens: int,
        protect_recent_turns: int = 10,
        large_tool_output_tokens: int = 500,
        small_turn_tokens: int = 50,
    ) -> None:
        if protect_recent_turns < 0:
            raise ValueError(f"protect_recent_turns must be >= 0, got {protect_recent_turns}")
        if small_turn_tokens >= large_tool_output_tokens:
            raise ValueError(
                "small_turn_tokens must be < large_tool_output_tokens, got "
                f"{small_turn_tokens} >= {large_tool_output_tokens}"
            )
        self._tokenizer = tokenizer
        self._summarizer = summarizer
        self._threshold_tokens = threshold_tokens
        self._protect_recent_turns = protect_recent_turns
        self._large_tool_output_tokens = large_tool_output_tokens
        self._small_turn_tokens = small_turn_tokens
        self._state: ContextState | None = None
        self._anchor_guard = AnchorHygieneGuard(tokenizer)

    @property
    def state(self) -> ContextState:
        assert self._state is not None, "maybe_compact has not been called yet"
        return self._state

    def _route(self, turn: Turn) -> str:
        """Returns "evict" | "keep" | "summarize" for one candidate turn."""
        turn_size = len(render_turns_to_token_ids([turn], self._tokenizer))
        if turn.role == "tool" and turn_size >= self._large_tool_output_tokens:
            return "evict"
        if turn_size < self._small_turn_tokens:
            return "keep"
        return "summarize"

    def maybe_compact(self, turns: list[Turn], step_idx: int) -> tuple[list[Turn], bool]:
        if self._state is None:
            # First call: turns is just [anchor] (harness contract shared with
            # append_only.py — see that module's docstring).
            self._state = ContextState(anchor=turns[0], live=list(turns[1:]))
        else:
            self._state.live.append(turns[-1])

        self._anchor_guard.check(turns[0], step_idx=step_idx)

        prompt_len = len(render_turns_to_token_ids(self._state.to_turns(), self._tokenizer))
        if prompt_len < self._threshold_tokens or len(self._state.live) < 2:
            return self._state.to_turns(), False

        protect_from = max(0, len(self._state.live) - self._protect_recent_turns)
        candidates, protected = self._state.live[:protect_from], self._state.live[protect_from:]
        if not candidates:
            return self._state.to_turns(), False

        kept: list[Turn] = []
        to_summarize: list[Turn] = []
        for turn in candidates:
            route = self._route(turn)
            if route == "keep":
                kept.append(turn)
            elif route == "summarize":
                to_summarize.append(turn)
            # "evict": neither list — dropped outright, no replacement.

        if kept == candidates:
            # Every candidate routed to "keep" — nothing actually changed.
            return self._state.to_turns(), False

        if to_summarize:
            summary_text = self._summarizer.summarize(to_summarize)
            self._state.frozen.append(
                Segment(
                    step_created=step_idx,
                    summary_text=summary_text,
                    n_turns_summarized=len(to_summarize),
                )
            )
        self._state.live = kept + protected
        return self._state.to_turns(), True
