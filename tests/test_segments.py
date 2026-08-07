from __future__ import annotations

import pytest

from agentkv.bench.replay import Turn
from agentkv.context.segments import (
    AnchorDriftError,
    AnchorHygieneGuard,
    ContextState,
    Segment,
    hash_anchor_rendering,
)


class WordCountTokenizer:
    def encode(self, text: str) -> list[int]:
        return [len(w) for w in text.split()]


def test_segment_to_turn_is_deterministic():
    seg = Segment(step_created=3, summary_text="the account balance is 42", n_turns_summarized=5)
    assert seg.to_turn(index=0) == seg.to_turn(index=0)
    assert "frozen summary #0" in seg.to_turn(index=0).content  # type: ignore[operator]
    assert "5 turns" in seg.to_turn(index=0).content  # type: ignore[operator]


def test_context_state_to_turns_flattens_in_order():
    anchor = Turn(role="system", content="anchor")
    seg = Segment(step_created=1, summary_text="summary", n_turns_summarized=2)
    live = [Turn(role="user", content="live turn")]
    state = ContextState(anchor=anchor, frozen=[seg], live=live)

    turns = state.to_turns()
    assert turns[0] == anchor
    assert turns[1] == seg.to_turn(index=0)
    assert turns[2] == live[0]


def test_hash_anchor_rendering_stable_for_identical_content():
    tokenizer = WordCountTokenizer()
    anchor = Turn(role="system", content="you are an agent")
    assert hash_anchor_rendering(anchor, tokenizer) == hash_anchor_rendering(anchor, tokenizer)


def test_hash_anchor_rendering_changes_with_content():
    tokenizer = WordCountTokenizer()
    a = Turn(role="system", content="you are an agent")
    b = Turn(role="system", content="you are an agent, timestamp=12345")
    assert hash_anchor_rendering(a, tokenizer) != hash_anchor_rendering(b, tokenizer)


def test_anchor_hygiene_guard_passes_stable_anchor():
    tokenizer = WordCountTokenizer()
    guard = AnchorHygieneGuard(tokenizer)
    anchor = Turn(role="system", content="stable anchor")
    for step_idx in range(5):
        guard.check(anchor, step_idx=step_idx)  # must not raise


def test_anchor_hygiene_guard_raises_on_change():
    tokenizer = WordCountTokenizer()
    guard = AnchorHygieneGuard(tokenizer)
    guard.check(Turn(role="system", content="stable anchor"), step_idx=0)
    with pytest.raises(AnchorDriftError):
        guard.check(Turn(role="system", content="stable anchor, mutated"), step_idx=1)
