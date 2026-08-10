from __future__ import annotations

import pytest

from agentkv.bench.replay import Turn
from agentkv.policies.kv_evict import KVEvictPolicy


class WordCountTokenizer:
    """Deterministic fake: one token id per word, no vLLM/GPU dependency."""

    def encode(self, text: str) -> list[int]:
        return [len(w) for w in text.split()]


def make_turns(n: int, words_per_turn: int = 20) -> list[Turn]:
    anchor = Turn(role="system", content="you are an on-call agent " * 3)
    turns = [anchor]
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        content = ("w" * 3 + " ") * words_per_turn + f"turn{i}"
        turns.append(Turn(role=role, content=content))
    return turns


def test_below_threshold_does_not_evict():
    tokenizer = WordCountTokenizer()
    policy = KVEvictPolicy(tokenizer, threshold_tokens=10_000, window_turns=3)

    turns = make_turns(5)
    result, fired = policy.maybe_compact(turns, step_idx=5)

    assert fired is False
    assert result == turns


def test_above_threshold_keeps_anchor_and_newest_window_turns():
    tokenizer = WordCountTokenizer()
    policy = KVEvictPolicy(tokenizer, threshold_tokens=5, window_turns=3)

    turns = make_turns(10)
    result, fired = policy.maybe_compact(turns, step_idx=10)

    assert fired is True
    # anchor preserved verbatim, unmoved
    assert result[0] == turns[0]
    # exactly the newest window_turns non-anchor turns survive, verbatim
    assert result[1:] == turns[-3:]
    assert len(result) == 4


def test_evicted_turns_leave_no_trace():
    """The defining difference from naive/append_only: no summary is ever
    synthesized for evicted content — it is simply gone."""
    tokenizer = WordCountTokenizer()
    policy = KVEvictPolicy(tokenizer, threshold_tokens=5, window_turns=2)

    turns = make_turns(8)
    result, fired = policy.maybe_compact(turns, step_idx=8)

    assert fired is True
    assert len(result) == 3  # anchor + 2-turn window, nothing else
    for turn in result[1:]:
        assert "compacted" not in (turn.content or "")
        assert "summary" not in (turn.content or "")


def test_threshold_crossed_but_turn_count_still_within_window_does_not_fire():
    """Token threshold can trip before turn count exceeds the window (e.g.
    very verbose early turns) — nothing to evict yet in that case."""
    tokenizer = WordCountTokenizer()
    policy = KVEvictPolicy(tokenizer, threshold_tokens=5, window_turns=50)

    turns = make_turns(10)
    result, fired = policy.maybe_compact(turns, step_idx=10)

    assert fired is False
    assert result == turns


def test_sliding_window_drops_oldest_entry_each_step_once_full():
    tokenizer = WordCountTokenizer()
    policy = KVEvictPolicy(tokenizer, threshold_tokens=5, window_turns=3)

    turns = make_turns(10)
    first, fired_first = policy.maybe_compact(turns, step_idx=10)
    assert fired_first is True
    assert first[1:] == turns[-3:]

    grown = first + make_turns(1)[1:]  # harness contract: prev result + 1 new raw turn
    second, fired_second = policy.maybe_compact(grown, step_idx=11)

    assert fired_second is True
    # the oldest window entry (first[1]) is dropped; the two newest carry
    # over plus the freshly appended turn.
    assert second[1:] == [first[2], first[3], grown[-1]]


def test_window_turns_must_be_positive():
    tokenizer = WordCountTokenizer()
    with pytest.raises(ValueError):
        KVEvictPolicy(tokenizer, threshold_tokens=100, window_turns=0)


def test_below_two_turns_never_fires():
    tokenizer = WordCountTokenizer()
    policy = KVEvictPolicy(tokenizer, threshold_tokens=0, window_turns=1)
    anchor_only = make_turns(0)
    result, fired = policy.maybe_compact(anchor_only, step_idx=0)
    assert fired is False
    assert result == anchor_only
