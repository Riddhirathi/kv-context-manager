from __future__ import annotations

from agentkv.bench.replay import Turn
from agentkv.policies.naive import NaivePolicy, Summarizer


class WordCountTokenizer:
    """Deterministic fake: one token id per word, no vLLM/GPU dependency."""

    def encode(self, text: str) -> list[int]:
        return [len(w) for w in text.split()]


class StubSummarizer(Summarizer):
    def __init__(self) -> None:
        self.calls: list[list[Turn]] = []

    def summarize(self, retired_turns: list[Turn]) -> str:
        self.calls.append(retired_turns)
        return f"summary of {len(retired_turns)} turns"


def make_turns(n: int, words_per_turn: int = 20) -> list[Turn]:
    anchor = Turn(role="system", content="you are an on-call agent " * 3)
    turns = [anchor]
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        content = ("w" * 3 + " ") * words_per_turn + f"turn{i}"
        turns.append(Turn(role=role, content=content))
    return turns


def test_below_threshold_does_not_compact():
    tokenizer = WordCountTokenizer()
    summarizer = StubSummarizer()
    policy = NaivePolicy(tokenizer, summarizer, threshold_tokens=10_000, retire_fraction=0.5)

    turns = make_turns(3)
    result, fired = policy.maybe_compact(turns, step_idx=3)

    assert fired is False
    assert result == turns
    assert summarizer.calls == []


def test_above_threshold_compacts_oldest_half_and_keeps_anchor():
    tokenizer = WordCountTokenizer()
    summarizer = StubSummarizer()
    policy = NaivePolicy(tokenizer, summarizer, threshold_tokens=5, retire_fraction=0.5)

    turns = make_turns(10)
    result, fired = policy.maybe_compact(turns, step_idx=10)

    assert fired is True
    # anchor preserved verbatim, unmoved
    assert result[0] == turns[0]
    # a synthetic summary turn was inserted right after the anchor
    assert result[1].content is not None
    assert "compacted summary of 5 earlier turns" in result[1].content
    # the newest half of the non-anchor turns survives verbatim
    assert result[2:] == turns[6:]
    assert len(summarizer.calls) == 1
    assert len(summarizer.calls[0]) == 5


def test_repeated_compaction_resweeps_previous_summary():
    """Naive is deliberately dumb (spec §1.1): a prior summary turn is just
    another turn subject to the next round's oldest-50% cut, unlike Phase 2's
    append-only policy."""
    tokenizer = WordCountTokenizer()
    summarizer = StubSummarizer()
    policy = NaivePolicy(tokenizer, summarizer, threshold_tokens=5, retire_fraction=0.5)

    turns = make_turns(10)
    compacted, fired = policy.maybe_compact(turns, step_idx=10)
    assert fired is True

    grown = compacted + make_turns(10)[1:]  # simulate more raw turns appended
    compacted_again, fired_again = policy.maybe_compact(grown, step_idx=20)

    assert fired_again is True
    # the first round's summary turn is among the retired (oldest) turns this time
    first_summary = compacted[1]
    retired_this_round = summarizer.calls[1]
    assert first_summary in retired_this_round
