from __future__ import annotations

from agentkv.bench.replay import Turn
from agentkv.policies.none import NoOpPolicy


def make_turns(n: int, words_per_turn: int = 200) -> list[Turn]:
    anchor = Turn(role="system", content="you are an on-call agent " * 3)
    turns = [anchor]
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        content = ("w" * 3 + " ") * words_per_turn + f"turn{i}"
        turns.append(Turn(role=role, content=content))
    return turns


def test_never_fires_regardless_of_length():
    policy = NoOpPolicy()
    turns = make_turns(50)
    result, fired = policy.maybe_compact(turns, step_idx=50)

    assert fired is False
    assert result == turns


def test_returns_turns_unchanged_across_repeated_calls():
    policy = NoOpPolicy()
    turns = make_turns(3)
    for step_idx in range(3):
        result, fired = policy.maybe_compact(turns[: step_idx + 2], step_idx)
        assert fired is False
        assert result == turns[: step_idx + 2]


def test_name_is_none():
    assert NoOpPolicy.name == "none"
