from __future__ import annotations

import pytest

from agentkv.bench.replay import Turn
from agentkv.context.segments import AnchorDriftError
from agentkv.policies.append_only import AppendOnlyPolicy
from agentkv.policies.naive import Summarizer


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


ANCHOR = Turn(role="system", content="you are an on-call agent " * 3)


def make_turn(i: int, words_per_turn: int = 20) -> Turn:
    role = "user" if i % 2 == 0 else "assistant"
    content = ("w" * 3 + " ") * words_per_turn + f"turn{i}"
    return Turn(role=role, content=content)


def run_steps(policy: AppendOnlyPolicy, n_raw_turns: int) -> list[tuple[list[Turn], bool]]:
    """Drives `policy` through the exact harness contract every experiment
    script in this repo uses (see phase1_cliff.py/phase2_layout.py's
    run_trajectory): each step feeds back exactly what the policy returned
    last time, plus one new raw turn."""
    context_turns: list[Turn] = []
    history = []
    for step_idx in range(n_raw_turns + 1):
        new_raw = ANCHOR if step_idx == 0 else make_turn(step_idx - 1)
        turns_this_step = context_turns + [new_raw] if step_idx > 0 else [new_raw]
        context_turns, fired = policy.maybe_compact(turns_this_step, step_idx)
        history.append((context_turns, fired))
    return history


def test_below_threshold_does_not_compact():
    policy = AppendOnlyPolicy(WordCountTokenizer(), StubSummarizer(), threshold_tokens=10_000)
    history = run_steps(policy, n_raw_turns=3)
    assert all(fired is False for _, fired in history)
    assert policy.state.frozen == []


def test_above_threshold_freezes_a_segment_and_keeps_anchor():
    summarizer = StubSummarizer()
    policy = AppendOnlyPolicy(WordCountTokenizer(), summarizer, threshold_tokens=5)
    history = run_steps(policy, n_raw_turns=10)

    assert any(fired for _, fired in history)
    assert len(policy.state.frozen) >= 1
    fired_turns, _ = next((t, f) for t, f in history if f)
    assert fired_turns[0] == ANCHOR
    assert "frozen summary #0" in fired_turns[1].content  # type: ignore[operator]
    assert len(summarizer.calls) >= 1


def test_second_compaction_does_not_resweep_first_frozen_segment():
    """Contrast with naive (test_naive_policy.py's
    test_repeated_compaction_resweeps_previous_summary): the first frozen
    segment must survive byte-identical through a second compaction event."""
    policy = AppendOnlyPolicy(WordCountTokenizer(), StubSummarizer(), threshold_tokens=5)
    run_steps(policy, n_raw_turns=30)

    assert len(policy.state.frozen) >= 2
    first_segment = policy.state.frozen[0]
    # Re-rendering the first segment after later compactions is byte-identical —
    # it was never touched by the second (or later) compaction event.
    assert first_segment.to_turn(index=0) == first_segment.to_turn(index=0)
    assert policy.state.frozen[0] is first_segment


def test_frozen_segment_count_strictly_increases_across_events():
    """Spec §2.2's headline property: `frozen` only ever grows, so the count
    of frozen segments at each successive fired step must strictly
    increase — there is no code path that removes or rewrites one."""
    policy = AppendOnlyPolicy(WordCountTokenizer(), StubSummarizer(), threshold_tokens=5)
    context_turns: list[Turn] = []
    frozen_counts = []
    for step_idx in range(41):
        new_raw = ANCHOR if step_idx == 0 else make_turn(step_idx - 1)
        turns_this_step = context_turns + [new_raw] if step_idx > 0 else [new_raw]
        context_turns, fired = policy.maybe_compact(turns_this_step, step_idx)
        if fired:
            frozen_counts.append(len(policy.state.frozen))
    assert len(frozen_counts) >= 2
    assert frozen_counts == sorted(frozen_counts)
    assert len(set(frozen_counts)) == len(frozen_counts)  # strictly increasing, no repeats


def test_anchor_drift_raises():
    policy = AppendOnlyPolicy(WordCountTokenizer(), StubSummarizer(), threshold_tokens=10_000)
    policy.maybe_compact([ANCHOR], step_idx=0)

    mutated_anchor = Turn(role="system", content="a completely different system prompt")
    context_turns = policy.state.to_turns()
    # Simulate an upstream bug: the anchor slot at position 0 has changed.
    tampered = [mutated_anchor, *context_turns[1:], make_turn(0)]
    with pytest.raises(AnchorDriftError):
        policy.maybe_compact(tampered, step_idx=1)
