from __future__ import annotations

import pytest

from agentkv.bench.replay import Turn
from agentkv.context.segments import AnchorDriftError
from agentkv.policies.hybrid import HybridPolicy
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


def make_tool_turn(n_words: int, tag: str = "tool") -> Turn:
    output = " ".join([tag] * n_words)
    return Turn(role="tool", tool_results=[{"name": "read_file", "output": output}])


def make_text_turn(role: str, n_words: int, tag: str) -> Turn:
    return Turn(role=role, content=" ".join([tag] * n_words))


def run_steps(policy: HybridPolicy, raw_turns: list[Turn]) -> list[tuple[list[Turn], bool]]:
    """Same harness contract as test_append_only_policy.py's run_steps."""
    context_turns: list[Turn] = []
    history = []
    all_turns = [ANCHOR, *raw_turns]
    for step_idx, new_turn in enumerate(all_turns):
        turns_this_step = context_turns + [new_turn] if step_idx > 0 else [new_turn]
        context_turns, fired = policy.maybe_compact(turns_this_step, step_idx)
        history.append((context_turns, fired))
    return history


def test_below_threshold_does_not_compact():
    policy = HybridPolicy(WordCountTokenizer(), StubSummarizer(), threshold_tokens=10_000)
    history = run_steps(policy, [make_text_turn("user", 5, f"t{i}") for i in range(3)])
    assert all(fired is False for _, fired in history)


def test_large_tool_output_is_evicted_not_summarized_or_kept():
    summarizer = StubSummarizer()
    policy = HybridPolicy(
        WordCountTokenizer(),
        summarizer,
        threshold_tokens=5,
        protect_recent_turns=0,
        large_tool_output_tokens=10,
        small_turn_tokens=3,
    )
    big_tool = make_tool_turn(100, tag="BIGDUMP")
    other = make_text_turn("user", 20, "filler")
    run_steps(policy, [big_tool, other])

    final_live_text = " ".join(t.content or "" for t in policy.state.live)
    final_frozen_text = " ".join(seg.summary_text for seg in policy.state.frozen)
    assert "BIGDUMP" not in final_live_text
    assert "BIGDUMP" not in final_frozen_text
    # the tool turn's content never reached the summarizer either
    for call in summarizer.calls:
        assert big_tool not in call


def test_small_turn_is_kept_verbatim():
    policy = HybridPolicy(
        WordCountTokenizer(),
        StubSummarizer(),
        threshold_tokens=5,
        protect_recent_turns=0,
        large_tool_output_tokens=50,
        small_turn_tokens=3,
    )
    tiny = make_text_turn("user", 1, "hi")
    padding = [make_text_turn("assistant", 20, f"pad{i}") for i in range(3)]
    run_steps(policy, [tiny, *padding])

    assert tiny in policy.state.live


def test_medium_turn_is_summarized_into_frozen_segment():
    summarizer = StubSummarizer()
    policy = HybridPolicy(
        WordCountTokenizer(),
        summarizer,
        threshold_tokens=5,
        protect_recent_turns=0,
        large_tool_output_tokens=100,
        small_turn_tokens=3,
    )
    medium = make_text_turn("assistant", 20, "medium")
    padding = [make_text_turn("user", 20, f"pad{i}") for i in range(3)]
    run_steps(policy, [medium, *padding])

    assert len(policy.state.frozen) >= 1
    assert any(medium in call for call in summarizer.calls)
    assert medium not in policy.state.live


def test_recent_turns_are_protected_regardless_of_route():
    """A turn that would otherwise be evicted (large tool output) survives
    verbatim if it's still within the protected recent window."""
    policy = HybridPolicy(
        WordCountTokenizer(),
        StubSummarizer(),
        threshold_tokens=5,
        protect_recent_turns=5,
        large_tool_output_tokens=10,
        small_turn_tokens=3,
    )
    big_tool = make_tool_turn(100, tag="SHOULDSURVIVE")
    run_steps(policy, [big_tool])  # only 1 raw turn -> always within the protected window

    final_live_text = " ".join(str(t.tool_results) for t in policy.state.live if t.tool_results)
    assert "SHOULDSURVIVE" in final_live_text


def test_fired_false_when_all_candidates_route_to_keep():
    policy = HybridPolicy(
        WordCountTokenizer(),
        StubSummarizer(),
        threshold_tokens=5,
        protect_recent_turns=0,
        large_tool_output_tokens=1001,
        small_turn_tokens=1000,  # everything is "small" -> always routes to keep
    )
    turns = [make_text_turn("user", 5, f"t{i}") for i in range(5)]
    history = run_steps(policy, turns)
    assert all(fired is False for _, fired in history)


def test_invalid_threshold_ordering_raises():
    with pytest.raises(ValueError):
        HybridPolicy(
            WordCountTokenizer(),
            StubSummarizer(),
            threshold_tokens=100,
            small_turn_tokens=50,
            large_tool_output_tokens=50,
        )


def test_negative_protect_recent_turns_raises():
    with pytest.raises(ValueError):
        HybridPolicy(
            WordCountTokenizer(), StubSummarizer(), threshold_tokens=100, protect_recent_turns=-1
        )


def test_anchor_drift_raises():
    policy = HybridPolicy(WordCountTokenizer(), StubSummarizer(), threshold_tokens=10_000)
    policy.maybe_compact([ANCHOR], step_idx=0)

    mutated_anchor = Turn(role="system", content="a completely different system prompt")
    context_turns = policy.state.to_turns()
    tampered = [mutated_anchor, *context_turns[1:], make_text_turn("user", 5, "x")]
    with pytest.raises(AnchorDriftError):
        policy.maybe_compact(tampered, step_idx=1)
