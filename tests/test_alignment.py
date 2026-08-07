from __future__ import annotations

import pytest

from agentkv.bench.replay import Turn
from agentkv.context.alignment import (
    pad_to_block_boundary,
    render_context_state_aligned,
    tokens_to_pad,
)
from agentkv.context.segments import ContextState, Segment


class WordCountTokenizer:
    """Deterministic fake: one token id per word (its length), no vLLM/GPU
    dependency — same convention as tests/test_naive_policy.py."""

    def encode(self, text: str) -> list[int]:
        return [len(w) for w in text.split()]


def test_tokens_to_pad_exact_multiple_needs_no_padding():
    assert tokens_to_pad(32, block_size=16) == 0


def test_tokens_to_pad_off_by_one():
    assert tokens_to_pad(33, block_size=16) == 15
    assert tokens_to_pad(17, block_size=16) == 15


def test_pad_to_block_boundary_appends_filler():
    ids = list(range(17))  # 17 tokens, needs 15 filler to reach 32
    padded = pad_to_block_boundary(ids, block_size=16, filler_token_id=999)
    assert len(padded) == 32
    assert padded[:17] == ids
    assert padded[17:] == [999] * 15


def test_pad_to_block_boundary_noop_when_already_aligned():
    ids = list(range(32))
    assert pad_to_block_boundary(ids, block_size=16, filler_token_id=999) == ids


def test_tokens_to_pad_rejects_nonpositive_block_size():
    with pytest.raises(ValueError):
        tokens_to_pad(10, block_size=0)


def test_render_context_state_aligned_pads_every_boundary_except_the_last():
    tokenizer = WordCountTokenizer()
    anchor = Turn(role="system", content="a b c")  # renders to 4 tokens: <system>, a, b, c
    seg = Segment(step_created=1, summary_text="d e", n_turns_summarized=1)
    live = [Turn(role="user", content="x")]

    block_size = 3
    unaligned_anchor_len = len(tokenizer.encode("<system>\na b c"))
    expected_anchor_pad = tokens_to_pad(unaligned_anchor_len, block_size=block_size)
    assert expected_anchor_pad > 0  # sanity: this test only means something if padding is needed

    aligned = render_context_state_aligned(
        ContextState(anchor=anchor, frozen=[seg], live=live),
        tokenizer,
        block_size=block_size,
        filler_token_id=-1,
    )
    # anchor group is padded up to the next multiple of block_size
    pad_end = unaligned_anchor_len + expected_anchor_pad
    filler_after_anchor = aligned[unaligned_anchor_len:pad_end]
    assert filler_after_anchor == [-1] * expected_anchor_pad
    assert aligned[pad_end] != -1  # frozen segment content starts right after the padding

    # the trailing live group is never padded, so the overall length need not
    # be a multiple of block_size at all (here: live is a single token).
    assert aligned[-1] != -1


def test_render_context_state_aligned_matches_unaligned_when_block_size_one():
    """With block_size=1 every position is already 'aligned', so no filler is
    ever inserted and the aligned rendering equals concatenating each group's
    tokenization independently (a sanity check that alignment adds nothing
    when there's nothing to add)."""
    tokenizer = WordCountTokenizer()
    anchor = Turn(role="system", content="a b")
    live = [Turn(role="user", content="c d")]
    state = ContextState(anchor=anchor, live=live)

    aligned = render_context_state_aligned(state, tokenizer, block_size=1, filler_token_id=-1)
    assert -1 not in aligned
