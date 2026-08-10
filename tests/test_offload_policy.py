from __future__ import annotations

from agentkv.kv.offload import (
    TokenSpan,
    blocks_worth_offloading,
    classify_decoded_text,
    classify_token_ids,
    token_boundary_for_char_offset,
    token_worth_offloading_mask,
)


class WordTokenizer:
    """Deterministic fake supporting both encode (one id per word) and decode
    (space-joined words back) — round-trips exactly for whitespace-split
    text, no vLLM/GPU dependency."""

    def __init__(self) -> None:
        self._vocab: list[str] = []
        self._index: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self._index:
                self._index[word] = len(self._vocab)
                self._vocab.append(word)
            ids.append(self._index[word])
        return ids

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self._vocab[i] for i in token_ids)


# --- classify_decoded_text ---------------------------------------------


def test_first_role_tag_is_always_anchor():
    text = "<system> you are an agent"
    spans = classify_decoded_text(text)
    assert len(spans) == 1
    start, end, worth, reason = spans[0]
    assert reason == "anchor"
    assert worth is True
    assert start == 0


def test_system_tag_with_frozen_marker_is_frozen():
    text = (
        "<system> anchor text "
        "<system> [frozen summary #0, 3 turns, created at step 1] the summary"
    )
    spans = classify_decoded_text(text)
    assert spans[0][3] == "anchor"
    assert spans[1][3] == "frozen"
    assert spans[1][2] is True


def test_system_tag_without_frozen_marker_after_anchor_is_live_other():
    """Only the *first* system tag is the anchor; a later plain <system> tag
    (no frozen marker) is just another live turn, not automatically frozen."""
    text = "<system> anchor text <system> a plain system turn, not a summary"
    spans = classify_decoded_text(text)
    assert spans[1][3] == "live_other"
    assert spans[1][2] is True


def test_short_tool_output_is_worth_offloading():
    text = "<system> anchor <tool> a short result"
    spans = classify_decoded_text(text, large_tool_output_chars=500)
    tool_span = spans[-1]
    assert tool_span[3] == "live_other"
    assert tool_span[2] is True


def test_large_tool_output_is_not_worth_offloading():
    text = "<system> anchor <tool> " + ("word " * 50)
    spans = classify_decoded_text(text, large_tool_output_chars=10)
    tool_span = spans[-1]
    assert tool_span[3] == "large_tool_output"
    assert tool_span[2] is False


def test_non_tool_span_defaults_worth_offloading_regardless_of_size():
    text = "<system> anchor <assistant> " + ("word " * 50)
    spans = classify_decoded_text(text, large_tool_output_chars=10)
    assistant_span = spans[-1]
    assert assistant_span[3] == "live_other"
    assert assistant_span[2] is True


def test_spans_are_contiguous_and_cover_full_text():
    text = "<system> anchor <user> hi <tool> result <assistant> reply"
    spans = classify_decoded_text(text)
    assert spans[0][0] == 0
    for prev, nxt in zip(spans, spans[1:], strict=False):
        assert prev[1] == nxt[0]
    assert spans[-1][1] == len(text)


# --- token_boundary_for_char_offset -------------------------------------


def test_token_boundary_for_char_offset_finds_exact_word_boundary():
    tokenizer = WordTokenizer()
    ids = tokenizer.encode("alpha beta gamma delta")
    # "alpha beta" is 10 chars; boundary for offset 10 should be token index 2.
    boundary = token_boundary_for_char_offset(tokenizer.decode, ids, 10)
    assert tokenizer.decode(ids[:boundary]) == "alpha beta"


def test_token_boundary_for_char_offset_zero_is_zero():
    tokenizer = WordTokenizer()
    ids = tokenizer.encode("alpha beta")
    assert token_boundary_for_char_offset(tokenizer.decode, ids, 0) == 0


def test_token_boundary_for_char_offset_full_length_is_all_tokens():
    tokenizer = WordTokenizer()
    ids = tokenizer.encode("alpha beta gamma")
    full_text = tokenizer.decode(ids)
    boundary = token_boundary_for_char_offset(tokenizer.decode, ids, len(full_text))
    assert boundary == len(ids)


# --- classify_token_ids (end-to-end) ------------------------------------


def test_classify_token_ids_end_to_end():
    tokenizer = WordTokenizer()
    text = (
        "<system> anchor content here "
        "<system> [frozen summary #0, 2 turns, created at step 1] a summary "
        "<user> hello there "
        "<tool> " + ("bigoutput " * 20) + " "
        "<assistant> a short reply"
    )
    ids = tokenizer.encode(text)
    spans = classify_token_ids(ids, tokenizer.decode, large_tool_output_chars=50)

    reasons = [s.reason for s in spans]
    assert reasons == ["anchor", "frozen", "live_other", "large_tool_output", "live_other"]
    assert [s.worth_offloading for s in spans] == [True, True, True, False, True]

    # Spans are contiguous and cover every token.
    assert spans[0].start == 0
    for prev, nxt in zip(spans, spans[1:], strict=False):
        assert prev.end == nxt.start
    assert spans[-1].end == len(ids)


# --- token_worth_offloading_mask / blocks_worth_offloading --------------
# (independent of how spans were derived — unchanged by the switch away
# from ContextState-based classification)


def test_token_worth_offloading_mask_matches_spans():
    spans = [
        TokenSpan(0, 3, True, "anchor"),
        TokenSpan(3, 8, False, "large_tool_output"),
        TokenSpan(8, 10, True, "live_other"),
    ]
    mask = token_worth_offloading_mask(spans, total_tokens=10)
    assert mask == [True, True, True, False, False, False, False, False, True, True]


def test_blocks_worth_offloading_majority_vote():
    # block_size=4: block0 all True -> True; block1 all False -> False;
    # block2 split 2/2 -> tie counts as worth-offloading (>=).
    mask = [True, True, True, True, False, False, False, False, True, True, False, False]
    result = blocks_worth_offloading(mask, block_size=4)
    assert result == [True, False, True]


def test_blocks_worth_offloading_ignores_partial_trailing_block():
    mask = [True, True, True, True, True, True]  # 1.5 blocks at block_size=4
    result = blocks_worth_offloading(mask, block_size=4)
    assert len(result) == 1
    assert result == [True]
