"""Block-boundary alignment (AGENTKV_SPEC.md §2.3).

Cache reuse is block-granular (vLLM's default `block_size = 16` tokens). If a
segment boundary sits one token before a block boundary, any future edit
immediately after that boundary invalidates the *whole* straddling block, not
just the tokens that actually changed. Padding each segment boundary out to a
block multiple with inert filler tokens turns that partial-block waste into
an explicit, small, bounded cost (at most `block_size - 1` filler tokens per
boundary) instead of an unpredictable one that depends on exactly where the
next edit happens to land.

Spec §2.3 requires the filler's neutrality — "does not perturb model
behavior" — be validated empirically (measure action agreement with and
without padding), not assumed; this module only provides the padding
mechanism, not that validation, which belongs to `bench/agreement.py`
(Phase 3).
"""
from __future__ import annotations

from agentkv.bench.replay import Turn
from agentkv.context.layout import Tokenizer, render_turns_to_token_ids
from agentkv.context.segments import ContextState


def tokens_to_pad(token_count: int, *, block_size: int) -> int:
    """Number of filler tokens needed after `token_count` tokens to land
    exactly on the next block boundary. Returns 0 if already aligned."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    remainder = token_count % block_size
    return 0 if remainder == 0 else block_size - remainder


def pad_to_block_boundary(
    token_ids: list[int], *, block_size: int, filler_token_id: int
) -> list[int]:
    """Appends `filler_token_id` repeated just enough times to align
    `token_ids` to the next block boundary."""
    pad = tokens_to_pad(len(token_ids), block_size=block_size)
    return token_ids + [filler_token_id] * pad


def render_context_state_aligned(
    state: ContextState,
    tokenizer: Tokenizer,
    *,
    block_size: int,
    filler_token_id: int,
) -> list[int]:
    """Renders anchor + frozen segments + live turns to token ids, padding
    after the anchor and after every frozen segment so the following segment
    always starts on a block boundary. The trailing `live` group is never
    padded — nothing follows it yet this step, and padding it would just
    waste tokens on a boundary that doesn't exist until the next compaction
    event freezes it.

    Each group is tokenized independently and concatenated, rather than
    joining all groups into one string and tokenizing once (as
    `render_turns_to_token_ids` does): padding must land at an exact,
    predictable boundary, which is only possible if tokenization happens
    group-by-group. This means a token right at a group boundary may be
    encoded slightly differently than it would be as part of one continuous
    string (a BPE merge that would have spanned the boundary can't happen)
    — an accepted, small tradeoff of the alignment scheme itself, not a bug
    in this function.
    """
    groups: list[list[Turn]] = [[state.anchor], *([s] for s in state.frozen_turns()), state.live]
    ids: list[int] = []
    for i, group in enumerate(groups):
        ids.extend(render_turns_to_token_ids(group, tokenizer))
        is_last = i == len(groups) - 1
        if not is_last:
            ids = pad_to_block_boundary(ids, block_size=block_size, filler_token_id=filler_token_id)
    return ids
