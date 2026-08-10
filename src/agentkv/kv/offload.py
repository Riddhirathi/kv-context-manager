"""Trajectory-aware CPU offload policy (AGENTKV_SPEC.md §4.3): "instead of
discarding evicted blocks, page them to host RAM via vLLM's KV connector
interface (LMCache is the reference implementation). Implement a
trajectory-aware eviction policy that beats LRU by exploiting agent structure:
the anchor is never evictable; frozen segments are re-referenced predictably;
large tool outputs are usually referenced once and never again."

This module is the **policy only** — pure Python, no vLLM or GPU dependency,
fully unit-testable. The `KVConnectorBase_V1` glue that acts on these
decisions against vLLM's real paged KV buffer lives in the sibling
`kv/offload_connector.py`, deliberately split out: importing real vLLM
classes takes over two minutes on this project's own hardware (confirmed
directly — a plain `import vllm` alone timed out past 120s), so keeping that
import out of this module is what keeps this policy's unit tests fast.

Classification works directly on **decoded token text**, not a `ContextState`
object. An earlier version of this module classified from `ContextState`
directly, on the assumption that the experiment driver (which builds
`ContextState`) could hand it to the connector before each request. That
assumption was wrong: this project's `VLLMEngine` talks to vLLM only over
HTTP, and `KVConnectorBase_V1` instances live *inside* the separate vLLM
server process — there is no in-process call the driver can make to the
connector at all. The connector only ever has `request.prompt_token_ids`, so
classification has to work from that alone. It still doesn't need a new
convention: every rendering in this project already tags each turn with a
literal `<role>` marker (`context/layout.py`), and `context/segments.py`'s
`Segment.to_turn` already renders frozen segments with a literal
`"[frozen summary #"` prefix — both are reused here as-is, not invented for
this module.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

_ROLE_TAG_RE = re.compile(r"<(system|user|assistant|tool)>")
_FROZEN_SEGMENT_MARKER = "[frozen summary #"


@dataclass(frozen=True)
class TokenSpan:
    """A `[start, end)` token range within a rendered turn sequence, and
    whether the trajectory-aware policy thinks it is worth the round-trip
    cost of offloading to host RAM if evicted."""

    start: int
    end: int
    worth_offloading: bool
    reason: str  # "anchor" | "frozen" | "large_tool_output" | "live_other"


def classify_decoded_text(
    text: str, *, large_tool_output_chars: int = 2000
) -> list[tuple[int, int, bool, str]]:
    """Classifies each `<role>` tag's span in `text` (character offsets) into
    an offload-worthiness verdict, per spec §4.3's three rules:

    - **anchor**: the *first* `<role>` tag in the text is always the anchor
      (this project's universal "turns[0] is the anchor" convention) —
      always worth offloading.
    - **frozen segments**: a `<system>` tag whose content starts with the
      `"[frozen summary #"` marker — always worth offloading (append-only,
      referenced by every subsequent request for the rest of the trajectory).
    - **large tool outputs**: a `<tool>` tag spanning at least
      `large_tool_output_chars` characters — *not* worth offloading (spec:
      "usually referenced once and never again"). Every other span defaults
      to worth-offloading.

    Returns `(start, end, worth_offloading, reason)` tuples in *character*
    offsets — `classify_token_ids` maps these back to token indices, since
    the connector only ever has token ids, never the original text.
    """
    matches = list(_ROLE_TAG_RE.finditer(text))
    results: list[tuple[int, int, bool, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        role = m.group(1)
        content = text[m.end() : end]
        if i == 0:
            reason, worth = "anchor", True
        elif role == "system" and content.lstrip().startswith(_FROZEN_SEGMENT_MARKER):
            reason, worth = "frozen", True
        elif role == "tool" and (end - start) >= large_tool_output_chars:
            reason, worth = "large_tool_output", False
        else:
            reason, worth = "live_other", True
        results.append((start, end, worth, reason))
    return results


def token_boundary_for_char_offset(
    decode_fn: Callable[[list[int]], str], token_ids: list[int], char_offset: int
) -> int:
    """Binary search for the smallest token count `k` such that
    `len(decode_fn(token_ids[:k])) >= char_offset`. Assumes decoded length is
    monotonically non-decreasing in token count — true for `decode` on every
    tokenizer this project uses, which only ever appends text as more tokens
    are included, never rewrites earlier text based on later tokens."""
    if char_offset <= 0:
        return 0
    lo, hi = 0, len(token_ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(decode_fn(token_ids[:mid])) < char_offset:
            lo = mid + 1
        else:
            hi = mid
    return lo


def classify_token_ids(
    token_ids: list[int],
    decode_fn: Callable[[list[int]], str],
    *,
    large_tool_output_chars: int = 2000,
) -> list[TokenSpan]:
    """`classify_decoded_text`, mapped back to token-index spans via
    `token_boundary_for_char_offset` — the entry point
    `kv/offload_connector.py` actually calls. `decode_fn` is injected (not a
    concrete tokenizer type) so this stays testable with a trivial fake, no
    real tokenizer or vLLM dependency needed here."""
    text = decode_fn(token_ids)
    char_spans = classify_decoded_text(text, large_tool_output_chars=large_tool_output_chars)
    spans: list[TokenSpan] = []
    cursor = 0
    for _char_start, char_end, worth, reason in char_spans:
        token_end = token_boundary_for_char_offset(decode_fn, token_ids, char_end)
        spans.append(TokenSpan(cursor, token_end, worth, reason))
        cursor = token_end
    if cursor < len(token_ids):
        # Trailing tokens after the last matched tag (shouldn't normally
        # happen — every rendering starts with a tag — but default to
        # worth-offloading rather than silently dropping them from the mask).
        spans.append(TokenSpan(cursor, len(token_ids), True, "live_other"))
    return spans


def token_worth_offloading_mask(spans: list[TokenSpan], total_tokens: int) -> list[bool]:
    """Per-token-position offload-worthiness, expanded from `spans`."""
    mask = [False] * total_tokens
    for span in spans:
        for i in range(span.start, min(span.end, total_tokens)):
            mask[i] = span.worth_offloading
    return mask


def blocks_worth_offloading(mask: list[bool], block_size: int) -> list[bool]:
    """Per-*block* offload-worthiness — only full blocks are ever offload
    candidates (matches vLLM's own block-granular caching: a partial trailing
    block isn't stably addressable across requests). A block straddling a
    turn boundary counts as worth offloading if at least half its tokens are
    (the token-level classification is turn-granular, not block-granular, so
    a straddling block needs a tie-breaking rule rather than being left
    unclassified)."""
    n_blocks = len(mask) // block_size
    result = []
    for b in range(n_blocks):
        block_tokens = mask[b * block_size : (b + 1) * block_size]
        result.append(sum(block_tokens) >= block_size / 2)
    return result
