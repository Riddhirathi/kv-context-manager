"""ContextState model (AGENTKV_SPEC.md §2.1): the core Phase 2 data structure.

    ContextState:
      anchor:  Segment            # system + tools + task, immutable
      frozen:  list[Segment]      # append-only summaries, never edited
      live:    list[Turn]         # verbatim recent turns

Phase 1's naive policy encoded "the anchor" and "a summary" as plain `Turn`
objects inside one flat list, with no structure enforcing that a prior
summary is never rewritten (spec §1.1: it deliberately re-sweeps prior
summaries). `ContextState` makes that distinction a type-level guarantee
instead of a convention: `frozen` is only ever appended to, never mutated in
place, by any code in this repo.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from agentkv.bench.replay import Turn
from agentkv.context.layout import Tokenizer, render_turns_to_token_ids


@dataclass(frozen=True)
class Segment:
    """One frozen, append-only summary chunk (spec §3's "Frozen segment":
    "written exactly once and never edited afterwards")."""

    step_created: int
    summary_text: str
    n_turns_summarized: int

    def to_turn(self, *, index: int) -> Turn:
        """Renders this segment as a `Turn` so it flows through the same
        `render_turns_to_token_ids` every policy shares (spec §2.1 doesn't
        require its own tokenization path — only its own rule for *which*
        turns to send). Content is a pure function of the segment's own
        fields, so re-rendering the same `Segment` twice is always
        byte-identical — the property the whole append-only cache argument
        depends on."""
        return Turn(
            role="system",
            content=(
                f"[frozen summary #{index}, {self.n_turns_summarized} turns, "
                f"created at step {self.step_created}]\n{self.summary_text}"
            ),
        )


@dataclass
class ContextState:
    """Mutable only via `live` reassignment and `frozen.append` — see
    `policies/append_only.py`, the sole writer of this state during a
    trajectory replay."""

    anchor: Turn
    frozen: list[Segment] = field(default_factory=list)
    live: list[Turn] = field(default_factory=list)

    def frozen_turns(self) -> list[Turn]:
        return [seg.to_turn(index=i) for i, seg in enumerate(self.frozen)]

    def to_turns(self) -> list[Turn]:
        """Flattens anchor + frozen segments (in append order) + live turns
        into the turn sequence to send this step."""
        return [self.anchor, *self.frozen_turns(), *self.live]


class AnchorDriftError(RuntimeError):
    """Raised when the anchor's rendered tokens changed between steps (spec
    §2.4: "a single timestamp or reordered JSON key in the system prompt
    destroys 100% of the cache")."""


def hash_anchor_rendering(anchor: Turn, tokenizer: Tokenizer) -> str:
    """Content hash of exactly what the anchor renders to as token ids.

    Deliberately hashes the *rendering*, not a canonicalized re-serialization
    of the `Turn` (e.g. `json.dumps(..., sort_keys=True)`): a canonical form
    would silently normalize away exactly the bug class §2.4 warns about — a
    reordered JSON key in a tool schema changes the actual tokenized bytes
    (`render_turns_to_token_ids` uses plain `json.dumps`, which preserves
    insertion order) even though a sorted-keys hash would call the two
    renderings identical.
    """
    ids = render_turns_to_token_ids([anchor], tokenizer)
    return hashlib.sha256(repr(ids).encode()).hexdigest()


class AnchorHygieneGuard:
    """Fails loudly the first time the anchor's rendering changes across
    steps (spec §2.4's "regression test that hashes the anchor every step and
    fails on change") — used both inside `AppendOnlyPolicy` at replay time and
    directly in tests."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._first_hash: str | None = None

    def check(self, anchor: Turn, *, step_idx: int) -> None:
        current = hash_anchor_rendering(anchor, self._tokenizer)
        if self._first_hash is None:
            self._first_hash = current
            return
        if current != self._first_hash:
            raise AnchorDriftError(
                f"step {step_idx}: anchor rendering changed since step 0 "
                f"({self._first_hash[:12]}... -> {current[:12]}...) — this "
                "invalidates 100% of the prefix cache every step and means "
                "something upstream is mutating the anchor (a timestamp, "
                "reordered JSON keys in a tool schema, etc.)."
            )
