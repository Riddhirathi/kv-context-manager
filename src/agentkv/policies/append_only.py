"""Append-only compaction (AGENTKV_SPEC.md §2.2): the core L1 contribution.

On a compaction event at step *t*:
1. Select the oldest turns in `live` for retirement.
2. Summarize them into a new `Segment` S_t.
3. Append S_t to `frozen`. Never modify S_1..S_{t-1}.
4. Drop the retired turns from `live`.

Contrast with `policies/naive.py`: naive re-sweeps a prior summary turn into
the *next* round's "oldest 50%" and re-summarizes it from scratch, so its
divergence point regresses to the same spot in the context every time. This
policy's divergence point after a compaction event is always the end of
`frozen[:-1]` — everything before the newest summary stays cached. The
preserved prefix should grow monotonically over a trajectory (spec §2.2's
"key property to verify and highlight" — see `bench/prefix_growth.py`... no
such helper is needed: it falls out directly from `frozen` only ever growing).

Statefulness note: unlike `NaivePolicy` (fully stateless — it re-derives
"anchor" and "rest" from the raw `turns` argument every call), this policy
keeps its own `ContextState` across calls and treats `turns[-1]` as *the one
new raw turn appended since last call*. This is safe under the harness
contract every experiment script in this repo uses (`phase1_cliff.py`'s
`run_trajectory`): each step computes
`turns_before_compaction = context_turns_from_last_call + [next_raw_turn]`,
so `turns[:-1]` is always exactly what this policy itself returned last time.
A fresh instance must be created per trajectory (already the established
pattern — see `NaivePolicy` usage in `experiments/phase1_cliff.py`).
"""
from __future__ import annotations

from agentkv.bench.replay import Turn
from agentkv.context.layout import Tokenizer, render_turns_to_token_ids
from agentkv.context.segments import AnchorHygieneGuard, ContextState, Segment
from agentkv.policies.base import CompactionPolicy
from agentkv.policies.naive import Summarizer


class AppendOnlyPolicy(CompactionPolicy):
    name = "append_only"

    def __init__(
        self,
        tokenizer: Tokenizer,
        summarizer: Summarizer,
        *,
        threshold_tokens: int,
        retire_fraction: float = 0.5,
    ) -> None:
        if not 0.0 < retire_fraction < 1.0:
            raise ValueError(f"retire_fraction must be in (0, 1), got {retire_fraction}")
        self._tokenizer = tokenizer
        self._summarizer = summarizer
        self._threshold_tokens = threshold_tokens
        self._retire_fraction = retire_fraction
        self._state: ContextState | None = None
        self._anchor_guard = AnchorHygieneGuard(tokenizer)

    @property
    def state(self) -> ContextState:
        """Exposed read-only for experiment scripts that want to inspect
        `frozen`/`live` directly (e.g. to verify the monotonic-prefix-growth
        property spec §2.2 calls out)."""
        assert self._state is not None, "maybe_compact has not been called yet"
        return self._state

    def maybe_compact(self, turns: list[Turn], step_idx: int) -> tuple[list[Turn], bool]:
        if self._state is None:
            # First call: `turns` is just `[anchor]` under the harness
            # contract (see module docstring) — turns[0] is the anchor
            # (spec §2.4), same convention `NaivePolicy` uses.
            self._state = ContextState(anchor=turns[0], live=list(turns[1:]))
        else:
            self._state.live.append(turns[-1])

        # Checks the *raw incoming* turns[0] every step, not `self._state.anchor`
        # (which never changes once set) — the point is to detect an upstream
        # bug that starts handing us a different anchor, not to re-verify our
        # own copy of the one we already locked in.
        self._anchor_guard.check(turns[0], step_idx=step_idx)

        prompt_len = len(render_turns_to_token_ids(self._state.to_turns(), self._tokenizer))
        if prompt_len < self._threshold_tokens or len(self._state.live) < 2:
            return self._state.to_turns(), False

        cutoff = max(1, int(len(self._state.live) * self._retire_fraction))
        retired, kept = self._state.live[:cutoff], self._state.live[cutoff:]

        summary_text = self._summarizer.summarize(retired)
        self._state.frozen.append(
            Segment(
                step_created=step_idx,
                summary_text=summary_text,
                n_turns_summarized=len(retired),
            )
        )
        self._state.live = kept
        return self._state.to_turns(), True
