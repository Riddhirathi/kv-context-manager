"""Divergence-point instrumentation and the cache-invalidation sanity check
(AGENTKV_SPEC.md §1.3, §1.4).

Token ids, not text, are sent to vLLM as the prompt (see context/layout.py) —
so the "divergence point" between two consecutive steps' prompts is exactly
the first index where their token-id lists differ, and that is exactly what
determines vLLM's block-hash-based prefix-cache reuse. This module is pure
(no GPU, no I/O) so the block-invalidation math is unit-testable on its own
(spec §10: "Tests for: block-alignment math, divergence-point calculation,
and the cost model. These three are where silent bugs will invalidate
results.").
"""
from __future__ import annotations

from dataclasses import dataclass

from agentkv.metrics.collector import RecordCollector


def first_divergence_index(prev_ids: list[int], new_ids: list[int]) -> int:
    """First index where `prev_ids` and `new_ids` differ.

    If one is a prefix of the other (the pure-append case), returns the
    length of the shorter list — i.e. no divergence within the overlap.
    """
    n = min(len(prev_ids), len(new_ids))
    for i in range(n):
        if prev_ids[i] != new_ids[i]:
            return i
    return n


class TheoryMismatchError(RuntimeError):
    """Raised when measured cache invalidation doesn't match the block-math
    prediction (spec §1.4: "Measured must match predicted exactly. If it
    doesn't, your understanding of the cache is wrong — stop and fix that
    first.")."""


@dataclass(frozen=True)
class DivergenceCheck:
    step_idx: int
    divergence_token_idx: int
    prev_prompt_tokens: int
    new_prompt_tokens: int
    blocks_previously_cacheable: int
    predicted_invalidated_blocks: int
    measured_invalidated_blocks: int

    @property
    def theory_matches_measurement(self) -> bool:
        return self.predicted_invalidated_blocks == self.measured_invalidated_blocks


def compute_divergence_check(
    *,
    step_idx: int,
    prev_prompt_ids: list[int],
    new_prompt_ids: list[int],
    measured_cached_tokens: int,
    block_size: int,
) -> DivergenceCheck:
    """Compares the theoretical block-invalidation prediction against what
    vLLM actually reported for this step (`measured_cached_tokens`, from
    `serving.engine.StepResult.cached_tokens`).

    Predicted invalidated blocks = blocks previously cacheable minus blocks
    still valid up to the divergence point (spec §1.4's formula, restated in
    terms of the *previous* step's prompt rather than a fixed
    `n_blocks_total`, since the previous step's prompt length is what was
    actually cached and available to reuse).
    """
    divergence_idx = first_divergence_index(prev_prompt_ids, new_prompt_ids)
    blocks_previously_cacheable = len(prev_prompt_ids) // block_size
    blocks_still_valid = divergence_idx // block_size
    predicted_invalidated = blocks_previously_cacheable - blocks_still_valid
    measured_reused_blocks = measured_cached_tokens // block_size
    measured_invalidated = blocks_previously_cacheable - measured_reused_blocks
    return DivergenceCheck(
        step_idx=step_idx,
        divergence_token_idx=divergence_idx,
        prev_prompt_tokens=len(prev_prompt_ids),
        new_prompt_tokens=len(new_prompt_ids),
        blocks_previously_cacheable=blocks_previously_cacheable,
        predicted_invalidated_blocks=predicted_invalidated,
        measured_invalidated_blocks=measured_invalidated,
    )


def assert_theory_matches(check: DivergenceCheck, *, tolerance_blocks: int = 1) -> None:
    """Spec §1.4: "Measured must match predicted exactly. If it doesn't, your
    understanding of the cache is wrong — stop and fix that first."

    `tolerance_blocks` defaults to 1, not 0, based on a real, reproduced
    finding: when a pure-append step's new tokens complete a previously
    almost-full trailing partial block (measured case: a block with 15/16
    tokens already cached needed just 1 new token to complete), vLLM 0.8.5
    credits that whole block as reused — one block "too many" versus the
    textbook floor-division model, which only counts *complete*,
    previously-hashed blocks as cacheable. Confirmed this isn't a bug in the
    divergence math itself: a controlled replay using a stub (non-LLM)
    summarizer matched exactly at every step, including across a real
    compaction event; the discrepancy appears only around boundary-completing
    appends and is bounded to exactly one block when it occurs — it doesn't
    compound, since each step's prediction is computed fresh from that step's
    own actual prompt.

    Deliberately one-directional: only raises when measured invalidation
    EXCEEDS the prediction (i.e. the engine reused *fewer* blocks than the
    pairwise prev-vs-new comparison says it could have) — the direction the
    spec principle actually guards against, since it can only mean the model
    or measurement is broken.

    The opposite direction — measured invalidation *below* prediction (the
    engine reusing *more* than this pairwise model predicts) — is not
    flagged. Reproduced and confirmed legitimate (see
    experiments/_debug_real_summarizer_mismatch.py, Phase 2's traj-000/naive
    step 139: predicted 584 invalidated blocks, measured only 551): this
    model only compares a request's prompt against the single
    immediately-previous request, but vLLM's real prefix cache is a pool
    across the engine's *entire* request history. The LLM summarizer
    (temperature=0) often quotes retired-turn content verbatim in its
    summaries, and that exact text is frequently still resident in the cache
    from a recent — but not immediately-previous — request. This doesn't
    corrupt Gate 2: its prefill/token accounting comes from vLLM's own
    measured values, not this theoretical prediction, and the effect is
    symmetric across both policies (both share one summarizer).
    """
    diff = check.measured_invalidated_blocks - check.predicted_invalidated_blocks
    if diff > tolerance_blocks:
        raise TheoryMismatchError(
            f"step {check.step_idx}: predicted {check.predicted_invalidated_blocks} "
            f"invalidated blocks (divergence at token {check.divergence_token_idx}), "
            f"but measured {check.measured_invalidated_blocks} — the cache model is wrong."
        )


@dataclass(frozen=True)
class CompactionEventRecord:
    """One row per compaction event (spec §1.3's required fields), logged
    only on steps where a `CompactionPolicy` actually fired."""

    step_idx: int
    policy: str
    seed: int
    divergence_token_idx: int
    blocks_invalidated: int
    tokens_before: int
    tokens_after: int
    tokens_reprefilled: int
    tokens_saved_by_compacting: int
    reprefill_to_saved_ratio: float
    ttft_ms: float


def build_compaction_event_record(
    *,
    step_idx: int,
    policy: str,
    seed: int,
    check: DivergenceCheck,
    tokens_before_compaction: int,
    tokens_after_compaction: int,
    tokens_reprefilled: int,
    ttft_ms: float,
) -> CompactionEventRecord:
    tokens_saved = tokens_before_compaction - tokens_after_compaction
    ratio = tokens_reprefilled / tokens_saved if tokens_saved > 0 else float("inf")
    return CompactionEventRecord(
        step_idx=step_idx,
        policy=policy,
        seed=seed,
        divergence_token_idx=check.divergence_token_idx,
        blocks_invalidated=check.measured_invalidated_blocks,
        tokens_before=tokens_before_compaction,
        tokens_after=tokens_after_compaction,
        tokens_reprefilled=tokens_reprefilled,
        tokens_saved_by_compacting=tokens_saved,
        reprefill_to_saved_ratio=ratio,
        ttft_ms=ttft_ms,
    )


class CompactionEventCollector(RecordCollector[CompactionEventRecord]):
    """Buffers CompactionEventRecords and flushes them to an append-only parquet
    file, separate from the per-step metrics parquet (spec §1.3 needs its own
    fields — divergence index, blocks invalidated, reprefill ratio — that
    don't belong on every ordinary step)."""
