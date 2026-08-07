from __future__ import annotations

import pytest

from agentkv.bench.divergence import (
    TheoryMismatchError,
    assert_theory_matches,
    build_compaction_event_record,
    compute_divergence_check,
    first_divergence_index,
)

BLOCK_SIZE = 16


def test_first_divergence_index_identical_lists():
    ids = list(range(50))
    assert first_divergence_index(ids, ids) == 50


def test_first_divergence_index_pure_append():
    prev = list(range(50))
    new = list(range(70))  # prev is an exact prefix of new
    assert first_divergence_index(prev, new) == 50


def test_first_divergence_index_mid_rewrite():
    prev = list(range(50))
    new = list(range(50))
    new[20] = 999
    assert first_divergence_index(prev, new) == 20


def test_divergence_check_pure_append_predicts_zero_invalidation():
    prev = list(range(48))  # 3 full blocks at block_size=16
    new = list(range(64))  # +1 block appended, nothing rewritten
    check = compute_divergence_check(
        step_idx=1,
        prev_prompt_ids=prev,
        new_prompt_ids=new,
        measured_cached_tokens=48,  # all 3 prior blocks reused
        block_size=BLOCK_SIZE,
    )
    assert check.predicted_invalidated_blocks == 0
    assert check.theory_matches_measurement
    assert_theory_matches(check)  # must not raise


def test_divergence_check_full_rewrite_predicts_full_invalidation():
    prev = list(range(64))  # 4 blocks
    new = [999] * 64  # divergence at token 0
    check = compute_divergence_check(
        step_idx=5,
        prev_prompt_ids=prev,
        new_prompt_ids=new,
        measured_cached_tokens=0,
        block_size=BLOCK_SIZE,
    )
    assert check.divergence_token_idx == 0
    assert check.predicted_invalidated_blocks == 4
    assert check.theory_matches_measurement


def test_divergence_check_partial_rewrite_matches_block_math():
    prev = list(range(80))  # 5 blocks
    new = list(range(40)) + [999] * 40  # divergence at token 40 -> 2 blocks still valid
    check = compute_divergence_check(
        step_idx=2,
        prev_prompt_ids=prev,
        new_prompt_ids=new,
        measured_cached_tokens=32,  # 2 blocks x 16
        block_size=BLOCK_SIZE,
    )
    assert check.divergence_token_idx == 40
    assert check.predicted_invalidated_blocks == 3  # 5 - 2
    assert check.theory_matches_measurement


def test_assert_theory_matches_raises_on_mismatch():
    prev = list(range(80))
    new = list(range(40)) + [999] * 40
    check = compute_divergence_check(
        step_idx=2,
        prev_prompt_ids=prev,
        new_prompt_ids=new,
        measured_cached_tokens=0,  # wrong: theory predicts 32 (off by 2 blocks)
        block_size=BLOCK_SIZE,
    )
    assert not check.theory_matches_measurement
    with pytest.raises(TheoryMismatchError):
        assert_theory_matches(check)


def test_assert_theory_matches_tolerates_one_block_of_slack():
    """A pure-append step that completes a previously almost-full trailing
    partial block can measure one block "too many" reused (vLLM 0.8.5
    credits the whole completed block) — a real, bounded discrepancy, not a
    cache-model bug, so it must not raise."""
    prev = list(range(63))  # 3 full blocks (48 tokens) + a 15-token partial block
    new = list(range(70))  # pure append, completes the partial block + more
    check = compute_divergence_check(
        step_idx=26,
        prev_prompt_ids=prev,
        new_prompt_ids=new,
        measured_cached_tokens=64,  # 4 blocks reused, one more than the 3 "theory" allows
        block_size=BLOCK_SIZE,
    )
    assert not check.theory_matches_measurement
    assert_theory_matches(check)  # must not raise: within the 1-block tolerance


def test_assert_theory_matches_allows_unbounded_extra_reuse():
    """The opposite direction of the mismatch above — measured invalidation
    *below* prediction, i.e. the engine reused *more* than the pairwise
    prev-vs-new comparison predicts — must never raise, however large. This
    model only compares against the single immediately-previous prompt, but
    vLLM's real prefix cache is a pool across the engine's whole request
    history; content can legitimately be reused from further back (e.g. an
    LLM summarizer quoting retired-turn text verbatim into a summary that's
    still resident in the cache from a recent, non-adjacent request —
    reproduced on Phase 2's traj-000/naive step 139). Gate 2's real
    prefill/token accounting comes from vLLM's own measured values, not this
    prediction, so extra reuse here is a legitimate bonus, not a bug."""
    prev = list(range(80))  # 5 blocks
    new = list(range(40)) + [999] * 40  # divergence at token 40 -> predicted 3 invalidated
    check = compute_divergence_check(
        step_idx=139,
        prev_prompt_ids=prev,
        new_prompt_ids=new,
        measured_cached_tokens=80,  # all 5 blocks reused despite divergence -> 0 invalidated
        block_size=BLOCK_SIZE,
    )
    assert check.predicted_invalidated_blocks == 3
    assert check.measured_invalidated_blocks == 0
    assert not check.theory_matches_measurement
    assert_theory_matches(check)  # must not raise: extra reuse, not a shortfall


def test_build_compaction_event_record_computes_reprefill_ratio():
    check = compute_divergence_check(
        step_idx=10,
        prev_prompt_ids=list(range(80)),
        new_prompt_ids=list(range(40)) + [999] * 10,
        measured_cached_tokens=32,
        block_size=BLOCK_SIZE,
    )
    record = build_compaction_event_record(
        step_idx=10,
        policy="naive",
        seed=0,
        check=check,
        tokens_before_compaction=200,
        tokens_after_compaction=50,
        tokens_reprefilled=18,
        ttft_ms=123.4,
    )
    assert record.tokens_saved_by_compacting == 150
    assert record.tokens_reprefilled == 18
    assert record.reprefill_to_saved_ratio == pytest.approx(18 / 150)
    assert record.policy == "naive"
