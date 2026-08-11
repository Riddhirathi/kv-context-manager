from __future__ import annotations

import pandas as pd

from agentkv.viz.dashboard import (
    GREEN,
    RED,
    block_segments,
    compress_segments,
    load_agreement,
    split_into_runs,
)


def test_block_segments_pure_append_has_no_red():
    # 100 tokens cached out of a previous 100-token prompt, now 116 tokens —
    # nothing invalidated, new tail appended.
    segments = block_segments(
        prev_prompt_tokens=100, prompt_tokens=116, cached_tokens=100, block_size=16
    )
    assert all(color == GREEN for color, _ in segments)
    # prev_cacheable = 100 // 16 = 6 (reused); ceil(116 / 16) = 8 total.
    assert sum(n for _, n in segments) == 8


def test_block_segments_compaction_event_shows_red_tail():
    # Previous prompt had 320 tokens (20 blocks); after compaction only 2 of
    # those blocks were reused, then a much shorter new prompt (80 tokens)
    # was built from the summary — the classic naive-style big invalidation.
    segments = block_segments(
        prev_prompt_tokens=320, prompt_tokens=80, cached_tokens=32, block_size=16
    )
    assert segments[0] == (GREEN, 2)
    assert segments[1] == (RED, 18)  # 20 previously-cacheable minus 2 reused
    assert len(segments) == 2  # no new-tail segment: 80 tokens (5 blocks) < prev 320


def test_block_segments_never_reports_more_reused_than_previously_cacheable():
    # Guards the min() clamp: a measurement quirk (or bad input) reporting
    # cached_tokens >= prev_prompt_tokens must not produce negative invalidation.
    segments = block_segments(
        prev_prompt_tokens=160, prompt_tokens=200, cached_tokens=200, block_size=16
    )
    assert all(color == GREEN for color, _ in segments)


def test_block_segments_first_step_has_no_prior_context():
    segments = block_segments(
        prev_prompt_tokens=0, prompt_tokens=58, cached_tokens=0, block_size=16
    )
    assert segments == [(GREEN, 4)]  # ceil(58/16) = 4, none of it "reused" yet


def test_compress_segments_noop_under_max_cells():
    segments = [(GREEN, 5), (RED, 3)]
    assert compress_segments(segments, max_cells=48) == segments


def test_compress_segments_never_hides_red():
    # A single red block buried inside a huge green run must still show up
    # after compression — this is the "sells the project in three seconds"
    # visual, so a real compaction event can never disappear.
    segments = [(GREEN, 500), (RED, 1), (GREEN, 499)]
    compressed = compress_segments(segments, max_cells=20)
    assert sum(n for _, n in compressed) == 20
    assert any(color == RED for color, _ in compressed)


def test_compress_segments_empty_input():
    assert compress_segments([], max_cells=48) == []


def test_split_into_runs_detects_step_idx_zero_boundaries():
    steps = pd.DataFrame(
        {
            "policy": ["naive"] * 5 + ["hybrid"] * 3,
            "step_idx": [0, 1, 2, 0, 1, 0, 1, 2],
            "prefill_tokens": [10, 20, 30, 40, 50, 60, 70, 80],
        }
    )
    naive_runs = split_into_runs(steps, "naive")
    assert len(naive_runs) == 2
    assert naive_runs[0]["step_idx"].tolist() == [0, 1, 2]
    assert naive_runs[1]["step_idx"].tolist() == [0, 1]

    hybrid_runs = split_into_runs(steps, "hybrid")
    assert len(hybrid_runs) == 1
    assert hybrid_runs[0]["step_idx"].tolist() == [0, 1, 2]


def test_split_into_runs_unknown_policy_returns_empty():
    steps = pd.DataFrame({"policy": ["naive"], "step_idx": [0], "prefill_tokens": [10]})
    assert split_into_runs(steps, "kv_evict") == []


def test_load_agreement_missing_file_returns_empty(tmp_path):
    assert load_agreement(tmp_path / "does_not_exist.parquet", "hybrid") == {}


def test_load_agreement_reads_matching_policy_only(tmp_path):
    path = tmp_path / "agreement.parquet"
    frame = pd.DataFrame(
        {
            "step_idx": [2, 5, 2],
            "policy": ["naive", "naive", "hybrid"],
            "tool_name_match": [True, False, True],
            "args_match": [True, False, False],
        }
    )
    frame.to_parquet(path, index=False)

    naive = load_agreement(path, "naive")
    assert naive == {2: (True, True), 5: (False, False)}

    hybrid = load_agreement(path, "hybrid")
    assert hybrid == {2: (True, False)}

    assert load_agreement(path, "kv_evict") == {}
