from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from demo_deck import fired_events_frame  # noqa: E402


def test_fired_events_frame_flags_only_invalidating_steps():
    # step 0: no prior context. step 1: pure append, nothing invalidated.
    # step 2: a compaction — prev prompt was 320 tokens (20 blocks), only 32
    # tokens (2 blocks) reused.
    run = pd.DataFrame(
        {
            "step_idx": [0, 1, 2],
            "prompt_tokens": [58, 116, 80],
            "cached_tokens": [0, 58, 32],
        }
    )
    events = fired_events_frame(run, block_size=16)
    assert events["step_idx"].tolist() == [2]


def test_fired_events_frame_no_events_is_empty():
    run = pd.DataFrame(
        {"step_idx": [0, 1, 2], "prompt_tokens": [58, 116, 180], "cached_tokens": [0, 58, 112]}
    )
    events = fired_events_frame(run, block_size=16)
    assert events.empty
    assert list(events.columns) == ["step_idx"]
