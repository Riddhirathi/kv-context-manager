from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentkv.bench.replay import TrajectoryReplayer, iter_trajectories, load_trajectory


def write_trajectory(path: Path, turns: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for turn in turns:
            f.write(json.dumps(turn) + "\n")


def test_load_trajectory_roundtrips_turns(tmp_path: Path):
    path = tmp_path / "traj-001.jsonl"
    write_trajectory(
        path,
        [
            {"role": "system", "content": "you are an agent"},
            {"role": "user", "content": "do the task"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"name": "read_file", "args": {"path": "a.txt"}}],
            },
            {
                "role": "tool",
                "tool_results": [{"name": "read_file", "output": "file contents"}],
            },
        ],
    )
    trajectory = load_trajectory(path)
    assert trajectory.trajectory_id == "traj-001"
    assert len(trajectory) == 4
    assert trajectory.turns[0].role == "system"
    assert trajectory.turns[2].tool_calls == [{"name": "read_file", "args": {"path": "a.txt"}}]


def test_load_trajectory_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "traj-002.jsonl"
    path.write_text(
        json.dumps({"role": "user", "content": "hi"}) + "\n\n" +
        json.dumps({"role": "assistant", "content": "hello"}) + "\n"
    )
    trajectory = load_trajectory(path)
    assert len(trajectory) == 2


def test_load_trajectory_rejects_empty_file(tmp_path: Path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError):
        load_trajectory(path)


def test_iter_trajectories_sorted_and_deterministic(tmp_path: Path):
    write_trajectory(tmp_path / "b.jsonl", [{"role": "user", "content": "b"}])
    write_trajectory(tmp_path / "a.jsonl", [{"role": "user", "content": "a"}])
    ids = [t.trajectory_id for t in iter_trajectories(tmp_path)]
    assert ids == ["a", "b"]


def test_replayer_steps_yield_growing_verbatim_prefix(tmp_path: Path):
    path = tmp_path / "traj-003.jsonl"
    write_trajectory(
        path,
        [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
        ],
    )
    trajectory = load_trajectory(path)
    replayer = TrajectoryReplayer(trajectory)
    assert len(replayer) == 3

    steps = list(replayer.steps())
    assert [idx for idx, _ in steps] == [0, 1, 2]
    assert [len(ctx) for _, ctx in steps] == [1, 2, 3]
    # Prefix must be verbatim and stable across steps (no rewriting).
    assert steps[2][1][0] == steps[0][1][0]
