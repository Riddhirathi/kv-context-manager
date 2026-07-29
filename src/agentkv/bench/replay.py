"""Deterministic trajectory replay (AGENTKV_SPEC.md §0.4).

Trajectories are recorded once against a strong model via a free API tier and
committed as JSONL under trajectories/ (one file per trajectory, one Turn per
line). Replaying them locally against the small model decouples "is the agent
good?" (irrelevant) from "what does compaction cost?" (the actual question).
This module never makes a network call — recording is a separate, one-time step.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class Turn(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_results: list[dict[str, Any]] | None = None


class Trajectory(BaseModel):
    trajectory_id: str
    turns: list[Turn]

    def __len__(self) -> int:
        return len(self.turns)


def load_trajectory(path: Path) -> Trajectory:
    """Loads one trajectory from a JSONL file: one Turn per line."""
    turns: list[Turn] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            turns.append(Turn.model_validate(json.loads(stripped)))
    if not turns:
        raise ValueError(f"{path} contains no turns.")
    return Trajectory(trajectory_id=path.stem, turns=turns)


def iter_trajectories(directory: Path) -> Iterator[Trajectory]:
    """Loads every *.jsonl trajectory in `directory`, sorted by filename for determinism."""
    for path in sorted(directory.glob("*.jsonl")):
        yield load_trajectory(path)


class TrajectoryReplayer:
    """Replays a trajectory verbatim, step by step.

    Phase 0 applies no compaction — this exists to prove the harness itself is
    reproducible (Gate 0) before any compaction policy sits on top of it in
    Phase 1+. Deliberately has no dependency on the serving engine: wiring this
    to `serving.engine.VLLMEngine` + `metrics.collector` happens in the
    experiment script that runs the actual Gate 0 check.
    """

    def __init__(self, trajectory: Trajectory) -> None:
        self._trajectory = trajectory

    def __len__(self) -> int:
        return len(self._trajectory)

    def steps(self) -> Iterator[tuple[int, list[Turn]]]:
        """Yields (step_idx, context_so_far) — the verbatim prefix through step_idx."""
        for step_idx in range(len(self._trajectory.turns)):
            yield step_idx, self._trajectory.turns[: step_idx + 1]
