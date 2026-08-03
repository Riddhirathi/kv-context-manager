"""CompactionPolicy ABC (AGENTKV_SPEC.md §10: "every policy implements the same
CompactionPolicy ABC. Adding a policy must require touching exactly one new file.").
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from agentkv.bench.replay import Turn


class CompactionPolicy(ABC):
    """Decides, at each agent step, whether to rewrite the context.

    `maybe_compact` receives the full turn sequence *as it would be sent this
    step* (the previous step's context, possibly already compacted, plus the
    one new raw turn just appended) and returns the turn sequence to actually
    send, plus whether a compaction event fired this step. Divergence-point
    and re-prefill accounting (spec §1.3) is computed generically outside any
    policy, by diffing this step's rendered token ids against the previous
    step's — it does not depend on which policy produced the turns.
    """

    name: ClassVar[str]

    @abstractmethod
    def maybe_compact(self, turns: list[Turn], step_idx: int) -> tuple[list[Turn], bool]:
        """Returns (turns to send this step, whether compaction fired)."""
        raise NotImplementedError
