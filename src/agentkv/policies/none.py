"""No-compaction baseline (AGENTKV_SPEC.md §4.5's Gate 4 needs "≥5 policies" on
the Pareto plot; §4.2 attn_evict was proven infeasible on this hardware and
never built, and §4.3 offload is an orthogonal serving-layer overlay, not a
distinct content policy on the same prefill-token axis — see
`results/phase4/phase4_pareto_summary.md` for that decision).

This is the ceiling every other policy is implicitly measured against: never
compact, ever. Maximum prefill cost (nothing is ever dropped or rewritten, so
every step's prompt is the full uncompacted history), and — if the model can
complete the task at all — the reference point for what task success looks
like with zero information loss from compaction. A real, honestly-labeled
data point, not a placeholder.
"""
from __future__ import annotations

from agentkv.bench.replay import Turn
from agentkv.policies.base import CompactionPolicy


class NoOpPolicy(CompactionPolicy):
    name = "none"

    def maybe_compact(self, turns: list[Turn], step_idx: int) -> tuple[list[Turn], bool]:
        return turns, False
