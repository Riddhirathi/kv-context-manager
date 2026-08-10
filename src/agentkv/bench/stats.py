"""Paired statistical significance test (AGENTKV_SPEC.md §7: "Every claimed
improvement needs a paired statistical test across seeds.").

Implemented directly, no scipy dependency (see pyproject.toml — this repo
deliberately keeps a small dependency set), as a two-sided Wilcoxon
signed-rank test with the standard normal approximation and average-rank tie
handling. Zero-differences are dropped before ranking (a "tie" means the
policy made no difference for that pair, which carries no directional
evidence either way) — the conventional handling, appropriate once
`n_nonzero` is moderately large, which every Phase 1/2 sweep in this repo
satisfies (spec §0.2: >= 5 seeds; Phase 1's full sweep used 14 trajectories).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PairedTestResult:
    n_pairs: int
    n_nonzero: int
    w_statistic: float
    z_statistic: float
    p_value: float

    @property
    def significant_at_05(self) -> bool:
        return self.p_value < 0.05


def wilcoxon_signed_rank(a: list[float], b: list[float]) -> PairedTestResult:
    """Two-sided Wilcoxon signed-rank test on paired samples `a` vs `b` (e.g.
    per-trajectory total prefill tokens under naive vs. append_only, same
    trajectory, same seed, paired by index)."""
    if len(a) != len(b):
        raise ValueError("a and b must be the same length (paired samples).")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    if n == 0:
        return PairedTestResult(
            n_pairs=len(a), n_nonzero=0, w_statistic=0.0, z_statistic=0.0, p_value=1.0
        )

    order = sorted(range(n), key=lambda i: abs(nonzero[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nonzero[order[j + 1]]) == abs(nonzero[order[i]]):
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-indexed average rank across the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    w_pos = sum(r for r, d in zip(ranks, nonzero, strict=True) if d > 0)
    w_neg = sum(r for r, d in zip(ranks, nonzero, strict=True) if d < 0)
    w = min(w_pos, w_neg)

    mean_w = n * (n + 1) / 4
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if std_w == 0:
        z = 0.0
    else:
        correction = 0.5 if w < mean_w else -0.5
        z = (w - mean_w + correction) / std_w
    p_value = min(1.0, 2 * (1 - _standard_normal_cdf(abs(z))))
    return PairedTestResult(
        n_pairs=len(a), n_nonzero=n, w_statistic=w, z_statistic=z, p_value=p_value
    )


def _standard_normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


@dataclass(frozen=True)
class WilsonInterval:
    proportion: float
    low: float
    high: float


def wilson_confidence_interval(successes: int, n: int, *, z: float = 1.96) -> WilsonInterval:
    """Wilson score interval for a binomial proportion (default z=1.96, 95%
    CI) — used for task-success rate error bars (spec §7: "report median and
    IQR" / "every claimed improvement needs a ... test", and Gate 4's "a
    Pareto plot exists ... and you can defend every point on it").

    Preferred over the naive `p +/- z*sqrt(p(1-p)/n)` normal-approximation
    interval because it stays inside [0, 1] and is well-behaved at the
    extremes (p=0 or p=1) that small-n task-success sweeps in this project
    routinely hit — e.g. Phase 3.3's 0/5 naive/append_only result.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes}")
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half_width = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return WilsonInterval(
        proportion=p, low=max(0.0, center - half_width), high=min(1.0, center + half_width)
    )
