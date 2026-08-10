from __future__ import annotations

import pytest

from agentkv.bench.stats import wilcoxon_signed_rank, wilson_confidence_interval


def test_identical_samples_are_not_significant():
    a = [100.0, 200.0, 300.0, 150.0, 250.0]
    result = wilcoxon_signed_rank(a, list(a))
    assert result.n_nonzero == 0
    assert result.p_value == 1.0
    assert result.significant_at_05 is False


def test_consistent_large_one_sided_difference_is_significant():
    # every pair has b much smaller than a, with n large enough for the
    # normal approximation to be meaningful (spec §0.2: >= 5 seeds; this uses 12).
    a = [1000.0 + i for i in range(12)]
    b = [400.0 + i for i in range(12)]
    result = wilcoxon_signed_rank(a, b)
    assert result.n_nonzero == 12
    assert result.significant_at_05 is True
    assert result.p_value < 0.01


def test_mixed_signs_roughly_balanced_is_not_significant():
    a = [100.0, 102.0, 98.0, 101.0, 99.0, 103.0]
    b = [101.0, 100.0, 99.0, 100.0, 100.0, 102.0]  # differences flip sign, small magnitude
    result = wilcoxon_signed_rank(a, b)
    assert result.significant_at_05 is False


def test_raises_on_mismatched_lengths():
    import pytest

    with pytest.raises(ValueError):
        wilcoxon_signed_rank([1.0, 2.0], [1.0])


def test_ties_get_average_rank():
    # two pairs share the same absolute difference magnitude (5) with opposite
    # signs, one pair with a bigger, unambiguous difference.
    a = [10.0, 20.0, 100.0]
    b = [15.0, 15.0, 20.0]  # diffs: -5, +5, +80
    result = wilcoxon_signed_rank(a, b)
    assert result.n_nonzero == 3
    # tied ranks (magnitude 5) each get rank 1.5; the larger difference gets rank 3.
    # w_pos = ranks{1.5, 3} = 4.5, w_neg = rank{1.5} = 1.5, w = min(...) = 1.5
    assert result.w_statistic == 1.5


def test_wilson_ci_zero_of_five_matches_known_value():
    # Cross-checked against results/phase3/phase3_summary.md's reported
    # "0/5 (95% Wilson CI: 0%-43%)" figure.
    ci = wilson_confidence_interval(0, 5)
    assert ci.proportion == 0.0
    assert ci.low == pytest.approx(0.0, abs=1e-3)
    assert ci.high == pytest.approx(0.434, abs=1e-3)


def test_wilson_ci_all_successes_stays_within_bounds():
    ci = wilson_confidence_interval(5, 5)
    assert ci.proportion == 1.0
    assert 0.0 <= ci.low <= 1.0
    assert ci.high == pytest.approx(1.0, abs=1e-9)


def test_wilson_ci_widens_with_smaller_n():
    narrow = wilson_confidence_interval(3, 20)
    wide = wilson_confidence_interval(3, 5)
    assert (wide.high - wide.low) > (narrow.high - narrow.low)


def test_wilson_ci_raises_on_invalid_inputs():
    with pytest.raises(ValueError):
        wilson_confidence_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_confidence_interval(6, 5)
