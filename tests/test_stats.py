from __future__ import annotations

from agentkv.bench.stats import wilcoxon_signed_rank


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
