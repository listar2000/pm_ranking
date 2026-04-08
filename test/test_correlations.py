"""Unit tests for the ranking correlation helpers in ``pm_rank.model.utils``."""
from pm_rank.model import kendall_correlation, spearman_correlation


def test_spearman_identical_rankings_is_one():
    a = {"alice": 1, "bob": 2, "carol": 3}
    assert spearman_correlation(a, a) == 1.0


def test_spearman_reversed_rankings_is_minus_one():
    a = {"alice": 1, "bob": 2, "carol": 3}
    b = {"alice": 3, "bob": 2, "carol": 1}
    assert spearman_correlation(a, b) == -1.0


def test_spearman_bounded():
    a = {"alice": 1, "bob": 2, "carol": 3, "dave": 4}
    b = {"alice": 2, "bob": 4, "carol": 1, "dave": 3}
    r = spearman_correlation(a, b)
    assert -1.0 <= r <= 1.0


def test_kendall_identical_rankings_is_one():
    a = {"alice": 1, "bob": 2, "carol": 3}
    assert kendall_correlation(a, a) == 1.0


def test_kendall_reversed_rankings_is_minus_one():
    a = {"alice": 1, "bob": 2, "carol": 3}
    b = {"alice": 3, "bob": 2, "carol": 1}
    assert kendall_correlation(a, b) == -1.0


def test_kendall_bounded():
    a = {"alice": 1, "bob": 2, "carol": 3, "dave": 4}
    b = {"alice": 2, "bob": 4, "carol": 1, "dave": 3}
    k = kendall_correlation(a, b)
    assert -1.0 <= k <= 1.0


def test_correlations_with_intersecting_keys_only():
    """Only forecasters present in both dicts should be compared."""
    a = {"alice": 1, "bob": 2, "carol": 3}
    b = {"alice": 1, "bob": 2, "dave": 3}  # "carol" and "dave" are disjoint
    # intersection is {alice, bob}, both identically ranked → correlation 1.0
    assert spearman_correlation(a, b) == 1.0
    assert kendall_correlation(a, b) == 1.0
