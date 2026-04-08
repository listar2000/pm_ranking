"""Unit tests for ``pm_rank.model.bradley_terry.GeneralizedBT``.

Runs the MM algorithm for a small number of iterations against the synthetic
challenge fixture. Note: ``fit(include_scores=False)`` returns a single-element
tuple ``(rankings,)``, matching the scoring-rule convention.
"""
import math

from pm_rank.model import GeneralizedBT


def test_generalized_bt_fit_returns_ranking_for_all_forecasters(synthetic_challenge):
    bt = GeneralizedBT(method="MM", num_iter=50)
    (rankings,) = bt.fit(synthetic_challenge.forecast_problems, include_scores=False)
    assert set(rankings.keys()) == {"alice", "bob", "carol"}


def test_generalized_bt_returns_finite_scores(synthetic_challenge):
    bt = GeneralizedBT(method="MM", num_iter=50)
    scores, rankings = bt.fit(synthetic_challenge.forecast_problems, include_scores=True)
    assert set(scores.keys()) == set(rankings.keys())
    for value in scores.values():
        assert math.isfinite(value)


def test_generalized_bt_ranks_are_unique(synthetic_challenge):
    bt = GeneralizedBT(method="MM", num_iter=100)
    (rankings,) = bt.fit(synthetic_challenge.forecast_problems, include_scores=False)
    assert sorted(rankings.values()) == [1, 2, 3]
