"""Unit tests for ``pm_rank.model.bradley_terry.GeneralizedBT``.

Runs the MM algorithm for a small number of iterations against the synthetic
challenge fixture and against hand-constructed dominance / symmetric cases
where the semantic ordering of the resulting scores can be predicted.

Note: ``fit(include_scores=False)`` returns a single-element tuple
``(rankings,)``, matching the scoring-rule convention.
"""
import math
from datetime import datetime

from pm_rank.data import ForecastEvent, ForecastProblem
from pm_rank.model import GeneralizedBT


def _event(fid, pid, user, probs):
    return ForecastEvent(
        forecast_id=fid,
        problem_id=pid,
        username=user,
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        probs=probs,
    )


def _two_player_problem(pid, alice_probs, bob_probs, correct_option_idx=(0,)):
    return ForecastProblem(
        title=pid,
        problem_id=pid,
        options=["A", "B"],
        correct_option_idx=list(correct_option_idx),
        forecasts=[
            _event(f"a_{pid}", pid, "alice", alice_probs),
            _event(f"b_{pid}", pid, "bob", bob_probs),
        ],
        end_time=datetime(2024, 1, 2),
        num_forecasters=2,
    )


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


# --- Numerical / semantic correctness -------------------------------------


def test_generalized_bt_dominance_orders_correctly():
    """If alice always puts high prob on the correct outcome and bob always
    puts high prob on the wrong one, alice's BT skill must exceed bob's and
    she must rank #1 after fitting.
    """
    problems = [
        _two_player_problem("p1", [0.9, 0.1], [0.1, 0.9]),
        _two_player_problem("p2", [0.8, 0.2], [0.2, 0.8]),
        _two_player_problem("p3", [0.95, 0.05], [0.05, 0.95]),
    ]
    scores, rankings = GeneralizedBT(method="MM", num_iter=200).fit(problems, include_scores=True)
    assert scores["alice"] > scores["bob"]
    assert rankings["alice"] == 1
    assert rankings["bob"] == 2


def test_generalized_bt_symmetric_identical_forecasters_tie():
    """When every forecaster makes identical predictions, their BT skill
    parameters should converge to the same value."""
    problems = [
        _two_player_problem("p1", [0.6, 0.4], [0.6, 0.4]),
        _two_player_problem("p2", [0.3, 0.7], [0.3, 0.7], correct_option_idx=(1,)),
    ]
    scores, _ = GeneralizedBT(method="MM", num_iter=500).fit(problems, include_scores=True)
    assert scores["alice"] == scores["bob"]
