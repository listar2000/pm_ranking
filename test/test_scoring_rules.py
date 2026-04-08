"""Unit tests for ``pm_rank.model.scoring_rule``.

Exercises Brier, Log, and Spherical scoring rules on the synthetic challenge
fixture. No raw data needed.

Note on API: ``ScoringRule.fit`` always returns a tuple — ``(rankings,)`` when
``include_scores=False`` and ``(scores, rankings)`` when it is True — so tests
unpack accordingly.
"""
import numpy as np

from pm_rank.model import BrierScoringRule, LogScoringRule, SphericalScoringRule


def test_brier_fit_returns_ranking_for_all_forecasters(synthetic_challenge):
    rule = BrierScoringRule()
    (rankings,) = rule.fit(synthetic_challenge.forecast_problems, include_scores=False)
    assert set(rankings.keys()) == {"alice", "bob", "carol"}


def test_brier_fit_returns_scores_when_requested(synthetic_challenge):
    rule = BrierScoringRule()
    result = rule.fit(synthetic_challenge.forecast_problems, include_scores=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    scores, rankings = result
    assert set(scores.keys()) == set(rankings.keys())
    assert all(np.isfinite(s) for s in scores.values())


def test_brier_ranks_strongest_forecaster_first_on_single_problem(synthetic_problem):
    """On problem 1 (correct = option 0), alice has probs [0.7, 0.2, 0.1] — best."""
    rule = BrierScoringRule()
    scores, rankings = rule.fit([synthetic_problem], include_scores=True)
    # higher brier (with negate=True) = better
    assert scores["alice"] > scores["bob"]
    assert scores["alice"] > scores["carol"]
    assert rankings["alice"] == 1


def test_log_scoring_rule_fit(synthetic_challenge):
    rule = LogScoringRule()
    (rankings,) = rule.fit(synthetic_challenge.forecast_problems, include_scores=False)
    assert set(rankings.keys()) == {"alice", "bob", "carol"}


def test_spherical_scoring_rule_fit(synthetic_challenge):
    rule = SphericalScoringRule()
    (rankings,) = rule.fit(synthetic_challenge.forecast_problems, include_scores=False)
    assert set(rankings.keys()) == {"alice", "bob", "carol"}


def test_brier_with_problem_discriminations_differs_from_unweighted(synthetic_challenge):
    rule = BrierScoringRule()
    problems = synthetic_challenge.forecast_problems
    unweighted_scores, _ = rule.fit(problems, include_scores=True)
    # Give problem 1 much more weight than problem 2.
    weighted_scores, _ = rule.fit(
        problems,
        include_scores=True,
        problem_discriminations=[5.0, 0.1],
    )
    # At least one forecaster should see a different aggregated score.
    diffs = [abs(unweighted_scores[user] - weighted_scores[user]) for user in unweighted_scores]
    assert max(diffs) > 1e-9
