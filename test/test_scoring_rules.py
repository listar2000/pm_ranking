"""Unit tests for ``pm_rank.model.scoring_rule``.

Exercises Brier, Log, and Spherical scoring rules on the synthetic challenge
fixture and on minimal hand-constructed cases where the expected value can
be computed from the documented formula.

Formulas being verified:

- Brier (negate=True): ``1 - sum((p_i - y_i)^2) / d`` where ``y`` is the
  one-hot of the correct option(s) and ``d`` is the number of options.
- Log: ``log(max(p_correct, clip_prob))``.
- Spherical: ``sum(p_correct) / ||p||_2``.

Note on API: ``ScoringRule.fit`` always returns a tuple — ``(rankings,)`` when
``include_scores=False`` and ``(scores, rankings)`` when it is True — so tests
unpack accordingly. For a single-problem single-forecaster case, the aggregated
score (mean) equals the per-problem score, which is what we assert against.
"""
import math
from datetime import datetime

import numpy as np
import pytest

from pm_rank.data import ForecastEvent, ForecastProblem
from pm_rank.model import BrierScoringRule, LogScoringRule, SphericalScoringRule


# --- Helpers ---------------------------------------------------------------


def _single_forecast_problem(probs, correct_option_idx, *, problem_id="p1", options=None):
    """Build a 1-forecaster, 1-problem ForecastProblem with the given probs."""
    if options is None:
        options = [chr(ord("A") + i) for i in range(len(probs))]
    event = ForecastEvent(
        forecast_id=f"f_{problem_id}",
        problem_id=problem_id,
        username="alice",
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        probs=probs,
    )
    return ForecastProblem(
        title=problem_id,
        problem_id=problem_id,
        options=options,
        correct_option_idx=correct_option_idx,
        forecasts=[event],
        end_time=datetime(2024, 1, 2),
        num_forecasters=1,
    )


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


# --- Numerical correctness (exact expected values) ------------------------


def test_brier_exact_value_three_options_negated():
    """Brier on probs=[0.7, 0.2, 0.1], correct=[0], 3 options.

    Expected = 1 - ((0.3)^2 + (0.2)^2 + (0.1)^2) / 3
             = 1 - 0.14 / 3
    """
    problem = _single_forecast_problem([0.7, 0.2, 0.1], correct_option_idx=[0])
    scores, _ = BrierScoringRule().fit([problem], include_scores=True)
    assert scores["alice"] == pytest.approx(1 - 0.14 / 3)


def test_brier_exact_value_non_negated():
    """With negate=False, the raw rescaled Brier (lower = better) is returned."""
    problem = _single_forecast_problem([0.7, 0.2, 0.1], correct_option_idx=[0])
    scores, _ = BrierScoringRule(negate=False).fit([problem], include_scores=True)
    assert scores["alice"] == pytest.approx(0.14 / 3)


def test_brier_exact_value_binary_correct_prediction():
    """probs=[1.0, 0.0] correct=[0] ⇒ perfect Brier = 1 (with negate=True)."""
    problem = _single_forecast_problem([1.0, 0.0], correct_option_idx=[0])
    scores, _ = BrierScoringRule().fit([problem], include_scores=True)
    assert scores["alice"] == pytest.approx(1.0)


def test_brier_exact_value_binary_uniform_guess():
    """probs=[0.5, 0.5] correct=[0], d=2 ⇒ Brier = 1 - 0.5/2 = 0.75."""
    problem = _single_forecast_problem([0.5, 0.5], correct_option_idx=[0])
    scores, _ = BrierScoringRule().fit([problem], include_scores=True)
    assert scores["alice"] == pytest.approx(0.75)


def test_log_exact_value_three_options():
    """Log score = log(p_correct). For probs=[0.7,...], correct=[0] ⇒ log(0.7)."""
    problem = _single_forecast_problem([0.7, 0.2, 0.1], correct_option_idx=[0])
    scores, _ = LogScoringRule().fit([problem], include_scores=True)
    assert scores["alice"] == pytest.approx(math.log(0.7))


def test_log_clip_prob_floor_applies():
    """With clip_prob=0.01, a correct-option prob of 0.001 is clipped to 0.01."""
    problem = _single_forecast_problem([0.001, 0.999], correct_option_idx=[0])
    scores, _ = LogScoringRule(clip_prob=0.01).fit([problem], include_scores=True)
    assert scores["alice"] == pytest.approx(math.log(0.01))


def test_spherical_exact_value_three_options():
    """Spherical = p_correct / ||p||. probs=[0.7,0.2,0.1] correct=[0] ⇒ 0.7/sqrt(0.54)."""
    problem = _single_forecast_problem([0.7, 0.2, 0.1], correct_option_idx=[0])
    scores, _ = SphericalScoringRule().fit([problem], include_scores=True)
    assert scores["alice"] == pytest.approx(0.7 / math.sqrt(0.54))


def test_spherical_exact_value_binary_uniform_guess():
    """probs=[0.5, 0.5] correct=[0] ⇒ 0.5 / sqrt(0.5) = 1/sqrt(2)."""
    problem = _single_forecast_problem([0.5, 0.5], correct_option_idx=[0])
    scores, _ = SphericalScoringRule().fit([problem], include_scores=True)
    assert scores["alice"] == pytest.approx(1.0 / math.sqrt(2.0))


def test_brier_mean_aggregation_over_multiple_problems():
    """The reported score is the mean of per-problem Brier scores.

    P1: probs=[0.8, 0.2] correct=[0] ⇒ 1 - (0.04 + 0.04)/2 = 0.96
    P2: probs=[0.3, 0.7] correct=[0] ⇒ 1 - (0.49 + 0.49)/2 = 0.51
    Mean ⇒ (0.96 + 0.51) / 2 = 0.735
    """
    p1 = _single_forecast_problem([0.8, 0.2], correct_option_idx=[0], problem_id="p1")
    p2 = _single_forecast_problem([0.3, 0.7], correct_option_idx=[0], problem_id="p2")
    scores, _ = BrierScoringRule().fit([p1, p2], include_scores=True)
    assert scores["alice"] == pytest.approx((0.96 + 0.51) / 2)


def test_brier_uniform_discriminations_equal_unweighted():
    """Uniform problem_discriminations must leave the per-forecaster score unchanged."""
    p1 = _single_forecast_problem([0.8, 0.2], correct_option_idx=[0], problem_id="p1")
    p2 = _single_forecast_problem([0.3, 0.7], correct_option_idx=[0], problem_id="p2")
    rule = BrierScoringRule()
    unweighted, _ = rule.fit([p1, p2], include_scores=True)
    weighted, _ = rule.fit([p1, p2], include_scores=True, problem_discriminations=[1.0, 1.0])
    assert weighted["alice"] == pytest.approx(unweighted["alice"])
