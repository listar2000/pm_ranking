"""Unit tests for ``pm_rank.model.calibration.CalibrationMetric``.

Covers structural behaviour on the synthetic challenge fixture plus two
hand-constructed boundary cases:

* A perfectly-calibrated forecaster whose per-bin confidence equals the
  per-bin empirical accuracy — expected ECE = 0.
* A maximally-miscalibrated forecaster who places all mass on the wrong
  option with prob 0.99 — expected ECE = 0.99.

Tests use ``weight_event=False`` because the ``weight_event=True`` code path
currently normalizes weights to sum to 1.0 while ``_bin_stats`` asserts the
weights sum equals ``len(probs)``, which does not hold for any forecaster that
participates in more than one event. Fixing that normalization is out of scope
for this testing setup.
"""
from datetime import datetime

import pytest

from pm_rank.data import ForecastEvent, ForecastProblem
from pm_rank.model import CalibrationMetric


def test_calibration_metric_fit_returns_score_per_forecaster(synthetic_challenge):
    metric = CalibrationMetric(num_bins=5, strategy="uniform", weight_event=False)
    scores, rankings = metric.fit(synthetic_challenge.forecast_problems, include_scores=True)
    assert set(scores.keys()) == {"alice", "bob", "carol"}
    assert set(rankings.keys()) == {"alice", "bob", "carol"}
    for value in scores.values():
        assert 0.0 <= value <= 1.0


def test_calibration_metric_fit_rankings_only(synthetic_challenge):
    metric = CalibrationMetric(num_bins=5, weight_event=False)
    rankings = metric.fit(synthetic_challenge.forecast_problems, include_scores=False)
    assert set(rankings.keys()) == {"alice", "bob", "carol"}


@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_calibration_metric_strategies(synthetic_challenge, strategy):
    metric = CalibrationMetric(num_bins=4, strategy=strategy, weight_event=False)
    scores, _ = metric.fit(synthetic_challenge.forecast_problems, include_scores=True)
    assert set(scores.keys()) == {"alice", "bob", "carol"}


def test_calibration_metric_num_bins_must_be_greater_than_one():
    with pytest.raises(AssertionError):
        CalibrationMetric(num_bins=1)


# --- Numerical correctness -----------------------------------------------


def _single_forecaster_problem(pid, probs, correct_option_idx):
    event = ForecastEvent(
        forecast_id=f"f_{pid}",
        problem_id=pid,
        username="alice",
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        probs=probs,
    )
    return ForecastProblem(
        title=pid,
        problem_id=pid,
        options=["A", "B"],
        correct_option_idx=[correct_option_idx],
        forecasts=[event],
        end_time=datetime(2024, 1, 2),
        num_forecasters=1,
    )


def test_calibration_metric_perfectly_calibrated_has_zero_ece():
    """10 binary problems with alice always predicting [0.7, 0.3].
    7 problems resolve to option 0, 3 to option 1, so the empirical frequency
    exactly matches the predicted probability in every non-empty bin.
    Expected ECE = 0 (up to floating-point noise).
    """
    problems = [_single_forecaster_problem(f"p{i}", [0.7, 0.3], 0) for i in range(7)]
    problems += [_single_forecaster_problem(f"p{i}", [0.7, 0.3], 1) for i in range(7, 10)]

    scores, _ = CalibrationMetric(num_bins=5, strategy="uniform", weight_event=False).fit(
        problems, include_scores=True
    )
    assert scores["alice"] == pytest.approx(0.0, abs=1e-9)


def test_calibration_metric_maximally_miscalibrated_has_ece_near_one():
    """Alice always predicts [0.99, 0.01] but option 1 is always correct.

    Bin containing 0.99: 10 samples, all labelled 0 ⇒ |0.99 - 0.0| = 0.99
    Bin containing 0.01: 10 samples, all labelled 1 ⇒ |0.01 - 1.0| = 0.99
    Each bin has weight 10/20 ⇒ ECE = 0.99.
    """
    problems = [_single_forecaster_problem(f"p{i}", [0.99, 0.01], 1) for i in range(10)]
    scores, _ = CalibrationMetric(num_bins=5, strategy="uniform", weight_event=False).fit(
        problems, include_scores=True
    )
    assert scores["alice"] == pytest.approx(0.99)
