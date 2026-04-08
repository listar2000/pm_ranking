"""Unit tests for ``pm_rank.model.calibration.CalibrationMetric``.

Tests use ``weight_event=False`` because the ``weight_event=True`` code path
currently normalizes weights to sum to 1.0 while ``_bin_stats`` asserts the
weights sum equals ``len(probs)``, which does not hold for any forecaster that
participates in more than one event. Fixing that normalization is out of scope
for this testing setup.
"""
import pytest

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
