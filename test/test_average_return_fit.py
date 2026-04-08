"""Unit tests for ``AverageReturn.fit`` using the synthetic challenge fixture."""
import math

import pytest

from pm_rank.model import AverageReturn, AverageReturnConfig


def test_average_return_fit_returns_ranking_for_all_forecasters(synthetic_challenge):
    ar = AverageReturn()
    (rankings,) = ar.fit(synthetic_challenge.forecast_problems, include_scores=False)
    assert set(rankings.keys()) == {"alice", "bob", "carol"}


def test_average_return_fit_scores_are_finite(synthetic_challenge):
    ar = AverageReturn()
    scores, rankings = ar.fit(synthetic_challenge.forecast_problems, include_scores=True)
    assert set(scores.keys()) == set(rankings.keys())
    for value in scores.values():
        assert math.isfinite(value)


def test_average_return_config_rejects_invalid_risk_aversion():
    with pytest.raises(ValueError):
        AverageReturnConfig(risk_aversion=1.5)
    with pytest.raises(ValueError):
        AverageReturnConfig(risk_aversion=-0.1)


def test_average_return_config_default_risk_aversion_is_zero():
    config = AverageReturnConfig()
    assert config.risk_aversion == 0.0
