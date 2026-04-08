"""Unit tests for ``AverageReturn.fit`` using the synthetic challenge fixture.

Also includes a hand-constructed 2-player case where the earnings of the
"good" forecaster are predictable: on a fair binary market with 50/50 odds,
a risk-neutral all-in bet with forecast [0.9, 0.1] on the correct outcome
yields a gross payout of exactly 2.0 per $1 staked, while the opposite
forecast earns 0.
"""
import math
from datetime import datetime

import pytest

from pm_rank.data import ForecastEvent, ForecastProblem
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


# --- Numerical / semantic correctness ------------------------------------


def _mk_event(fid, pid, user, probs, odds, no_odds):
    return ForecastEvent(
        forecast_id=fid,
        problem_id=pid,
        username=user,
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        probs=probs,
        odds=odds,
        no_odds=no_odds,
    )


def test_average_return_risk_neutral_confident_winner_earns_two():
    """Fair 50/50 binary market, correct outcome is option 0.

    Alice goes all-in on option 0 with forecast [0.9, 0.1]. Under the
    risk-neutral CRRA strategy, the $1 stake buys ``1 / 0.5 = 2`` "yes"
    contracts on option 0, which pay out 2.0 when the market resolves to 0.

    Bob takes the opposite position and earns 0.
    """
    problem = ForecastProblem(
        title="t",
        problem_id="p1",
        options=["A", "B"],
        correct_option_idx=[0],
        forecasts=[
            _mk_event("fa", "p1", "alice", [0.9, 0.1], [0.5, 0.5], [0.5, 0.5]),
            _mk_event("fb", "p1", "bob", [0.1, 0.9], [0.5, 0.5], [0.5, 0.5]),
        ],
        end_time=datetime(2024, 1, 2),
        num_forecasters=2,
    )

    scores, rankings = AverageReturn().fit([problem], include_scores=True)
    assert scores["alice"] == pytest.approx(2.0)
    assert scores["bob"] == pytest.approx(0.0)
    assert rankings["alice"] == 1
    assert rankings["bob"] == 2
