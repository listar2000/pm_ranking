"""Unit tests for the pydantic data models in ``pm_rank.data.base``.

All tests run against synthetic in-memory data. No raw data files required.
"""
from datetime import datetime

import pytest
from pydantic import ValidationError

from pm_rank.data import ForecastChallenge, ForecastEvent, ForecastProblem
from pm_rank.data.base import SMOOTH_ODDS_EPS


_TS = datetime(2024, 1, 1, 12, 0, 0)
_SENTINEL = object()


def _event(fid="f1", problem_id="p1", user="alice", probs=_SENTINEL, **kwargs):
    if probs is _SENTINEL:
        probs = [0.5, 0.5]
    return ForecastEvent(
        forecast_id=fid,
        problem_id=problem_id,
        username=user,
        timestamp=_TS,
        probs=probs,
        **kwargs,
    )


# --- ForecastEvent ----------------------------------------------------------


def test_forecast_event_probs_sum_to_one_accepted():
    ev = _event(probs=[0.3, 0.3, 0.4])
    assert ev.probs == [0.3, 0.3, 0.4]


def test_forecast_event_probs_not_summing_raises():
    with pytest.raises(ValidationError):
        _event(probs=[0.5, 0.4])


def test_forecast_event_negative_prob_raises():
    with pytest.raises(ValidationError):
        _event(probs=[-0.1, 1.1])


def test_forecast_event_empty_probs_raises():
    with pytest.raises(ValidationError):
        _event(probs=[])


def test_forecast_event_negative_weight_raises():
    with pytest.raises(ValidationError):
        _event(probs=[0.5, 0.5], weight=-0.1)


def test_forecast_event_unnormalized_probs_default_to_probs():
    ev = _event(probs=[0.2, 0.8])
    assert ev.unnormalized_probs == [0.2, 0.8]


def test_forecast_event_odds_smoothed_away_from_zero_and_one():
    ev = _event(probs=[0.5, 0.5], odds=[0.0, 1.0])
    assert ev.odds is not None
    assert ev.odds[0] == pytest.approx(SMOOTH_ODDS_EPS)
    assert ev.odds[1] == pytest.approx(1 - SMOOTH_ODDS_EPS)


def test_forecast_event_odds_length_mismatch_raises():
    with pytest.raises(ValidationError):
        _event(probs=[0.5, 0.5], odds=[0.5, 0.3, 0.2])


def test_forecast_event_odds_out_of_range_raises():
    with pytest.raises(ValidationError):
        _event(probs=[0.5, 0.5], odds=[0.5, 1.5])


# --- ForecastProblem --------------------------------------------------------


def _problem(**overrides):
    defaults = dict(
        title="Test Problem",
        problem_id="p1",
        options=["A", "B"],
        correct_option_idx=[0],
        forecasts=[
            _event(fid="f1", user="alice", probs=[0.7, 0.3]),
            _event(fid="f2", user="bob", probs=[0.4, 0.6]),
        ],
        end_time=datetime(2024, 1, 2),
        num_forecasters=2,
    )
    defaults.update(overrides)
    return ForecastProblem(**defaults)


def test_forecast_problem_constructs():
    problem = _problem()
    assert len(problem.forecasts) == 2
    assert problem.num_forecasters == 2


def test_forecast_problem_correct_idx_out_of_range_raises():
    with pytest.raises(ValidationError):
        _problem(correct_option_idx=[5])


def test_forecast_problem_correct_idx_duplicate_raises():
    with pytest.raises(ValidationError):
        _problem(options=["A", "B", "C"], correct_option_idx=[1, 1])


def test_forecast_problem_forecast_probs_length_mismatch_raises():
    bad_forecast = _event(fid="f99", user="eve", probs=[0.1, 0.2, 0.7])
    good_forecast = _event(fid="f1", user="alice", probs=[0.6, 0.4])
    with pytest.raises(ValidationError):
        _problem(forecasts=[good_forecast, bad_forecast])


def test_forecast_problem_duplicate_forecast_ids_raises():
    a = _event(fid="same", user="alice", probs=[0.5, 0.5])
    b = _event(fid="same", user="bob", probs=[0.5, 0.5])
    with pytest.raises(ValidationError):
        _problem(forecasts=[a, b])


def test_forecast_problem_crowd_probs_averages_forecasts():
    problem = _problem(
        forecasts=[
            _event(fid="f1", user="alice", probs=[0.8, 0.2]),
            _event(fid="f2", user="bob", probs=[0.4, 0.6]),
        ]
    )
    assert problem.crowd_probs[0] == pytest.approx(0.6)
    assert problem.crowd_probs[1] == pytest.approx(0.4)


def test_forecast_problem_has_odds_reflects_presence():
    with_odds = _problem(
        forecasts=[
            _event(fid="f1", user="alice", probs=[0.5, 0.5], odds=[0.5, 0.5]),
            _event(fid="f2", user="bob", probs=[0.5, 0.5], odds=[0.5, 0.5]),
        ]
    )
    without_odds = _problem()
    assert with_odds.has_odds is True
    assert without_odds.has_odds is False


def test_forecast_problem_unique_forecasters():
    problem = _problem(
        forecasts=[
            _event(fid="f1", user="alice", probs=[0.5, 0.5]),
            _event(fid="f2", user="alice", probs=[0.6, 0.4]),
            _event(fid="f3", user="bob", probs=[0.3, 0.7]),
        ]
    )
    assert set(problem.unique_forecasters) == {"alice", "bob"}


# --- ForecastChallenge ------------------------------------------------------


def _challenge():
    p1 = _problem(problem_id="p1")
    p2 = _problem(
        problem_id="p2",
        forecasts=[
            _event(fid="f3", problem_id="p2", user="alice", probs=[0.2, 0.8]),
            _event(fid="f4", problem_id="p2", user="bob", probs=[0.6, 0.4]),
        ],
    )
    return ForecastChallenge(title="challenge", forecast_problems=[p1, p2])


def test_forecast_challenge_duplicate_problem_ids_raises():
    p = _problem(problem_id="p1")
    with pytest.raises(ValidationError):
        ForecastChallenge(title="dup", forecast_problems=[p, p])


def test_forecast_challenge_forecaster_map_has_all_users():
    challenge = _challenge()
    assert set(challenge.forecaster_map.keys()) == {"alice", "bob"}
    assert len(challenge.forecaster_map["alice"]) == 2
    assert len(challenge.forecaster_map["bob"]) == 2


def test_forecast_challenge_num_forecasters():
    challenge = _challenge()
    assert challenge.num_forecasters == 2


def test_forecast_challenge_get_problem_by_id():
    challenge = _challenge()
    problem = challenge.get_problem_by_id("p2")
    assert problem is not None
    assert problem.problem_id == "p2"
    assert challenge.get_problem_by_id("nonexistent") is None


def test_forecast_challenge_get_forecaster_problems():
    challenge = _challenge()
    alice_problems = challenge.get_forecaster_problems("alice")
    assert len(alice_problems) == 2


def test_forecast_challenge_stream_problems_sequential():
    challenge = _challenge()
    batches = list(challenge.stream_problems(order="sequential", increment=1))
    assert len(batches) == 2
    assert len(batches[0]) == 1
    assert len(batches[1]) == 1


def test_forecast_challenge_get_problems_default_returns_all():
    challenge = _challenge()
    assert len(challenge.get_problems()) == 2
    assert len(challenge.get_problems(nums=1)) == 1
