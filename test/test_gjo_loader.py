"""Tests for ``GJOChallengeLoader``.

These tests load the Good Judgment Open (GJO) sports-challenge raw data from
``src/pm_rank/data/raw``. The entire module is auto-skipped by ``conftest.py``
when those files are absent (see the ``requires_gjo_data`` marker).
"""
from pathlib import Path

import pytest

from pm_rank.data import ForecastChallenge, ForecastEvent, ForecastProblem, GJOChallengeLoader

pytestmark = pytest.mark.requires_gjo_data

_RAW_DIR = Path(__file__).resolve().parent.parent / "src" / "pm_rank" / "data" / "raw"
GJO_PREDICTIONS = _RAW_DIR / "all_predictions.json"
GJO_METADATA = _RAW_DIR / "sports_challenge_metadata.json"

CHALLENGE_TITLE = "Sports Challenge 2024"


def _make_loader() -> GJOChallengeLoader:
    return GJOChallengeLoader(
        predictions_file=str(GJO_PREDICTIONS),
        metadata_file=str(GJO_METADATA),
        challenge_title=CHALLENGE_TITLE,
    )


def test_gjo_loader_basic():
    loader = _make_loader()
    metadata = loader.get_challenge_metadata()
    assert metadata["title"] == CHALLENGE_TITLE
    assert metadata["num_problems"] > 0
    assert "predictions_file" in metadata
    assert "metadata_file" in metadata


def test_gjo_loader_full_challenge():
    loader = _make_loader()
    challenge = loader.load_challenge()

    assert isinstance(challenge, ForecastChallenge)
    assert challenge.title == CHALLENGE_TITLE
    assert len(challenge.forecast_problems) > 0
    assert challenge.num_forecasters > 0

    first_problem = challenge.forecast_problems[0]
    assert isinstance(first_problem, ForecastProblem)
    assert first_problem.title is not None
    assert first_problem.problem_id > 0
    assert len(first_problem.options) > 0
    assert first_problem.correct_option_idx[0] < len(first_problem.options)
    assert len(first_problem.forecasts) > 0

    first_forecast = first_problem.forecasts[0]
    assert isinstance(first_forecast, ForecastEvent)
    assert first_forecast.username is not None
    assert len(first_forecast.probs) == len(first_problem.options)
    assert abs(sum(first_forecast.probs) - 1.0) < 1e-6


def test_gjo_loader_with_filters():
    loader = _make_loader()
    challenge = loader.load_challenge(forecaster_filter=2, problem_filter=5)

    assert isinstance(challenge, ForecastChallenge)
    assert len(challenge.forecast_problems) > 0

    for problem in challenge.forecast_problems:
        assert len(problem.forecasts) >= 5

    forecaster_problem_counts: dict = {}
    for problem in challenge.forecast_problems:
        for forecast in problem.forecasts:
            forecaster_problem_counts.setdefault(forecast.username, set()).add(problem.problem_id)

    for problem_ids in forecaster_problem_counts.values():
        assert len(problem_ids) >= 2


def test_challenge_properties():
    loader = _make_loader()
    challenge = loader.load_challenge()

    assert len(challenge.forecaster_map) == challenge.num_forecasters
    assert len(challenge.unique_forecasters) == challenge.num_forecasters

    if challenge.forecast_problems:
        first_problem_id = challenge.forecast_problems[0].problem_id
        found_problem = challenge.get_problem_by_id(first_problem_id)
        assert found_problem is not None
        assert found_problem.problem_id == first_problem_id

    if challenge.unique_forecasters:
        first_forecaster = challenge.unique_forecasters[0]
        forecaster_problems = challenge.get_forecaster_problems(first_forecaster)
        assert len(forecaster_problems) > 0


def test_problem_properties():
    loader = _make_loader()
    challenge = loader.load_challenge()

    if challenge.forecast_problems:
        problem = challenge.forecast_problems[0]

        crowd_probs = problem.crowd_probs
        assert len(crowd_probs) == len(problem.options)
        assert abs(sum(crowd_probs) - 1.0) < 1e-6

        unique_forecasters = problem.unique_forecasters
        assert len(unique_forecasters) <= len(problem.forecasts)

        assert problem.has_odds == (problem.forecasts[0].odds is not None)
