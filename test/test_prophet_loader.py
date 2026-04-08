"""Tests for ``ProphetArenaChallengeLoader``.

Loads the Prophet Arena CSV from ``src/pm_rank/data/raw``. Auto-skipped if the
file is absent via the ``requires_prophet_data`` marker (see conftest.py).
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from pm_rank.data import ForecastChallenge, ProphetArenaChallengeLoader

pytestmark = pytest.mark.requires_prophet_data

_RAW_DIR = Path(__file__).resolve().parent.parent / "src" / "pm_rank" / "data" / "raw"
ARENA_PREDICTIONS_FILE = _RAW_DIR / "prophet_arena_full.csv"

CHALLENGE_TITLE = "Prophet Arena Example"


@pytest.fixture(scope="module")
def prophet_challenge() -> ForecastChallenge:
    loader = ProphetArenaChallengeLoader(
        predictions_file=str(ARENA_PREDICTIONS_FILE),
        challenge_title=CHALLENGE_TITLE,
    )
    return loader.load_challenge()


def test_prophet_arena_loader_basic():
    loader = ProphetArenaChallengeLoader(
        predictions_file=str(ARENA_PREDICTIONS_FILE),
        challenge_title=CHALLENGE_TITLE,
    )
    metadata = loader.get_challenge_metadata()
    assert metadata["title"] == CHALLENGE_TITLE
    assert metadata["num_problems"] > 0
    assert metadata["num_forecasters"] > 0
    assert "predictions_file" in metadata


def test_prophet_arena_loader_full_challenge(prophet_challenge: ForecastChallenge):
    assert prophet_challenge.title == CHALLENGE_TITLE
    assert len(prophet_challenge.forecast_problems) > 0
    for problem in prophet_challenge.forecast_problems:
        assert problem.title is not None
        assert len(problem.options) > 0
        assert len(problem.forecasts) > 0
        if problem.has_odds:
            all_odds = [forecast.odds for forecast in problem.forecasts]
            assert all(abs(sum(odds) - 1.0) < 0.1 for odds in all_odds)


def test_prophet_arena_odds_calculation():
    df = pd.read_csv(ARENA_PREDICTIONS_FILE)
    first_row = df.iloc[0]
    options = json.loads(first_row["markets"]) if isinstance(first_row["markets"], str) else first_row["markets"]
    market_info = json.loads(first_row["market_info"]) if isinstance(first_row["market_info"], str) else first_row["market_info"]

    odds = ProphetArenaChallengeLoader._calculate_implied_probs_for_problem(market_info, options)

    assert isinstance(odds, list)
    assert len(odds) == len(options)
    if sum(odds) > 0:
        assert abs(sum(odds) - 1.0) < 0.1


def test_prophet_arena_stream_problems_over_time(prophet_challenge: ForecastChallenge):
    streamed_problem = 0
    for bucket in prophet_challenge.stream_problems_over_time(increment_by="day", min_bucket_size=10):
        streamed_problem += len(bucket[1])

    assert streamed_problem == len(prophet_challenge.forecast_problems)


def test_prophet_arena_average_return_fit_stream_with_timestamp(prophet_challenge: ForecastChallenge):
    from pm_rank.model.average_return import AverageReturn

    average_return = AverageReturn(verbose=True)
    average_return.fit_stream_with_timestamp(
        prophet_challenge.stream_problems_over_time(increment_by="day", min_bucket_size=10)
    )
