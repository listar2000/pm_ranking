#!/usr/bin/env python3
"""
Simple test for ProphetArenaChallengeLoader using assert-based testing.
"""
import json
import pandas as pd
from pm_rank.data import ProphetArenaChallengeLoader, ForecastChallenge
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

ARENA_PREDICTIONS_FILE = "src/pm_rank/data/raw/prophet_arena_full.csv"


def test_prophet_arena_loader_basic():
    """Test basic functionality of ProphetArenaChallengeLoader."""
    loader = ProphetArenaChallengeLoader(
        predictions_file=ARENA_PREDICTIONS_FILE,
        challenge_title="Prophet Arena Example"
    )
    metadata = loader.get_challenge_metadata()
    assert metadata['title'] == "Prophet Arena Example"
    assert metadata['num_problems'] > 0
    assert metadata['num_forecasters'] > 0
    assert 'predictions_file' in metadata
    print("✓ ProphetArena metadata loading test passed")


def test_prophet_arena_loader_full_challenge():
    """Test loading the full Prophet Arena challenge."""
    loader = ProphetArenaChallengeLoader(
        predictions_file=ARENA_PREDICTIONS_FILE,
        challenge_title="Prophet Arena Example"
    )
    challenge = loader.load_challenge()
    assert challenge.title == "Prophet Arena Example"
    assert len(challenge.forecast_problems) > 0
    for problem in challenge.forecast_problems:
        assert problem.title is not None
        assert len(problem.options) > 0
        assert len(problem.forecasts) > 0
        # Odds should be present and sum to ~1
        if problem.odds:
            assert abs(sum(problem.odds) - 1.0) < 0.1
    print("✓ ProphetArena full challenge loading test passed")
    return challenge


def test_prophet_arena_odds_calculation(challenge: ForecastChallenge):
    """Test the odds calculation helper for Prophet Arena."""
    df = pd.read_csv(ARENA_PREDICTIONS_FILE)
    first_row = df.iloc[0]
    options = json.loads(first_row['markets']) if isinstance(
        first_row['markets'], str) else first_row['markets']
    market_info = json.loads(first_row['market_info']) if isinstance(
        first_row['market_info'], str) else first_row['market_info']
    odds = ProphetArenaChallengeLoader._calculate_implied_probs_for_problem(
        market_info, options)
    assert isinstance(odds, list)
    assert len(odds) == len(options)
    if sum(odds) > 0:
        assert abs(sum(odds) - 1.0) < 0.1
    print("✓ ProphetArena odds calculation test passed")


def test_prophet_arena_stream_problems_over_time(challenge: ForecastChallenge):
    """Test the stream_problems_over_time method."""
    streamed_problem = 0

    for bucket in challenge.stream_problems_over_time(increment_by="day", min_bucket_size=10):
        print(f"Bucket: {bucket[0]} has {len(bucket[1])} problems")
        streamed_problem += len(bucket[1])

    assert streamed_problem == len(challenge.forecast_problems)
    print("✓ ProphetArena stream_problems_over_time test passed")


def test_prophet_arena_average_return_fit_stream_with_timestamp(challenge: ForecastChallenge):
    """Test the market earning fit_stream_with_timestamp method."""
    from pm_rank.model.average_return import AverageReturn
    average_return = AverageReturn(verbose=True)
    average_return.fit_stream_with_timestamp(
        challenge.stream_problems_over_time(increment_by="day", min_bucket_size=10))
    print("✓ ProphetArena market earning fit_stream_with_timestamp test passed")


def run_all_tests():
    """Run all tests."""
    print("Running ProphetArenaChallengeLoader tests...")
    print("=" * 50)

    try:
        test_prophet_arena_loader_basic()
        challenge = test_prophet_arena_loader_full_challenge()
        test_prophet_arena_odds_calculation(challenge)
        test_prophet_arena_stream_problems_over_time(challenge)
        test_prophet_arena_average_return_fit_stream_with_timestamp(challenge)
        print("=" * 50)
        print("✓ All tests passed!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    run_all_tests()
