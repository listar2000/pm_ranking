#!/usr/bin/env python3
"""
Simple test for GJOChallengeLoader using assert-based testing.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from pm_rank.data import GJOChallengeLoader, ForecastChallenge, ForecastProblem, ForecastEvent

PREDICTIONS_FILE = "src/pm_rank/data/raw/all_predictions.json"
METADATA_FILE = "src/pm_rank/data/raw/sports_challenge_metadata.json"

def test_gjo_loader_basic():
    """Test basic functionality of GJOChallengeLoader."""
    
    # Initialize the loader with the actual data files
    loader = GJOChallengeLoader(
        predictions_file=PREDICTIONS_FILE,
        metadata_file=METADATA_FILE,
        challenge_title="Sports Challenge 2024"
    )
    
    # Test metadata loading
    metadata = loader.get_challenge_metadata()
    assert metadata['title'] == "Sports Challenge 2024"
    assert metadata['num_problems'] > 0
    assert 'predictions_file' in metadata
    assert 'metadata_file' in metadata
    
    print("✓ Metadata loading test passed")


def test_gjo_loader_full_challenge():
    """Test loading the full challenge without filters."""
    
    loader = GJOChallengeLoader(
        predictions_file=PREDICTIONS_FILE,
        metadata_file=METADATA_FILE,
        challenge_title="Sports Challenge 2024"
    )
    
    # Load the challenge
    challenge = loader.load_challenge()
    
    # Basic assertions
    assert isinstance(challenge, ForecastChallenge)
    assert challenge.title == "Sports Challenge 2024"
    assert len(challenge.forecast_problems) > 0
    assert challenge.num_forecasters > 0
    
    # Test that we have problems
    assert len(challenge.forecast_problems) > 0
    
    # Test the first problem
    first_problem = challenge.forecast_problems[0]
    assert isinstance(first_problem, ForecastProblem)
    assert first_problem.title is not None
    assert first_problem.problem_id > 0
    assert len(first_problem.options) > 0
    assert first_problem.correct_option_idx[0] < len(first_problem.options)
    assert len(first_problem.forecasts) > 0
    
    # Test that forecasts are properly structured
    first_forecast = first_problem.forecasts[0]
    assert isinstance(first_forecast, ForecastEvent)
    assert first_forecast.username is not None
    assert len(first_forecast.probs) == len(first_problem.options)
    assert abs(sum(first_forecast.probs) - 1.0) < 1e-6  # Probabilities sum to 1
    
    print("✓ Full challenge loading test passed")


def test_gjo_loader_with_filters():
    """Test loading with forecaster and problem filters."""
    
    loader = GJOChallengeLoader(
        predictions_file=PREDICTIONS_FILE,
        metadata_file=METADATA_FILE,
        challenge_title="Sports Challenge 2024"
    )
    
    # Load with filters (require at least 5 forecasts per problem and 2 problems per forecaster)
    challenge = loader.load_challenge(forecaster_filter=2, problem_filter=5)
    
    # Should still have a valid challenge
    assert isinstance(challenge, ForecastChallenge)
    assert len(challenge.forecast_problems) > 0
    
    # Test that filtering worked - each problem should have at least 5 forecasts
    for problem in challenge.forecast_problems:
        assert len(problem.forecasts) >= 5
    
    # Test that each forecaster participated in at least 2 problems
    forecaster_problem_counts = {}
    for problem in challenge.forecast_problems:
        for forecast in problem.forecasts:
            if forecast.username not in forecaster_problem_counts:
                forecaster_problem_counts[forecast.username] = set()
            forecaster_problem_counts[forecast.username].add(problem.problem_id)
    
    for username, problem_ids in forecaster_problem_counts.items():
        assert len(problem_ids) >= 2
    
    print("✓ Filtered challenge loading test passed")


def test_challenge_properties():
    """Test computed properties of the challenge."""
    
    loader = GJOChallengeLoader(
        predictions_file=PREDICTIONS_FILE,
        metadata_file=METADATA_FILE,
        challenge_title="Sports Challenge 2024"
    )
    
    challenge = loader.load_challenge()
    
    # Test forecaster map
    assert len(challenge.forecaster_map) == challenge.num_forecasters
    assert len(challenge.unique_forecasters) == challenge.num_forecasters
    
    # Test that we can get problems by ID
    if challenge.forecast_problems:
        first_problem_id = challenge.forecast_problems[0].problem_id
        found_problem = challenge.get_problem_by_id(first_problem_id)
        assert found_problem is not None
        assert found_problem.problem_id == first_problem_id
    
    # Test that we can get forecaster problems
    if challenge.unique_forecasters:
        first_forecaster = challenge.unique_forecasters[0]
        forecaster_problems = challenge.get_forecaster_problems(first_forecaster)
        assert len(forecaster_problems) > 0
    
    print("✓ Challenge properties test passed")


def test_problem_properties():
    """Test computed properties of problems."""
    
    loader = GJOChallengeLoader(
        predictions_file=PREDICTIONS_FILE,
        metadata_file=METADATA_FILE,
        challenge_title="Sports Challenge 2024"
    )
    
    challenge = loader.load_challenge()
    
    if challenge.forecast_problems:
        problem = challenge.forecast_problems[0]
        
        # Test crowd probabilities
        crowd_probs = problem.crowd_probs
        assert len(crowd_probs) == len(problem.options)
        assert abs(sum(crowd_probs) - 1.0) < 1e-6  # Should sum to 1
        
        # Test unique forecasters
        unique_forecasters = problem.unique_forecasters
        assert len(unique_forecasters) <= len(problem.forecasts)
        
        # Test odds property
        assert problem.has_odds == (problem.odds is not None)
    
    print("✓ Problem properties test passed")


def run_all_tests():
    """Run all tests."""
    print("Running GJOChallengeLoader tests...")
    print("=" * 50)
    
    try:
        test_gjo_loader_basic()
        test_gjo_loader_full_challenge()
        test_gjo_loader_with_filters()
        test_challenge_properties()
        test_problem_properties()
        
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