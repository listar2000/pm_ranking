#!/usr/bin/env python3
"""
Tests for average_return.py functions using assert-based testing.
"""
from pm_rank.model.average_return import _get_risk_neutral_bets, _get_risk_averse_log_bets
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))


def test_risk_neutral_bets_basic():
    """Test basic functionality of _get_risk_neutral_bets."""
    # Simple case with 2 forecasters and 3 options
    forecast_probs = np.array([
        [0.4, 0.3, 0.3],  # Forecaster 1
        [0.2, 0.5, 0.3]   # Forecaster 2
    ])
    implied_probs = np.array([0.33, 0.33, 0.34])

    bets = _get_risk_neutral_bets(forecast_probs, implied_probs)

    # Check shape
    assert bets.shape == (2, 3)

    # Check that each forecaster bets on exactly one option (all-in strategy)
    assert np.all(np.sum(bets > 0, axis=1) == 1)

    # Check that the bets are placed on the option with maximum edge
    edges = forecast_probs - implied_probs
    expected_max_edges = np.argmax(edges, axis=1)

    for i in range(2):
        assert bets[i, expected_max_edges[i]] > 0
        assert np.all(bets[i, :] == 0) or np.sum(bets[i, :] > 0) == 1

    print("✓ Risk neutral bets basic test passed")


def test_risk_neutral_bets_edge_calculation():
    """Test that risk neutral bets correctly identify the maximum edge."""
    # Test case where the edge is clear
    forecast_probs = np.array([
        [0.8, 0.1, 0.1],  # Strong belief in option 0
        [0.1, 0.8, 0.1]   # Strong belief in option 1
    ])
    implied_probs = np.array([0.33, 0.33, 0.34])

    bets = _get_risk_neutral_bets(forecast_probs, implied_probs)

    # Forecaster 1 should bet on option 0 (edge = 0.8 - 0.33 = 0.47)
    assert bets[0, 0] > 0
    assert bets[0, 1] == 0
    assert bets[0, 2] == 0

    # Forecaster 2 should bet on option 1 (edge = 0.8 - 0.33 = 0.47)
    assert bets[1, 0] == 0
    assert bets[1, 1] > 0
    assert bets[1, 2] == 0

    print("✓ Risk neutral bets edge calculation test passed")


def test_risk_neutral_bets_bet_values():
    """Test that bet values are correctly calculated as 1/implied_prob."""
    forecast_probs = np.array([[0.6, 0.4]])  # Single forecaster
    implied_probs = np.array([0.5, 0.5])

    bets = _get_risk_neutral_bets(forecast_probs, implied_probs)

    # Edge for option 0: 0.6 - 0.5 = 0.1
    # Edge for option 1: 0.4 - 0.5 = -0.1
    # Should bet on option 0 with value 1/0.5 = 2.0
    assert bets[0, 0] == 2.0
    assert bets[0, 1] == 0.0

    print("✓ Risk neutral bets value calculation test passed")


def test_log_risk_averse_bets_basic():
    """Test basic functionality of _get_risk_averse_log_bets."""
    # Simple case with 2 forecasters and 3 options
    forecast_probs = np.array([
        [0.4, 0.3, 0.3],  # Forecaster 1
        [0.2, 0.5, 0.3]   # Forecaster 2
    ])
    implied_probs = np.array([0.33, 0.33, 0.34])

    bets = _get_risk_averse_log_bets(forecast_probs, implied_probs)

    # Check shape
    assert bets.shape == (2, 3)

    # Check that bets are proportional to forecast_probs / implied_probs
    expected_bets = forecast_probs / implied_probs
    np.testing.assert_array_almost_equal(bets, expected_bets)

    print("✓ Log risk averse bets basic test passed")


def test_log_risk_averse_bets_proportional():
    """Test that log risk averse bets are proportional to forecast probabilities."""
    forecast_probs = np.array([[0.6, 0.4]])  # Single forecaster
    implied_probs = np.array([0.5, 0.5])

    bets = _get_risk_averse_log_bets(forecast_probs, implied_probs)

    # Expected bets: [0.6/0.5, 0.4/0.5] = [1.2, 0.8]
    expected_bets = np.array([[1.2, 0.8]])
    np.testing.assert_array_almost_equal(bets, expected_bets)

    print("✓ Log risk averse bets proportional test passed")


def test_log_risk_averse_bets_different_implied_probs():
    """Test log risk averse bets with different implied probabilities."""
    forecast_probs = np.array([[0.5, 0.5]])  # Single forecaster
    implied_probs = np.array([0.3, 0.7])  # Different implied probabilities

    bets = _get_risk_averse_log_bets(forecast_probs, implied_probs)

    # Expected bets: [0.5/0.3, 0.5/0.7] = [1.67, 0.71]
    expected_bets = np.array([[0.5/0.3, 0.5/0.7]])
    np.testing.assert_array_almost_equal(bets, expected_bets, decimal=2)

    print("✓ Log risk averse bets different implied probs test passed")


def test_both_functions_consistency():
    """Test that both functions handle edge cases consistently."""
    # Test with equal forecast and implied probabilities
    forecast_probs = np.array([[0.33, 0.33, 0.34]])
    implied_probs = np.array([0.33, 0.33, 0.34])

    risk_neutral_bets = _get_risk_neutral_bets(forecast_probs, implied_probs)
    log_risk_averse_bets = _get_risk_averse_log_bets(
        forecast_probs, implied_probs)

    # Risk neutral should still pick one option (even with equal edges)
    assert np.sum(risk_neutral_bets > 0) == 1

    # Log risk averse should bet on all options proportionally
    assert np.all(log_risk_averse_bets > 0)

    print("✓ Both functions consistency test passed")


def test_input_validation():
    """Test that functions handle input validation correctly."""
    # Test with mismatched shapes
    forecast_probs = np.array([[0.5, 0.5]])
    implied_probs = np.array([0.5])  # Wrong shape

    try:
        _get_risk_neutral_bets(forecast_probs, implied_probs)
        assert False, "Should have raised an AssertionError"
    except AssertionError:
        pass

    try:
        _get_risk_averse_log_bets(forecast_probs, implied_probs)
        assert False, "Should have raised an AssertionError"
    except AssertionError:
        pass

    print("✓ Input validation test passed")


def run_all_tests():
    """Run all tests."""
    print("Running average_return.py tests...")
    print("=" * 50)

    try:
        test_risk_neutral_bets_basic()
        test_risk_neutral_bets_edge_calculation()
        test_risk_neutral_bets_bet_values()
        test_log_risk_averse_bets_basic()
        test_log_risk_averse_bets_proportional()
        test_log_risk_averse_bets_different_implied_probs()
        test_both_functions_consistency()
        test_input_validation()

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
