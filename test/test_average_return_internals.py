"""Tests for private betting helpers in ``pm_rank.model.average_return``.

These exercise ``_get_risk_neutral_bets`` and ``_get_risk_averse_log_bets`` with
purely synthetic numpy inputs, so no raw data is required.
"""
import numpy as np

from pm_rank.model.average_return import _get_risk_averse_log_bets, _get_risk_neutral_bets


def test_risk_neutral_bets_basic():
    """Each forecaster bets all-in on exactly one option."""
    forecast_probs = np.array([
        [0.4, 0.3, 0.3],
        [0.2, 0.5, 0.3],
    ])
    implied_probs = np.array([0.33, 0.33, 0.34])

    bets = _get_risk_neutral_bets(forecast_probs, implied_probs)

    assert bets.shape == (2, 3)
    assert np.all(np.sum(bets > 0, axis=1) == 1)

    edges = forecast_probs - implied_probs
    expected_max_edges = np.argmax(edges, axis=1)
    for i in range(2):
        assert bets[i, expected_max_edges[i]] > 0


def test_risk_neutral_bets_edge_calculation():
    """With a clear edge, each forecaster bets on the option it is strongest on."""
    forecast_probs = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
    ])
    implied_probs = np.array([0.33, 0.33, 0.34])

    bets = _get_risk_neutral_bets(forecast_probs, implied_probs)

    assert bets[0, 0] > 0
    assert bets[0, 1] == 0
    assert bets[0, 2] == 0

    assert bets[1, 0] == 0
    assert bets[1, 1] > 0
    assert bets[1, 2] == 0


def test_risk_neutral_bets_bet_values():
    """Bet value for the chosen option equals 1 / implied_prob."""
    forecast_probs = np.array([[0.6, 0.4]])
    implied_probs = np.array([0.5, 0.5])

    bets = _get_risk_neutral_bets(forecast_probs, implied_probs)

    assert bets[0, 0] == 2.0
    assert bets[0, 1] == 0.0


def test_log_risk_averse_bets_basic():
    """Log-risk-averse bets equal ``forecast_probs / implied_probs``."""
    forecast_probs = np.array([
        [0.4, 0.3, 0.3],
        [0.2, 0.5, 0.3],
    ])
    implied_probs = np.array([0.33, 0.33, 0.34])

    bets = _get_risk_averse_log_bets(forecast_probs, implied_probs)

    assert bets.shape == (2, 3)
    expected_bets = forecast_probs / implied_probs
    np.testing.assert_array_almost_equal(bets, expected_bets)


def test_log_risk_averse_bets_proportional():
    forecast_probs = np.array([[0.6, 0.4]])
    implied_probs = np.array([0.5, 0.5])

    bets = _get_risk_averse_log_bets(forecast_probs, implied_probs)

    expected_bets = np.array([[1.2, 0.8]])
    np.testing.assert_array_almost_equal(bets, expected_bets)


def test_log_risk_averse_bets_different_implied_probs():
    forecast_probs = np.array([[0.5, 0.5]])
    implied_probs = np.array([0.3, 0.7])

    bets = _get_risk_averse_log_bets(forecast_probs, implied_probs)

    expected_bets = np.array([[0.5 / 0.3, 0.5 / 0.7]])
    np.testing.assert_array_almost_equal(bets, expected_bets, decimal=2)


def test_both_functions_consistency():
    """With equal forecast and implied probs, risk-neutral still picks one, log spreads."""
    forecast_probs = np.array([[0.33, 0.33, 0.34]])
    implied_probs = np.array([0.33, 0.33, 0.34])

    risk_neutral_bets = _get_risk_neutral_bets(forecast_probs, implied_probs)
    log_risk_averse_bets = _get_risk_averse_log_bets(forecast_probs, implied_probs)

    assert np.sum(risk_neutral_bets > 0) == 1
    assert np.all(log_risk_averse_bets > 0)
