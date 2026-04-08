"""Shared pytest fixtures and marker-based auto-skip logic for the pm_rank test suite."""
import importlib.util
from datetime import datetime
from pathlib import Path

import pytest

from pm_rank.data import ForecastChallenge, ForecastEvent, ForecastProblem

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "src" / "pm_rank" / "data" / "raw"

GJO_PREDICTIONS = RAW_DIR / "all_predictions.json"
GJO_METADATA = RAW_DIR / "sports_challenge_metadata.json"
PROPHET_CSV = RAW_DIR / "prophet_arena_full.csv"


def _has_gjo_data() -> bool:
    return GJO_PREDICTIONS.exists() and GJO_METADATA.exists()


def _has_prophet_data() -> bool:
    return PROPHET_CSV.exists()


def _has_pyro() -> bool:
    # Use find_spec so we don't actually import pyro (and its heavy transitive
    # deps) during test collection — we only need to know if it's installable.
    return importlib.util.find_spec("pyro") is not None


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests whose prerequisites are unavailable.

    Tests opt in by applying one of the registered markers (``requires_gjo_data``,
    ``requires_prophet_data``, ``requires_pyro``) at module or function level. This
    hook then attaches a ``skip`` marker when the prerequisite is missing, so CI
    runs stay green on a fresh clone without raw data or the ``full`` extra.
    """
    skip_gjo = pytest.mark.skip(reason="GJO raw data files not present in src/pm_rank/data/raw")
    skip_prophet = pytest.mark.skip(reason="Prophet Arena raw CSV not present in src/pm_rank/data/raw")
    skip_pyro = pytest.mark.skip(reason="pyro-ppl not installed (install the 'full' extra)")

    has_gjo = _has_gjo_data()
    has_prophet = _has_prophet_data()
    has_pyro = _has_pyro()

    for item in items:
        if "requires_gjo_data" in item.keywords and not has_gjo:
            item.add_marker(skip_gjo)
        if "requires_prophet_data" in item.keywords and not has_prophet:
            item.add_marker(skip_prophet)
        if "requires_pyro" in item.keywords and not has_pyro:
            item.add_marker(skip_pyro)


# --- Synthetic data fixtures (no raw data required) -------------------------


def _mk_event(forecast_id: str, problem_id: str, username: str, probs, odds=None, no_odds=None):
    return ForecastEvent(
        forecast_id=forecast_id,
        problem_id=problem_id,
        username=username,
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        probs=probs,
        odds=odds,
        no_odds=no_odds,
    )


def _complement(odds):
    return [1.0 - o for o in odds]


@pytest.fixture
def synthetic_forecasts():
    """Three forecasters predicting on a 3-option problem, with YES and NO odds."""
    yes = [0.5, 0.3, 0.2]
    no = _complement(yes)
    return [
        _mk_event("f1", "p1", "alice", [0.7, 0.2, 0.1], odds=yes, no_odds=no),
        _mk_event("f2", "p1", "bob", [0.3, 0.4, 0.3], odds=yes, no_odds=no),
        _mk_event("f3", "p1", "carol", [0.2, 0.2, 0.6], odds=yes, no_odds=no),
    ]


@pytest.fixture
def synthetic_problem(synthetic_forecasts):
    """Single problem with correct_option_idx=[0] (so alice is best)."""
    return ForecastProblem(
        title="Test Problem 1",
        problem_id="p1",
        options=["A", "B", "C"],
        correct_option_idx=[0],
        forecasts=synthetic_forecasts,
        end_time=datetime(2024, 1, 2, 0, 0, 0),
        num_forecasters=3,
    )


@pytest.fixture
def synthetic_challenge(synthetic_problem):
    """Two-problem challenge. All three forecasters appear in both problems."""
    yes2 = [0.5, 0.5]
    no2 = _complement(yes2)
    forecasts2 = [
        _mk_event("g1", "p2", "alice", [0.55, 0.45], odds=yes2, no_odds=no2),
        _mk_event("g2", "p2", "bob", [0.50, 0.50], odds=yes2, no_odds=no2),
        _mk_event("g3", "p2", "carol", [0.15, 0.85], odds=yes2, no_odds=no2),
    ]
    problem2 = ForecastProblem(
        title="Test Problem 2",
        problem_id="p2",
        options=["X", "Y"],
        correct_option_idx=[1],  # carol is best on this one
        forecasts=forecasts2,
        end_time=datetime(2024, 1, 3, 0, 0, 0),
        num_forecasters=3,
    )
    return ForecastChallenge(
        title="Synthetic Challenge",
        forecast_problems=[synthetic_problem, problem2],
    )
