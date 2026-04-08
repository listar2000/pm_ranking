"""``pm_rank``: a toolkit for scoring and ranking prediction-market forecasters.

``pm_rank`` provides a unified data model for prediction-market events and a
collection of principled scoring and ranking algorithms that operate on it.
It was originally developed for the
`Prophet Arena <https://www.prophetarena.co/>`_ LLM prediction-market platform
but is general purpose and works with any data source you can wrap in a
:class:`ChallengeLoader` (e.g. Good Judgment Open).

**Core data classes** (see :mod:`pm_rank.data`):

- :class:`ForecastEvent` — a single prediction by a forecaster on one problem.
- :class:`ForecastProblem` — a problem with options + the correct option + the
  list of forecasts made on it.
- :class:`ForecastChallenge` — a collection of problems with both batch and
  streaming iteration interfaces.

**Models** (see :mod:`pm_rank.model`):

- :class:`BrierScoringRule`, :class:`LogScoringRule`, :class:`SphericalScoringRule`
  — proper scoring rules.
- :class:`GeneralizedBT` — a generalized Bradley-Terry skill model fit by the
  Majorization-Minimization (MM) algorithm.
- :class:`AverageReturn` — CRRA market-earnings ranking (requires ``odds`` and
  ``no_odds`` on the forecast events).
- :class:`CalibrationMetric` — reliability diagnostic (ECE).
- :class:`IRTModel` (optional, requires ``pyro-ppl``) — Item Response Theory
  model with SVI or MCMC inference.

**Install:** ``pip install pm-rank`` for the base install, or
``pip install pm-rank[full]`` to also pull in the IRT extras.

A copy-pasteable quick-start example lives in the project README and at the
Mintlify docs site.
"""

# Import main subpackages
from . import data
from . import model

# Import commonly used classes for convenience
from .data import (
    ForecastEvent,
    ForecastProblem,
    ForecastChallenge,
    ChallengeLoader,
    GJOChallengeLoader,
    ProphetArenaChallengeLoader
)

from .model import (
    GeneralizedBT,
    BrierScoringRule,
    LogScoringRule,
    SphericalScoringRule,
    AverageReturn,
    spearman_correlation,
    kendall_correlation,
    CalibrationMetric
)

__all__ = [
    # Subpackages
    'data',
    'model',

    # Data classes
    'ForecastEvent',
    'ForecastProblem',
    'ForecastChallenge',
    'ChallengeLoader',
    'GJOChallengeLoader',
    'ProphetArenaChallengeLoader',

    # Model classes
    'GeneralizedBT',
    'BrierScoringRule',
    'LogScoringRule',
    'SphericalScoringRule',
    'AverageReturn',
    'spearman_correlation',
    'kendall_correlation',
    'CalibrationMetric'
]

# optionally import based on whether `pyro-ppl` is installed
try:
    from .model import IRTModel, SVIConfig, MCMCConfig, __all__
    __all__.extend(['IRTModel', 'SVIConfig', 'MCMCConfig'])
except ImportError:
    pass
