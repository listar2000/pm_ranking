"""Model subpackage for ``pm_rank``.

Bundles every scoring/ranking method ``pm_rank`` ships with. Each model takes
a list of :class:`pm_rank.data.ForecastProblem` objects via its ``fit()`` method
and returns either a single ``rankings`` dict or a ``(scores, rankings)`` tuple.

**Proper scoring rules** (`scoring_rule.py`) — score each forecast against
the realized outcome:

- :class:`BrierScoringRule` — quadratic score over the full probability vector.
- :class:`LogScoringRule` — logarithmic score on the probability of the truth.
- :class:`SphericalScoringRule` — cosine-similarity-style score.

**Comparative skill models:**

- :class:`GeneralizedBT` (`bradley_terry.py`) — generalized Bradley-Terry model
  estimating per-forecaster skill via Majorization-Minimization.

**Market-earning model:**

- :class:`AverageReturn` (`average_return.py`) — simulates CRRA-utility betting
  against the market and ranks by realized return. Requires ``odds`` and
  ``no_odds`` on each :class:`pm_rank.data.ForecastEvent`.

**Diagnostics:**

- :class:`CalibrationMetric` (`calibration.py`) — Expected Calibration Error
  (ECE) reliability diagnostic.
- :func:`spearman_correlation`, :func:`kendall_correlation` (`utils.py`) —
  rank-correlation utilities for comparing two rankings.

**Optional (requires ``pyro-ppl``):**

- :class:`IRTModel` (`irt/`) — Item Response Theory model with SVI or MCMC
  inference, jointly estimating forecaster ability and per-problem
  difficulty/discrimination parameters.
"""

# all models should be imported here
from .bradley_terry import GeneralizedBT
from .scoring_rule import BrierScoringRule, SphericalScoringRule, LogScoringRule
from .average_return import AverageReturn, AverageReturnConfig
from .calibration import CalibrationMetric
from .utils import spearman_correlation, kendall_correlation

__all__ = [
    "GeneralizedBT",
    "BrierScoringRule",
    "SphericalScoringRule",
    "LogScoringRule",
    "AverageReturn",
    "AverageReturnConfig",
    "spearman_correlation",
    "kendall_correlation",
    "CalibrationMetric"
]

try:
    from .irt import IRTModel, SVIConfig, MCMCConfig
    __all__.extend(["IRTModel", "SVIConfig", "MCMCConfig"])
except ImportError:
    pass
