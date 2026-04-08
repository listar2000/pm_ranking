"""Data subpackage for ``pm_rank``.

Defines the unified data model used by every scoring/ranking method in
``pm_rank`` and the loaders that convert raw data sources into it. The
hierarchy is bottom-up:

- :class:`ForecastEvent` — a single prediction by one forecaster on one problem.
- :class:`ForecastProblem` — a problem with options, the correct option, and a
  list of :class:`ForecastEvent` objects.
- :class:`ForecastChallenge` — a collection of :class:`ForecastProblem` objects;
  exposes both ``get_problems()`` (full-batch) and ``stream_problems()``
  (streaming/online) interfaces.

To use a new data source, subclass :class:`ChallengeLoader` and implement
``load_challenge()``. Two concrete loaders are bundled:

- :class:`GJOChallengeLoader` — Good Judgment Open scraped data.
- :class:`ProphetArenaChallengeLoader` — Prophet Arena prediction-market data.
"""

from .base import (
    ForecastEvent,
    ForecastProblem,
    ForecastChallenge,
    ChallengeLoader,
)

from .loaders import (
    GJOChallengeLoader,
    ProphetArenaChallengeLoader
)

__all__ = [
    'ForecastEvent',
    'ForecastProblem',
    'ForecastChallenge',
    'ChallengeLoader',
    'GJOChallengeLoader',
    'ProphetArenaChallengeLoader',
]
