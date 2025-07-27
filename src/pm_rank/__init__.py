"""
`pm_rank`: A toolkit for scoring and ranking prediction market forecasters.
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
    kendall_correlation
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
    'kendall_correlation'
]

# optionally import based on whether `pyro-ppl` is installed
try:
    from .model import IRTModel, SVIConfig, MCMCConfig, __all__
    __all__.extend(['IRTModel', 'SVIConfig', 'MCMCConfig'])
except ImportError:
    pass