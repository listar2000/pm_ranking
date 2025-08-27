"""Model subpackage for pm_rank."""

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