"""
IRT (Item Response Theory) models for forecaster ranking.

This module provides IRT-based models for ranking forecasters based on their performance
across multiple prediction problems.
"""

from ._pyro_models import IRTModel, MCMCConfig, SVIConfig
from ._dataset import IRTObs

__all__ = [
    "IRTModel",
    "IRTObs", 
    "MCMCConfig",
    "SVIConfig",
]
