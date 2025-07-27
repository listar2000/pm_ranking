"""
Pyro-based Item Response Theory (IRT) Models for Ranking Forecasters.

This module implements Item Response Theory models using Pyro for probabilistic programming.
IRT models are used to estimate latent abilities of forecasters and difficulty/discrimination
parameters of prediction problems based on their performance patterns.

The module provides two inference methods:

- **SVI (Stochastic Variational Inference)**: Fast approximate inference using variational methods
- **NUTS (No-U-Turn Sampler)**: Exact inference using Hamiltonian Monte Carlo sampling

Key Concepts:

* **Item Response Theory**: A psychometric framework that models the relationship between
  a person's latent ability and their probability of answering items correctly.

* **Forecaster Ability (θ)**: Latent parameter representing each forecaster's skill level.

* **Problem Difficulty (b)**: Parameter representing how difficult each prediction problem is.

* **Problem Discrimination (a)**: Parameter representing how well each problem distinguishes
  between forecasters of different abilities.

* **Category Parameters (p)**: Parameters for the discretized scoring bins used in the model.

Reference: https://en.wikipedia.org/wiki/Item_response_theory
"""

from ._pyro_models import IRTModel, MCMCConfig, SVIConfig
from ._dataset import IRTObs

__all__ = [
    "IRTModel",
    "IRTObs",
    "MCMCConfig",
    "SVIConfig",
]
