"""
Generalized Bradley-Terry Model for Ranking Forecasters in Prediction Markets.

This module implements the generalized Bradley-Terry model to rank forecasters based on their
probabilistic predictions. The Bradley-Terry model is a statistical method for modeling
pairwise comparisons, which we extend to handle prediction market scenarios where each
event outcome is viewed as a contest between winning and losing "teams" composed of
forecaster contributions.

Reference: https://www.jmlr.org/papers/v7/huang06a.html

Key Concepts:

- **Bradley-Terry Model**: A statistical model for analyzing pairwise comparisons that
  estimates the relative strengths of competitors.
- **Generalized Extension**: Adapts the traditional pairwise model to prediction markets
  by treating each event outcome as a contest between winning and losing teams.
- **Skill Parameters**: Each forecaster has a skill parameter (theta) that represents
  their relative predictive ability.
- **Majorization-Minimization (MM)**: An iterative algorithm for fitting the model
  parameters that guarantees convergence.
"""

from collections import OrderedDict
import numpy as np
from typing import Literal, List, Dict, Any, Tuple
from pm_rank.data.base import ForecastProblem
from pm_rank.model.utils import forecaster_data_to_rankings, get_logger, log_ranking_table
import logging


class GeneralizedBT(object):
    """Generalized Bradley-Terry model for ranking forecasters in prediction markets.

    This class implements a generalization of the traditional Bradley-Terry model to
    handle prediction market scenarios. Each event outcome is treated as a contest
    between two "pseudo-teams": a winning team (the realized outcome) and a losing
    team (all other outcomes). Each forecaster contributes fractions of their capability
    proportional to their predicted probabilities.

    The model estimates skill parameters for each forecaster using an iterative
    Majorization-Minimization (MM) algorithm, which provides convergence guarantees
    and intuitive comparative scores similar to Elo ratings.

    :param method: Optimization method to use ("MM" for Majorization-Minimization).
    :param num_iter: Maximum number of iterations for the MM algorithm (default: 100).
    :param threshold: Convergence threshold for parameter updates (default: 1e-3).
    :param verbose: Whether to enable verbose logging (default: False).
    """

    def __init__(self, method: Literal["MM", "Elo"] = "MM", num_iter: int = 100, threshold: float = 1e-3, verbose: bool = False):
        """Initialize the generalized Bradley-Terry model.

        :param method: Optimization method to use ("MM" for Majorization-Minimization).
        :param num_iter: Maximum number of iterations for the MM algorithm (default: 100).
        :param threshold: Convergence threshold for parameter updates (default: 1e-3).
        :param verbose: Whether to enable verbose logging (default: False).
        """
        self.method = method
        self.num_iter = num_iter
        self.threshold = threshold
        self.verbose = verbose
        self.logger = get_logger(f"pm_rank.model.{self.__class__.__name__}")
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)
        self.logger.info(f"Initialized {self.__class__.__name__} with hyperparam: \n" +
                         f"method={method}, num_iter={num_iter}, threshold={threshold}")

    def fit(self, problems: List[ForecastProblem], include_scores: bool = True) -> \
            Tuple[Dict[str, Any], Dict[str, int]] | Dict[str, int]:
        """Fit the generalized Bradley-Terry model to the given problems.

        This method estimates skill parameters for each forecaster using the MM algorithm
        and returns rankings based on these parameters. The skill parameters represent
        the relative predictive ability of each forecaster.

        :param problems: List of ForecastProblem instances to evaluate.
        :param include_scores: Whether to include scores in the results (default: True).

        :returns: Ranking results, either as a tuple of (scores, rankings) or just rankings.
        """
        skills = self._fit_mm(problems, full_trajectory=False)
        result = forecaster_data_to_rankings(
            skills, include_scores=include_scores, ascending=False)  # type: ignore

        if self.verbose:
            log_ranking_table(self.logger, result)
        return result

    def _fit_mm(self, problems: List[ForecastProblem], full_trajectory: bool = False):
        """Fit the model using the Majorization-Minimization (MM) algorithm.

        The MM algorithm iteratively updates skill parameters by solving a series of
        weighted least squares problems. Each iteration computes win counts (W_t) and
        total match counts (D_t) for each forecaster, then updates skills as W_t / D_t.

        :param problems: List of ForecastProblem instances to evaluate.
        :param full_trajectory: Whether to return the full parameter trajectory
                                during optimization (default: False).

        :returns: Dictionary mapping usernames to skill parameters, or tuple of
                  (skills, trajectory) if full_trajectory=True.

        :note: Currently assumes each forecaster makes only one forecast per problem.
        """
        # TODO: currently we assume that each forecaster only makes one forecast per problem
        # Might need to come back and change this if this assumption fails.
        unique_forecasters = OrderedDict()
        num_forecasters = 0
        for problem in problems:
            for forecast in problem.forecasts:
                if forecast.username not in unique_forecasters:
                    unique_forecasters[forecast.username] = num_forecasters
                    num_forecasters += 1

        thetas = np.ones(num_forecasters)  # initialize the skills to be all 1
        old_thetas = np.zeros(num_forecasters)
        if full_trajectory:
            trajectory = [thetas]

        for t in range(self.num_iter):
            if self.verbose and (t % 10 == 0 or t == self.num_iter - 1):
                self.logger.debug(f"Iteration {t}/{self.num_iter}")
            # for each round, we need to calculate the W_t and D_t from existing thetas
            W_t, D_t = np.zeros(num_forecasters), np.zeros(num_forecasters)
            for problem in problems:
                D_p_denom = 0
                W_p_numer = np.zeros(len(problem.forecasts))
                indicators = []
                correct_option_first = problem.correct_option_idx[0]
                for i, forecast in enumerate(problem.forecasts):
                    user_idx = unique_forecasters[forecast.username]
                    indicators.append(user_idx)

                    W_p_numer[i] = thetas[user_idx] * forecast.probs[correct_option_first]
                    D_p_denom += thetas[user_idx]

                indicators = np.array(indicators)
                W_t[indicators] += W_p_numer / np.sum(W_p_numer)
                D_t[indicators] += 1 / D_p_denom

            # update the thetas
            old_thetas = thetas.copy()
            # only update thetas with non-zero D_t
            thetas[D_t > 0] = W_t[D_t > 0] / D_t[D_t > 0]
            # thetas should sum to num_forecasters
            thetas = thetas * (num_forecasters / np.sum(thetas))

            if full_trajectory:
                trajectory.append(thetas)

            # convergence check
            if np.max(np.abs(thetas - old_thetas)) < self.threshold:
                if self.verbose:
                    self.logger.info(
                        f"Generalized Bradley-Terry model converged after {t} iterations")
                break

        # return a dict of user_id, theta skill
        skills = dict(zip(unique_forecasters.keys(), thetas))

        if full_trajectory:
            return skills, trajectory
        else:
            return skills
