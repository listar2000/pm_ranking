"""
Scoring Rules for Ranking Forecasters in Prediction Markets.

This module implements proper scoring rules to evaluate and rank forecasters based on their
probabilistic predictions. Proper scoring rules are essential for ensuring that forecasters
are incentivized to report their true beliefs, as they are rewarded for accuracy and
calibration rather than just getting the highest probability outcome correct.

Reference: https://www.cis.upenn.edu/~aaroth/courses/slides/agt17/lect23.pdf

Key Concepts:

* **Proper Scoring Rules**: Mathematical functions that incentivize honest reporting of
  probabilistic beliefs by rewarding accuracy and calibration.

* **Brier Score**: A quadratic scoring rule that measures the squared difference between
  predicted probabilities and actual outcomes.

* **Logarithmic Score**: A scoring rule based on the logarithm of the predicted probability
  of the actual outcome.

* **Spherical Score**: A scoring rule that normalizes predictions to unit vectors and
  measures the cosine similarity with the actual outcome.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Iterator, Dict, Tuple, Any
from pm_rank.data.base import ForecastProblem
from pm_rank.model.utils import forecaster_data_to_rankings, get_logger, log_ranking_table
import logging

# we use the following quantiles to cap the problem weights
MAX_PROBLEM_WEIGHT_QUANTILE = 0.75
MIN_PROBLEM_WEIGHT_QUANTILE = 0.25


class ScoringRule(ABC):
    """Abstract base class for proper scoring rules.

    This class provides the foundation for implementing various proper scoring rules
    used to evaluate probabilistic forecasts. Proper scoring rules ensure that
    forecasters are incentivized to report their true beliefs by rewarding both
    accuracy and calibration.

    :param verbose: Whether to enable verbose logging (default: False).
    """

    def __init__(self, verbose: bool = False):
        """Initialize the scoring rule.

        :param verbose: Whether to enable verbose logging (default: False).
        """
        self.verbose = verbose
        self.logger = get_logger(f"pm_rank.model.{self.__class__.__name__}")
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

    @abstractmethod
    def _score_fn(self, correct_probs: np.ndarray, all_probs: np.ndarray) -> np.ndarray:
        """Implement the scoring function for the specific rule.

        This abstract method must be implemented by subclasses to define the
        specific mathematical formulation of the scoring rule.

        :param correct_probs: Array of predicted probabilities for the actual outcomes.
                             Shape (n,) where n is the number of forecasts.
        :param all_probs: Array of all predicted probability distributions.
                         Shape (n, k) where n is number of forecasts, k is number of options.

        :returns: Array of scores for each forecast. Shape (n,).
        """
        pass

    def _get_problem_weights(self, problem_discriminations: np.ndarray) -> np.ndarray:
        """Calculate problem weights based on discrimination parameters.

        This method implements a weighting scheme that gives more importance to
        problems that better distinguish between strong and weak forecasters.
        The weights are capped using quantiles to prevent extreme values from
        dominating the overall score.

        :param problem_discriminations: Array of discrimination parameters for each problem.
                                       Higher values indicate problems that better distinguish
                                       between forecasters.

        :returns: Normalized problem weights that sum to the number of problems.
        """
        # cap the problem weights
        lower_bound = np.quantile(
            problem_discriminations, MIN_PROBLEM_WEIGHT_QUANTILE)
        upper_bound = np.quantile(
            problem_discriminations, MAX_PROBLEM_WEIGHT_QUANTILE)
        problem_weights = np.clip(
            problem_discriminations, a_min=lower_bound, a_max=upper_bound)
        # normalize the problem weights
        problem_weights = len(problem_discriminations) * \
            problem_weights / np.sum(problem_weights)
        return problem_weights

    def fit(self, problems: List[ForecastProblem], problem_discriminations: np.ndarray | List[float] | None = None, include_scores: bool = True) \
            -> Tuple[Dict[str, Any], Dict[str, int]] | Dict[str, int]:
        """Fit the scoring rule to the given problems and return rankings.

        This method processes all problems and calculates scores for each forecaster
        using the implemented scoring rule. Optionally, problem weights can be applied
        based on discrimination parameters to give more importance to more informative
        problems.

        :param problems: List of ForecastProblem instances to evaluate.
        :param problem_discriminations: Optional array of discrimination parameters for
                                       weighting problems. If None, all problems are weighted equally.
        :param include_scores: Whether to include scores in the results (default: True).

        :returns: Ranking results, either as a tuple of (scores, rankings) or just rankings.
        """
        forecaster_data = {}

        if problem_discriminations is not None:
            problem_weights = self._get_problem_weights(
                np.array(problem_discriminations))
        else:
            problem_weights = np.ones(len(problems))

        for i, problem in enumerate(problems):
            correct_probs, all_probs, usernames = [], [], []
            for forecast in problem.forecasts:
                username = forecast.username
                if username not in forecaster_data:
                    forecaster_data[username] = []
                usernames.append(username)
                correct_probs.append(forecast.correct_prob)
                all_probs.append(forecast.probs)

            correct_probs = np.array(correct_probs)
            all_probs = np.array(all_probs)
            # weight the scores by the problem weights
            scores = self._score_fn(
                correct_probs, all_probs) * problem_weights[i]
            # attribute the scores to the forecasters
            for username, score in zip(usernames, scores):
                forecaster_data[username].append(score)

        result = forecaster_data_to_rankings(
            forecaster_data, include_scores=include_scores, ascending=False, aggregate="mean")
        if self.verbose:
            log_ranking_table(self.logger, result)
        return result

    def fit_stream(self, problem_iter: Iterator[List[ForecastProblem]], include_scores: bool = True) -> Dict[int, Tuple[Dict[str, Any], Dict[str, int]]]:
        """Fit the scoring rule to streaming problems and return incremental results.

        This method processes problems as they arrive and returns rankings after each batch,
        allowing for incremental analysis of forecaster performance over time.

        :param problem_iter: Iterator over batches of ForecastProblem instances.
        :param include_scores: Whether to include scores in the results (default: True).

        :returns: Mapping of batch indices to ranking results.
        """
        forecaster_data = {}
        batch_results = {}
        batch_id = 0

        for batch in problem_iter:
            if self.verbose:
                self.logger.debug(f"Processing batch {batch_id}")
            for problem in batch:
                correct_probs, all_probs, usernames = [], [], []
                for forecast in problem.forecasts:
                    username = forecast.username
                    if username not in forecaster_data:
                        forecaster_data[username] = []
                    usernames.append(username)
                    correct_probs.append(forecast.correct_prob)
                    all_probs.append(forecast.probs)

                # batch process the scores
                correct_probs = np.array(correct_probs)
                all_probs = np.array(all_probs)
                scores = self._score_fn(correct_probs, all_probs)

                for username, score in zip(usernames, scores):
                    forecaster_data[username].append(score)

            batch_results[batch_id] = forecaster_data_to_rankings(
                forecaster_data, include_scores=include_scores, ascending=False, aggregate="mean")

            if self.verbose:
                log_ranking_table(self.logger, batch_results[batch_id])

            batch_id += 1

        return batch_results


class LogScoringRule(ScoringRule):
    """Logarithmic scoring rule for evaluating probabilistic forecasts.

    The logarithmic scoring rule is a proper scoring rule that rewards forecasters
    based on the logarithm of their predicted probability for the actual outcome.
    This rule heavily penalizes overconfident predictions and rewards well-calibrated
    forecasts.

    :param clip_prob: Minimum probability value to prevent log(0) (default: 0.01).
    :param verbose: Whether to enable verbose logging (default: False).
    """

    def __init__(self, clip_prob: float = 0.01, verbose: bool = False):
        """Initialize the logarithmic scoring rule.

        :param clip_prob: Minimum probability value to prevent log(0) (default: 0.01).
        :param verbose: Whether to enable verbose logging (default: False).
        """
        super().__init__(verbose=verbose)
        self.clip_prob = clip_prob
        self.logger.info(
            f"Initialized {self.__class__.__name__} with hyperparam: clip_prob={clip_prob}")

    def _score_fn(self, correct_probs: np.ndarray, all_probs: np.ndarray) -> np.ndarray:
        """Calculate logarithmic scores for the forecasts.

        The logarithmic score is computed as log(p_correct), where p_correct is the
        predicted probability of the actual outcome. To prevent numerical issues,
        probabilities are clipped to a minimum value.

        :param correct_probs: Array of predicted probabilities for the actual outcomes.
                             Shape (n,) where n is the number of forecasts.
        :param all_probs: Array of all predicted probability distributions.
                         Shape (n, k) where n is number of forecasts, k is number of options.

        :returns: Array of logarithmic scores. Shape (n,).
        """
        return np.log(np.maximum(correct_probs, self.clip_prob))


class BrierScoringRule(ScoringRule):
    """Brier scoring rule for evaluating probabilistic forecasts.

    The Brier score is a quadratic proper scoring rule that measures the squared
    difference between predicted probabilities and actual outcomes. It is widely
    used in prediction markets and provides a good balance between rewarding
    accuracy and calibration.

    :param negate: Whether to negate the scores so that higher values are better
                   (default: True).
    :param verbose: Whether to enable verbose logging (default: False).
    """

    def __init__(self, negate: bool = True, verbose: bool = False):
        """Initialize the Brier scoring rule.

        :param negate: Whether to negate the scores so that higher values are better
                       (default: True).
        :param verbose: Whether to enable verbose logging (default: False).
        """
        super().__init__(verbose=verbose)
        self.negate = negate
        self.logger.info(
            f"Initialized {self.__class__.__name__} with hyperparam: negate={negate}")

    def _score_fn(self, correct_probs: np.ndarray, all_probs: np.ndarray, negate: bool = True) -> np.ndarray:
        """Calculate Brier scores for the forecasts.

        The Brier score is computed as the average squared difference between
        predicted probabilities and actual outcomes. The formula is:

        Brier Score = (1 - p_correct)² + Σ(p_incorrect)²

        where p_correct is the predicted probability of the actual outcome and
        p_incorrect are the predicted probabilities of incorrect outcomes.

        :param correct_probs: Array of predicted probabilities for the actual outcomes.
                             Shape (n,) where n is the number of forecasts.
        :param all_probs: Array of all predicted probability distributions.
                         Shape (n, k) where n is number of forecasts, k is number of options.
        :param negate: Whether to negate the scores so that higher values are better
                       (default: True).

        :returns: Array of Brier scores. Shape (n,).
        """
        # correct_probs is 1D with shape (n,), all_probs is 2D with shape (n, k)
        # (1) we obtain (n,) correct_scores
        correct_scores = (1 - correct_probs) ** 2 - correct_probs ** 2
        # (2) we obtain (n,) incorrect scores
        incorrect_scores = np.sum(all_probs ** 2, axis=1)
        # (3) we obtain (n,) scores, rescaled so that it lies in [0, 1]
        scores = (correct_scores + incorrect_scores) / 2
        # (4) negate the result since higher scores are better
        return -scores if negate else scores


class SphericalScoringRule(ScoringRule):
    """Spherical scoring rule for evaluating probabilistic forecasts.

    The spherical scoring rule normalizes probability vectors to unit vectors and
    measures the cosine similarity with the actual outcome. This rule is less
    sensitive to extreme probability values compared to the logarithmic rule.

    :param verbose: Whether to enable verbose logging (default: False).
    """

    def __init__(self, verbose: bool = False):
        """Initialize the spherical scoring rule.

        :param verbose: Whether to enable verbose logging (default: False).
        """
        super().__init__(verbose=verbose)
        self.logger.info(f"Initialized {self.__class__.__name__}")

    def _score_fn(self, correct_probs: np.ndarray, all_probs: np.ndarray) -> np.ndarray:
        """Calculate spherical scores for the forecasts.

        The spherical score is computed as the cosine similarity between the
        normalized probability vector and the actual outcome vector. The formula is:

        Spherical Score = p_correct / $\\lVert p \\rVert    $

        where p_correct is the predicted probability of the actual outcome and
        $\\lVert p \\rVert$ is the L2 norm of the entire probability vector.

        :param correct_probs: Array of predicted probabilities for the actual outcomes.
                             Shape (n,) where n is the number of forecasts.
        :param all_probs: Array of all predicted probability distributions.
                         Shape (n, k) where n is number of forecasts, k is number of options.

        :returns: Array of spherical scores. Shape (n,).
        """
        # formula: r_j / sum_i r_i where r_j is the correct probability of the j-th option
        correct_scores = correct_probs / np.linalg.norm(all_probs, axis=1)
        return correct_scores


if __name__ == "__main__":
    # check the implementation of the scoring rules
    correct_probs = np.array([0, 0.5, 1])
    all_probs = np.array([[0, 0.4, 0.6], [0.2, 0.5, 0.3], [0, 0, 1]])
    print(LogScoringRule()._score_fn(correct_probs, all_probs))
    print(BrierScoringRule()._score_fn(correct_probs, all_probs))
    print(SphericalScoringRule()._score_fn(correct_probs, all_probs))
