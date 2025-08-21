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
from typing import List, Iterator, Dict, Tuple, Any, Literal, Callable
from collections import OrderedDict
from pm_rank.data.base import ForecastProblem, ForecastChallenge
from pm_rank.model.utils import forecaster_data_to_rankings, get_logger, log_ranking_table, BootstrapCIConfig, DEFAULT_BOOTSTRAP_CI_CONFIG
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
    def _score_fn(self, correct_option_idx: np.ndarray, all_probs: np.ndarray) -> np.ndarray:
        """Implement the scoring function for the specific rule.

        This abstract method must be implemented by subclasses to define the
        specific mathematical formulation of the scoring rule.

        :param correct_option_idx: Array of indices of the correct options.
                             Shape (m,) where m is the number of correct options.
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

    def fit(self, problems: List[ForecastProblem], problem_discriminations: np.ndarray | List[float] | None = None, include_scores: bool = True, \
        include_bootstrap_ci: bool = False, include_per_problem_info: bool = False, \
        bootstrap_ci_config: BootstrapCIConfig = DEFAULT_BOOTSTRAP_CI_CONFIG) -> Tuple[Dict[str, Any], Dict[str, int]] | Dict[str, int]:
        """Fit the scoring rule to the given problems and return rankings.

        This method processes all problems and calculates scores for each forecaster
        using the implemented scoring rule. Optionally, problem weights can be applied
        based on discrimination parameters to give more importance to more informative
        problems.

        :param problems: List of ForecastProblem instances to evaluate.
        :param problem_discriminations: Optional array of discrimination parameters for
                                       weighting problems. If None, all problems are weighted equally.
        :param include_scores: Whether to include scores in the results (default: True).
        :param include_bootstrap_ci: Whether to include bootstrap confidence intervals (default: False).
        :param include_per_problem_info: Whether to include per-problem info in the results (default: False).
        :param bootstrap_ci_config: Configuration for bootstrap confidence intervals.

        :returns: Ranking results, either as a tuple of (scores, rankings) or just rankings.
                  If include_bootstrap_ci is True, adds bootstrap_cis to the tuple.
                  If include_per_problem_info is True, adds per_problem_info to the tuple.
        """
        forecaster_data = {}
        if include_per_problem_info:
            per_problem_info = []

        if problem_discriminations is not None:
            problem_weights = self._get_problem_weights(
                np.array(problem_discriminations))
        else:
            problem_weights = np.ones(len(problems))

        for i, problem in enumerate(problems):
            all_probs, usernames = [], []
            correct_option_idx = np.array(problem.correct_option_idx)
            for forecast in problem.forecasts:
                username = forecast.username
                if username not in forecaster_data:
                    forecaster_data[username] = []
                usernames.append(username)
                all_probs.append(forecast.unnormalized_probs)

            all_probs = np.array(all_probs)
            # weight the scores by the problem weights
            scores = self._score_fn(
                correct_option_idx, all_probs) * problem_weights[i]
            # attribute the scores to the forecasters
            for username, score in zip(usernames, scores):
                # we will weight the scores by the forecast weight for this event.
                forecaster_data[username].append(score * forecast.weight)

            if include_per_problem_info:
                for i, forecast in enumerate(problem.forecasts):
                    info = {
                        "forecast_id": forecast.forecast_id,
                        "username": forecast.username,
                        "problem_title": problem.title,
                        "problem_id": problem.problem_id,
                        "problem_category": problem.category,
                        "score": scores[i],
                        "probs": forecast.unnormalized_probs
                    }
                    if hasattr(forecast, "submission_id"):
                        info["submission_id"] = forecast.submission_id
                    per_problem_info.append(info)

        result = forecaster_data_to_rankings(
            forecaster_data, include_scores=include_scores, include_bootstrap_ci=include_bootstrap_ci, 
            ascending=False, aggregate="mean", bootstrap_ci_config=bootstrap_ci_config)
        if self.verbose:
            log_ranking_table(self.logger, result)
        return (*result, per_problem_info) if include_per_problem_info else result

    def _fit_stream_generic(self, batch_iter: Iterator, key_fn: Callable, include_scores: bool = True, use_ordered: bool = False):
        """Generic streaming fit function for both index and timestamp keys.

        This is a helper method that implements the common logic for streaming fits,
        whether using batch indices or timestamps as keys.

        :param batch_iter: Iterator over batches of problems.
        :param key_fn: Function to extract key and batch from iterator items.
        :param include_scores: Whether to include scores in the results (default: True).
        :param use_ordered: Whether to use OrderedDict for results (default: False).

        :returns: Mapping of keys to ranking results.
        """
        forecaster_data = {}
        batch_results = OrderedDict() if use_ordered else {}

        for i, item in enumerate(batch_iter):
            key, batch = key_fn(i, item)
            if self.verbose:
                msg = f"Processing batch {key}" if not use_ordered else f"Processing batch {i} at {key}"
                self.logger.debug(msg)

            # Process each problem in the batch
            for problem in batch:
                all_probs, usernames = [], []
                correct_option_idx = np.array(problem.correct_option_idx)
                for forecast in problem.forecasts:
                    username = forecast.username
                    if username not in forecaster_data:
                        forecaster_data[username] = []
                    usernames.append(username)
                    all_probs.append(forecast.unnormalized_probs)

                # batch process the scores
                all_probs = np.array(all_probs)
                scores = self._score_fn(correct_option_idx, all_probs)

                for username, score in zip(usernames, scores):
                    # we will weight the scores by the forecast weight for this event.
                    forecaster_data[username].append(score * forecast.weight)

            # Generate rankings for this batch
            batch_results[key] = forecaster_data_to_rankings(
                forecaster_data, include_scores=include_scores, ascending=False, aggregate="mean"
            )
            if self.verbose:
                log_ranking_table(self.logger, batch_results[key])

        return batch_results

    def fit_stream(self, problem_iter: Iterator[List[ForecastProblem]], include_scores: bool = True) -> Dict[int, Tuple[Dict[str, Any], Dict[str, int]]]:
        """Fit the scoring rule to streaming problems and return incremental results.

        This method processes problems as they arrive and returns rankings after each batch,
        allowing for incremental analysis of forecaster performance over time.

        :param problem_iter: Iterator over batches of ForecastProblem instances.
        :param include_scores: Whether to include scores in the results (default: True).

        :returns: Mapping of batch indices to ranking results.
        """
        return self._fit_stream_generic(
            problem_iter,
            key_fn=lambda i, batch: (i, batch),
            include_scores=include_scores,
            use_ordered=False
        )

    def fit_stream_with_timestamp(self, problem_time_iter: Iterator[Tuple[str, List[ForecastProblem]]], include_scores: bool = True) -> OrderedDict:
        """Fit the scoring rule to streaming problems with timestamps and return incremental results.

        This method processes problems with associated timestamps and returns rankings
        after each batch, maintaining chronological order.

        :param problem_time_iter: Iterator over (timestamp, problems) tuples.
        :param include_scores: Whether to include scores in the results (default: True).

        :returns: Chronologically ordered mapping of timestamps to ranking results.
        """
        return self._fit_stream_generic(
            problem_time_iter,
            key_fn=lambda i, item: (item[0], item[1]),
            include_scores=include_scores,
            use_ordered=True
        )

    def fit_by_category(self, problems: List[ForecastProblem], include_scores: bool = True, stream_with_timestamp: bool = False,
                        stream_increment_by: Literal["day", "week", "month"] = "day", min_bucket_size: int = 1) -> \
            Tuple[Dict[str, Any], Dict[str, int]] | Dict[str, int]:
        """Fit the scoring rule to the given problems by category.

        This method processes problems grouped by category and returns rankings for each category.
        Optionally, it can stream problems within each category over time.

        :param problems: List of ForecastProblem instances to process.
        :param include_scores: Whether to include scores in the results (default: True).
        :param stream_with_timestamp: Whether to stream problems with timestamps (default: False).
        :param stream_increment_by: The increment by which to stream problems (default: "day").
        :param min_bucket_size: The minimum number of problems to include in a bucket (default: 1).

        :returns: Mapping of categories to ranking results.
        """
        category_to_problems = dict()
        for problem in problems:
            if problem.category not in category_to_problems:
                category_to_problems[problem.category] = []
            category_to_problems[problem.category].append(problem)

        if not stream_with_timestamp:
            # simply fit the model to each category
            results_dict = dict()
            for category, category_problems in category_to_problems.items():
                results_dict[category] = self.fit(category_problems, include_scores=include_scores)
            results_dict["overall"] = self.fit(problems, include_scores=include_scores)
            return results_dict
        else:
            # create a separate iterator for overall problems
            overall_iterator = ForecastChallenge._stream_problems_over_time(
                problems=problems,
                increment_by=stream_increment_by,
                min_bucket_size=min_bucket_size
            )
            # create a separate iterator for each category
            results_dict = dict()
            for category, category_problems in category_to_problems.items():
                category_iterator = ForecastChallenge._stream_problems_over_time(
                    problems=category_problems,
                    increment_by=stream_increment_by,
                    min_bucket_size=min_bucket_size
                )

                results_dict[category] = self._fit_stream_generic(
                    category_iterator,
                    key_fn=lambda i, item: (item[0], item[1]),
                    include_scores=include_scores,
                    use_ordered=True
                )

            results_dict["overall"] = self._fit_stream_generic(
                overall_iterator,
                key_fn=lambda i, item: (item[0], item[1]),
                include_scores=include_scores,
                use_ordered=True
            )
            return results_dict


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

    def _score_fn(self, correct_option_idx: np.ndarray, all_probs: np.ndarray) -> np.ndarray:
        """Calculate logarithmic scores for the forecasts.

        The logarithmic score is computed as log(p_correct), where p_correct is the
        predicted probability of the actual outcome. To prevent numerical issues,
        probabilities are clipped to a minimum value.

        :param correct_option_idx: Array of indices of the correct options.
                             Shape (m,) where m is the number of correct options.
        :param all_probs: Array of all predicted probability distributions.
                         Shape (n, k) where n is number of forecasts, k is number of options.

        :returns: Array of logarithmic scores. Shape (n,).
        """
        return np.log(np.maximum(all_probs[:, correct_option_idx], self.clip_prob))


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

    def _score_fn(self, correct_option_idx: np.ndarray, all_probs: np.ndarray, negate: bool = True) -> np.ndarray:
        """Calculate Brier scores for the forecasts.

        The Brier score is computed as the average squared difference between
        predicted probabilities and actual outcomes. The formula is:

        Brier Score = Σ(1 - p_correct)² for all correct options

        where p_correct is the predicted probability of the actual outcome.

        :param correct_option_idx: Array of indices of the correct options.
                             Shape (m,) where m is the number of correct options.
        :param all_probs: Array of all predicted probability distributions.
                         Shape (n, k) where n is number of forecasts, k is number of options.
        :param negate: Whether to negate the scores so that higher values are better
                       (default: True).

        :returns: Array of Brier scores. Shape (n,).
        """
        one_hot = np.zeros(all_probs.shape[1])
        # correct_option_idx might be an empty array
        if len(correct_option_idx) > 0:
            one_hot[correct_option_idx] = 1
        # ignore above
        brier_scores = np.sum((all_probs - one_hot) ** 2, axis=1)
        # (3) we obtain (n,) scores, rescaled so that it lies in [0, 1]
        scores = brier_scores / all_probs.shape[1]
        # (4) negate the result since higher scores are better
        return 1 - scores if negate else scores


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

    def _score_fn(self, correct_option_idx: np.ndarray, all_probs: np.ndarray) -> np.ndarray:
        """Calculate spherical scores for the forecasts.

        The spherical score is computed as the cosine similarity between the
        normalized probability vector and the actual outcome vector. The formula is:

        Spherical Score = p_correct / $\\lVert p \\rVert    $

        where p_correct is the predicted probability of the actual outcome and
        $\\lVert p \\rVert$ is the L2 norm of the entire probability vector.

        :param correct_option_idx: Array of indices of the correct options.
                             Shape (m,) where m is the number of correct options.
        :param all_probs: Array of all predicted probability distributions.
                         Shape (n, k) where n is number of forecasts, k is number of options.

        :returns: Array of spherical scores. Shape (n,).
        """
        # formula: r_j / sum_i r_i where r_j is the correct probability of the j-th option
        return np.sum(all_probs[:, correct_option_idx], axis=1) / np.linalg.norm(all_probs, axis=1)
