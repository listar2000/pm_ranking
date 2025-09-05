"""
Calibration Metric for LLM Predictions. 

Currently this model adopts the following definition of a (perfectly-calibrated) probabilistic predictor f():

For all $p \\in [0, 1]$ and a pair of covariate $X$ and binary outcome $Y$, we have:
$$
\\mathbb{P}(Y = 1 | f(X) = p) = p
$$

We then define the (theoretical) **expected calibration error** (ECE) as a measure of deviation from the above property:
$$
\\text{ECE}^* = \\mathbb{E}_{X, Y}[ | \\mathbb{P}(Y = 1 | f(X)) - f(X) | ]
$$

In practice, we will calculate an empirical version of the above ECE via binning (discretization).

Reference: https://arxiv.org/pdf/2501.19047v2
"""

import numpy as np
from typing import Literal, List, Dict
from pm_rank.data.base import ForecastProblem
from pm_rank.model.utils import get_logger
from pm_rank.plotting.plot_reliability_diagram import plot_reliability_diagram
import logging


def _bin_stats(probs: List[float], labels: List[float], weights: List[float], n_bins: int, strategy: Literal["uniform", "quantile"]) \
    -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper function to calculate the bin statistics for the calibration metric.

    :param probs: List of probabilities.
    :param labels: List of labels.
    :param weights: List of weights.
    :param n_bins: Number of bins.
    :param strategy: Strategy to use for discretization.

    :returns: A tuple of bin centers, confidence, accuracy, and counts.
    """
    probs, labels, weights = np.asarray(probs, dtype=float), np.asarray(labels, dtype=float), np.asarray(weights, dtype=float)
    assert probs.shape == labels.shape and probs.shape == weights.shape and probs.ndim == 1, "probs, labels, and weights must have the same shape"

    # we check that the weights need to sum to equal the length of the input arrays
    assert np.isclose(weights.sum(), len(probs)), f"weights need to sum to equal the length of the input arrays, but got {weights.sum()} != {len(probs)}"
    
    n = len(probs)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:  # strategy == "quantile"
        # unique quantiles so we don't create empty bins when duplicates occur
        q = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(probs, q))
        # ensure at least 2 edges (if all probs identical)
        if edges.size < 2:
            edges = np.array([probs.min(), probs.max() + 1e-12])

    # assign to bins; `right=False` so the left edge is inclusive, right edge exclusive
    bin_ids = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, len(edges) - 2)

    counts, conf, acc = [], [], []
    bin_left, bin_right = edges[:-1], edges[1:]

    for b in range(len(edges)-1):
        idx = (bin_ids == b)
        if not np.any(idx):
            counts.append(0)
            conf.append(np.nan)
            acc.append(np.nan)
        else:
            counts.append(int(weights[idx].sum()))
            conf.append((probs[idx] * weights[idx]).sum() / weights[idx].sum())
            acc.append((labels[idx] * weights[idx]).sum() / weights[idx].sum())

    counts, conf, acc = np.asarray(counts), np.asarray(conf), np.asarray(acc)

    # Bin centers & widths for plotting
    bin_centers = 0.5 * (bin_left + bin_right)
    bin_widths = bin_right - bin_left

    return bin_centers, bin_widths, conf, acc, counts


def _calculate_ece(conf: np.ndarray, acc: np.ndarray, counts: np.ndarray, total_samples: int) -> float:
    """
    Calculate the Expected Calibration Error (ECE) from bin statistics.
    
    :param conf: Confidence (average predicted probability) for each bin.
    :param acc: Accuracy (fraction of correct predictions) for each bin.
    :param counts: Number of samples in each bin.
    :param total_samples: Total number of samples.
    
    :returns: The ECE score.
    """
    valid_mask = counts > 0
    ece_score = np.sum(np.abs(conf[valid_mask] - acc[valid_mask]) * (counts[valid_mask] / total_samples))
    return ece_score


class CalibrationMetric:
    def __init__(self, num_bins: int = 10, strategy: Literal["uniform", "quantile"] = "uniform", weight_event: bool = True, verbose: bool = False):
        """
        Initialize the CalibrationMetric.

        :param num_bins: The number of bins to use for discretization.
        :param strategy: The strategy to use for discretization.
        :param weight_event: Whether to weight the event by the number of markets in it. If `False`, then each market will be treated equally.
        """
        assert num_bins > 1, "num_bins must be greater than 1"

        self.num_bins = num_bins
        self.strategy = strategy
        self.weight_event = weight_event
        self.verbose = verbose

        self._fitted = False
        self._fitted_info = None

        self.logger = get_logger(f"pm_rank.model.{self.__class__.__name__}")
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)
        self.logger.info(f"Initialized {self.__class__.__name__} with config: \n" +
                         f"num_bins={self.num_bins}, strategy={self.strategy}, weight_event={self.weight_event}")

    def _prepare_forecaster_dicts(self, problems: List[ForecastProblem]) -> tuple[dict, dict, dict]:
        """
        Prepare the forecaster dictionaries for the calibration metric.

        :param problems: List of ForecastProblem instances to process.

        :returns: A tuple of dictionaries containing the probabilities, labels, weights, and number of events for each forecaster.
        """
        probs_dict, labels_dict, weights_dict, num_events = {}, {}, {}, {}
        for problem in problems:
            correct_option_idx = problem.correct_option_idx
            # turn this into a vector containing only 0/1
            labels = np.zeros(len(problem.options), dtype=int)
            labels[correct_option_idx] = 1

            for forecast in problem.forecasts:
                if forecast.username not in probs_dict:
                    probs_dict[forecast.username] = []
                    labels_dict[forecast.username] = []
                    weights_dict[forecast.username] = []
                    num_events[forecast.username] = 0

                num_markets = len(forecast.unnormalized_probs)
                probs_dict[forecast.username].extend(forecast.unnormalized_probs)
                labels_dict[forecast.username].extend(labels.tolist())
                num_events[forecast.username] += 1

                if self.weight_event:
                    # in this case, the weight of each market is inversely proportional to the number of markets. We need to further weight this
                    weights_dict[forecast.username].extend([1 / num_markets] * num_markets)
                else:
                    # NOTE: in this case, each market has a weight 1 equally
                    weights_dict[forecast.username].extend([1] * num_markets)

        if self.weight_event:
            # normalize the weights
            for username in weights_dict.keys():
                weights_dict[username] = np.asarray(weights_dict[username], dtype=float) / num_events[username]

        return probs_dict, labels_dict, weights_dict

    def fit(self, problems: List[ForecastProblem], include_scores: bool = True):
        """Fit the calibration metric to the given problems.

        :param problems: List of ForecastProblem instances to process.

        :returns: A dictionary containing the calibration metric.
        """
        probs_dict, labels_dict, weights_dict = self._prepare_forecaster_dicts(problems)
        
        forecaster_scores, fitted_info = {}, {}
        # calculate the bin stats for each forecaster
        for username in probs_dict.keys():
            probs = probs_dict[username]
            labels = labels_dict[username]
            weights = weights_dict[username]

            n = len(probs)

            bin_stats = _bin_stats(probs, labels, weights, self.num_bins, self.strategy)
            
            # Extract bin statistics
            bin_centers, bin_widths, conf, acc, counts = bin_stats

            # store the fitted info for each forecaster
            fitted_info[username] = (bin_centers, bin_widths, conf, acc, counts)

            # Calculate ECE using the extracted function
            forecaster_scores[username] = _calculate_ece(conf, acc, counts, n)
        
        # perform ranking
        sorted_usernames = sorted(forecaster_scores, key=lambda u: forecaster_scores[u])
        forecaster_ranking = {username: rank for rank, username in enumerate(sorted_usernames)}

        self._fitted = True
        self._fitted_info = fitted_info

        return (forecaster_scores, forecaster_ranking) if include_scores else forecaster_ranking

    def plot(self, name: str, title: str = "Reliability diagram", save_path: str = None, figsize: tuple[float, float] = (4,4), percent: bool = True):
        if not self._fitted:
            raise ValueError("CalibrationMetric must be fitted before plotting")
        if name not in self._fitted_info:
            raise ValueError(f"Forecaster {name} not found in fitted info")
        
        bin_centers, bin_widths, conf, acc, counts = self._fitted_info[name]

        ece = _calculate_ece(conf, acc, counts, counts.sum())

        fig, ax = plot_reliability_diagram(ece, bin_centers, bin_widths, conf, acc, counts, self.num_bins, title, \
            save_path=save_path, figsize=figsize, percent=percent)

        return fig, ax