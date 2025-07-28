"""
This is an internal class that handles the transformation of data from the `ForecastProblem` level to
the `observations` that can be used to fit an IRT model (through the `pyro` library).

Reference: https://github.com/nd-ball/py-irt/blob/master/py_irt/dataset.py
"""
from typing import Tuple, List, Literal, Dict
from pm_rank.data.base import ForecastProblem
from pm_rank.model.scoring_rule import BrierScoringRule
import numpy as np
import torch
from dataclasses import dataclass
from functools import cached_property


@dataclass
class IRTObs:
    """
    An internal, helper class that handles the transformation of data from the `ForecastProblem` level to an internal format

    The `forecaster_id_to_idx` and `problem_id_to_idx` are used to map the forecaster and problem ids to indices.
    This is useful for the `pyro` library, which requires the data to be in a certain format.

    :param forecaster_ids: A tensor of shape `(k,)` with the forecaster ids
    :param problem_ids: A tensor of shape `(k,)` with the problem ids
    :param forecaster_id_to_idx: A dictionary with the forecaster ids as keys and the indices as values
    :param problem_id_to_idx: A dictionary with the problem ids as keys and the indices as values
    :param scores: A tensor of shape `(k,)` with the scores of the forecasts (discretized from scoring rules)
    :param discretized_scores: A tensor of shape `(k,)` with the discretized scores of the forecasts
    :param anchor_points: A tensor of shape `(n_bins,)` with the anchor points of the discretized scores
    """
    forecaster_ids: torch.Tensor
    problem_ids: torch.Tensor
    forecaster_id_to_idx: Dict[str, int]
    problem_id_to_idx: Dict[str, int]
    scores: torch.Tensor
    discretized_scores: torch.Tensor
    anchor_points: torch.Tensor

    @cached_property
    def forecaster_idx_to_id(self) -> Dict[int, str]:
        return {v: k for k, v in self.forecaster_id_to_idx.items()}

    @cached_property
    def problem_idx_to_id(self) -> Dict[int, str]:
        return {v: k for k, v in self.problem_id_to_idx.items()}


def _prepare_pyro_obs(forecast_problems: List[ForecastProblem], n_bins: int = 6, use_empirical_quantiles: bool = False,
                      device: Literal["cpu", "cuda"] = "cpu") -> IRTObs:
    """
    Let there be `n` forecasters and `m` problems. Since not every forecaster has forecasted every problem,
    assume that we have a total of `k` forecasts (events) with k << n * m.
    Take the `forecast_problems` and prepare the following dictionary of the following:
        - `forecaster_ids`: a tensor of shape `(k,)` with the forecaster ids
        - `problem_ids`: a tensor of shape `(k,)` with the problem ids
        - `scores`: a tensor of shape `(k,)` with the scores of the forecasts (discretized from scoring rules)
        - `discretized_scores`: a tensor of shape `(k,)` with the discretized scores of the forecasts
        - `anchor_points`: a tensor of shape `(n_bins,)` with the anchor points of the discretized scores
        - `forecaster_id_to_idx`: a dictionary with the forecaster ids as keys and the indices as values
        - `problem_id_to_idx`: a dictionary with the problem ids as keys and the indices as values
    """
    forecaster_ids, problem_ids, scores = [], [], []
    forecaster_id_to_idx = {}
    problem_id_to_idx = {}

    brier_scoring_rule = BrierScoringRule(negate=False, verbose=False)

    for forecast_problem in forecast_problems:
        # we leverage the fact that for a single problem, `all_probs` have the same shape
        all_probs = []
        correct_option_idx = forecast_problem.correct_option_idx
        for forecast in forecast_problem.forecasts:
            # get the forecaster id and problem id
            forecaster_id, problem_id = forecast.username, forecast_problem.problem_id
            if forecaster_id not in forecaster_id_to_idx:
                forecaster_id_to_idx[forecaster_id] = len(forecaster_id_to_idx)
            if problem_id not in problem_id_to_idx:
                problem_id_to_idx[problem_id] = len(problem_id_to_idx)

            forecaster_ids.append(forecaster_id_to_idx[forecaster_id])
            problem_ids.append(problem_id_to_idx[problem_id])
            all_probs.append(forecast.unnormalized_probs)

        # calculate the scores for this problem
        scores.extend(brier_scoring_rule._score_fn(
            np.array(correct_option_idx), np.array(all_probs), negate=False))

    # discretize the scores
    discretized_indices, bin_edges = _discretize_scoring_rules(
        np.array(scores), n_bins, use_empirical_quantiles)

    # convert to tensors
    return IRTObs(
        forecaster_ids=torch.tensor(
            forecaster_ids, device=device, dtype=torch.long),
        problem_ids=torch.tensor(problem_ids, device=device, dtype=torch.long),
        forecaster_id_to_idx=forecaster_id_to_idx,
        problem_id_to_idx=problem_id_to_idx,
        scores=torch.tensor(scores, device=device, dtype=torch.float),
        discretized_scores=torch.tensor(
            discretized_indices, device=device, dtype=torch.long),
        anchor_points=torch.tensor(
            bin_edges, device=device, dtype=torch.float),
    )


def _discretize_scoring_rules(scores: np.ndarray, n_bins: int = 6, use_empirical_quantiles: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretize the scores into a number of bins.

    Args:
        scores: The scores (from scoring rules)to discretize.
        n_bins: The number of bins to discretize the scores into.
        use_empirical_quantiles: Whether to use empirical quantiles to determine the bins.
            If True, the bins will be determined by the empirical quantiles of the scores.
            If False, the bins will be evenly spaced between 0 and 1.

    Returns:
        A tuple of two arrays, where the first array is the discretized indices of the scores ([0, n_bins - 1]),
        and the second array is the bin edges.
    """
    # make sure all scores are between 0 and 1
    assert np.all(scores >= 0) and np.all(
        scores <= 1), f"Scores must be between 0 and 1, got {[score for score in scores if score < 0 or score > 1]}"

    if use_empirical_quantiles:
        anchor_points = np.quantile(scores, np.linspace(0, 1, n_bins))
        # change the first and last anchor points to 0 and 1
        anchor_points[0] = 0
        anchor_points[-1] = 1
    else:
        anchor_points = np.linspace(0, 1, n_bins)

    dists = np.abs(scores[:, None] - anchor_points[None, :])
    discretized_indices = np.argmin(dists, axis=1)

    return discretized_indices, anchor_points


"""
Some simple in-file tests.
"""


def _test_discretization():
    # test the discretization
    scores = np.array([0, 0.1, 0.1, 0.2, 0.2, 0.8, 0.9])
    discretized_indices, bin_edges = _discretize_scoring_rules(
        scores, n_bins=3, use_empirical_quantiles=False)
    print(discretized_indices)
    print(bin_edges)

    discretized_indices, bin_edges = _discretize_scoring_rules(
        scores, n_bins=3, use_empirical_quantiles=True)
    print(discretized_indices)
    print(bin_edges)


def _test_and_profile_pyro_obs():
    import time
    from pm_rank.data.loaders import GJOChallengeLoader

    predictions_file = "data/raw/all_predictions.json"
    metadata_file = "data/raw/sports_challenge_metadata.json"

    # load the data
    challenge_loader = GJOChallengeLoader(
        predictions_file, metadata_file, challenge_title="GJO Challenge")
    challenge = challenge_loader.load_challenge(
        forecaster_filter=20, problem_filter=20)

    start_time = time.time()
    # prepare the dataset
    dataset = _prepare_pyro_obs(
        challenge.forecast_problems, n_bins=6, use_empirical_quantiles=False, device="cpu")
    end_time = time.time()
    print(
        f"Time taken to prepare the dataset: {end_time - start_time} seconds")

    # print the shape of the dataset
    print(
        f"Shape of the dataset: {dataset['forecaster_ids'].shape}, {dataset['problem_ids'].shape}, {dataset['scores'].shape}, {dataset['discretized_scores'].shape}, {dataset['anchor_points'].shape}")
    print(
        f"Shape of the dataset: {dataset['forecaster_id_to_idx']}, {dataset['problem_id_to_idx']}")


if __name__ == "__main__":
    _test_discretization()
    # _test_and_profile_pyro_obs()
