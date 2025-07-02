"""
Use proper scoring rules to rank the forecastors.

Reference: https://www.cis.upenn.edu/~aaroth/courses/slides/agt17/lect23.pdf
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Iterator, Dict, Tuple, Any
from data.base import ForecastProblem
from tqdm import tqdm


class ScoringRule(ABC):
    """
    Abstract base class for scoring rules.
    """

    @abstractmethod
    def _score_fn(self, correct_probs: np.ndarray, all_probs: np.ndarray) -> np.ndarray:
        """ implement the scoring function for the rule """
        pass

    def fit(self, problems: List[ForecastProblem]) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """ fit the scoring rule to the problems """
        forecaster_data = {}
        
        for problem in problems:
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
            scores = self._score_fn(correct_probs, all_probs)
            # attribute the scores to the forecasters
            for username, score in zip(usernames, scores):
                # if username == 'smitch15':
                #     print(f"Score for {username} in problem {problem.problem_id}: {score}")
                forecaster_data[username].append(score)
        
        # Batch process all probabilities for each forecaster
        fitted_scores = {}
        for forecaster, scores in forecaster_data.items():
            fitted_scores[forecaster] = np.mean(scores)
        
        # Calculate rankings more efficiently
        sorted_forecasters = sorted(fitted_scores.keys(), key=lambda x: fitted_scores[x], reverse=True)
        forecastor_rankings = {forecaster: rank for rank, forecaster in enumerate(sorted_forecasters)}

        return fitted_scores, forecastor_rankings

    def fit_stream(self, problem_iter: Iterator[List[ForecastProblem]]) -> Dict[int, Tuple[Dict[str, Any], Dict[str, int]]]:
        """ return the fitted scores and rankings as problems are streamed in instead of all given at once """
        # Initialize running state
        forecaster_data = {}
        batch_results = {}            # Results for each batch
        batch_id = 0
        
        for batch in tqdm(problem_iter, desc=f"Fitting {self.__class__.__name__}"):
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

            # batch process the scores
            fitted_scores = {}
            for forecaster, scores in forecaster_data.items():
                fitted_scores[forecaster] = np.mean(scores)
            
            # Calculate current rankings
            sorted_forecasters = sorted(fitted_scores.keys(), key=lambda x: fitted_scores[x], reverse=True)

            forecastor_rankings = {forecaster: rank + 1 for rank, forecaster in enumerate(sorted_forecasters)}
            
            # Store results for this batch
            batch_results[batch_id] = (fitted_scores, forecastor_rankings)
            batch_id += 1
        
        return batch_results


class LogScoringRule(ScoringRule):
    """
    Log scoring rule.
    """
    def __init__(self, clip_prob: float = 0.01):
        """ initialize the scoring rule """
        self.clip_prob = clip_prob

    def _score_fn(self, correct_probs: np.ndarray, all_probs: np.ndarray) -> np.ndarray:
        """ implement the scoring function for the rule """
        return np.log(np.maximum(correct_probs, self.clip_prob))


class BrierScoringRule(ScoringRule):
    """
    Brier scoring rule.
    """
    def _score_fn(self, correct_probs: np.ndarray, all_probs: np.ndarray) -> np.ndarray:
        """ implement the scoring function for the rule """
        # correct_probs is 1D with shape (n,), all_probs is 2D with shape (n, k)
        # (1) we obtain (n,) correct_scores
        correct_scores = (1 - correct_probs) ** 2 - correct_probs ** 2
        # (2) we obtain (n,) incorrect scores
        incorrect_scores = np.sum(all_probs ** 2, axis=1)
        # (3) we obtain (n,) scores
        scores = correct_scores + incorrect_scores
        # (4) negate the result since higher scores are better
        return -scores


class SphericalScoringRule(ScoringRule):
    """
    Spherical scoring rule.
    """
    def _score_fn(self, correct_probs: np.ndarray, all_probs: np.ndarray) -> np.ndarray:
        """ formula: r_j / sum_i r_i where r_j is the correct probability of the j-th option """
        correct_scores = correct_probs / np.linalg.norm(all_probs, axis=1)
        return correct_scores


if __name__ == "__main__":
    # check the implementation of the scoring rules
    correct_probs = np.array([0, 0.5, 1])
    all_probs = np.array([[0, 0.4, 0.6], [0.2, 0.5, 0.3], [0, 0, 1]])
    print(LogScoringRule()._score_fn(correct_probs, all_probs))
    print(BrierScoringRule()._score_fn(correct_probs, all_probs))
    print(SphericalScoringRule()._score_fn(correct_probs, all_probs))