"""
Using the generalized Bradley-Terry model to rank the forecasters.

Reference: https://www.jmlr.org/papers/v7/huang06a.html
"""
from collections import OrderedDict
import numpy as np
from typing import Literal, List, Dict, Any, Tuple
from pm_rank.data.base import ForecastProblem
from pm_rank.model.utils import forecaster_data_to_rankings, get_logger, log_ranking_table
import logging


class GeneralizedBT(object):
    def __init__(self, method: Literal["MM", "Elo"] = "MM", num_iter: int = 100, threshold: float = 1e-3, verbose: bool = False):
        self.method = method
        self.num_iter = num_iter
        self.threshold = threshold
        self.verbose = verbose
        self.logger = get_logger(f"pm_rank.model.{self.__class__.__name__}")
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)
        self.logger.info(f"Initialized {self.__class__.__name__} with hyperparam: \n" + \
            f"method={method}, num_iter={num_iter}, threshold={threshold}")

    def fit(self, problems: List[ForecastProblem], include_scores: bool = True) -> \
        Tuple[Dict[str, Any], Dict[str, int]] | Dict[str, int]:
        skills = self._fit_mm(problems, full_trajectory=False)
        result = forecaster_data_to_rankings(skills, include_scores=include_scores, ascending=False) # type: ignore

        if self.verbose:
            log_ranking_table(self.logger, result)
        return result

    def _fit_mm(self, problems: List[ForecastProblem], full_trajectory: bool = False):
        # TODO: currently we assume that each forecaster only makes one forecast per problem
        # Might need to come back and change this if this assumption fails.
        unique_forecasters = OrderedDict()
        num_forecasters = 0
        for problem in problems:
            for forecast in problem.forecasts:
                if forecast.username not in unique_forecasters:
                    unique_forecasters[forecast.username] = num_forecasters
                    num_forecasters += 1

        thetas = np.ones(num_forecasters) # initialize the skills to be all 1
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
                for i, forecast in enumerate(problem.forecasts):
                    user_idx = unique_forecasters[forecast.username]
                    indicators.append(user_idx)

                    W_p_numer[i] = thetas[user_idx] * forecast.correct_prob
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
                    self.logger.info(f"Generalized Bradley-Terry model converged after {t} iterations")
                break
            
        # return a dict of user_id, theta skill
        skills = dict(zip(unique_forecasters.keys(), thetas))

        if full_trajectory:
            return skills, trajectory
        else:
            return skills


# some in-file tests
if __name__ == "__main__":
    from pm_rank.data.loaders import GJOChallengeLoader
    from pm_rank.model.scoring_rule import BrierScoringRule

    predictions_file = "data/raw/all_predictions.json"
    metadata_file = "data/raw/sports_challenge_metadata.json"

    # load the data
    challenge_loader = GJOChallengeLoader(predictions_file, metadata_file, challenge_title="GJO Challenge")
    challenge = challenge_loader.load_challenge(forecaster_filter=20, problem_filter=20)

    gbt_model = GeneralizedBT(method="MM", num_iter=1000)
    fitted_scores, rankings = gbt_model.fit(challenge.forecast_problems, include_scores=True)

    # print the results
    print("GBT rankings:" + "-"*30)
    for forecaster, score in fitted_scores.items(): # type: ignore
        print(f"  {forecaster}: score={score}, rank={rankings[forecaster]}") # type: ignore

    # compare against the Brier score
    brier_scoring_rule = BrierScoringRule()
    fitted_brier_scores, fitted_brier_rankings = brier_scoring_rule.fit(challenge.forecast_problems, include_scores=True)
    print("Brier rankings:" + "-"*30)
    for forecaster, score in fitted_brier_scores.items(): # type: ignore
        print(f"  {forecaster}: score={score}, rank={fitted_brier_rankings[forecaster]}") # type: ignore

    from pm_rank.model.utils import spearman_correlation, kendall_correlation
    print(f"Spearman correlation: {spearman_correlation(rankings, fitted_brier_rankings)}")
    print(f"Kendall correlation: {kendall_correlation(rankings, fitted_brier_rankings)}")

