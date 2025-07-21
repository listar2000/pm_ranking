"""
Use how much money the forecaster can make from the prediction market to rank.
Note: the forecast problem need to have the field `odds` in order to use this
problem for evaluation.

IMPORTANT DEFINITIONS:
- `implied_probs`: the implied probabilities calculated from the market odds across
    all functions below. In our setting, a `p_i` implied prob for the outcome `i` significes
    that a `buy contract` will cost `p_i` dollars and pay out 1 dollar if the outcome is `i`.
- `number of bets` means the number of contracts (see above) to buy for each outcome.
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Iterator
from pm_rank.data.base import ForecastProblem
from pm_rank.model.utils import forecaster_data_to_rankings

def _get_risk_neutral_bets(forecast_probs: np.ndarray, implied_probs: np.ndarray) -> np.ndarray:
    """
    Calculate the number of bets to each option that a risk-neutral investor would make.
    From simple calculation, we know that in this case the investor would "all-in" to the 
    outcome with the largest `edge`, i.e. where `forecast_probs - implied_probs` is the largest.

    Args:
        forecast_probs: a (n x d) numpy array of forecast probabilities for n forecasters and d options.
        implied_probs: a (d,) numpy array of implied probabilities for d options.
    Returns:
        The number of bets to each option that a risk-neutral investor would make.
    """
    n, d = forecast_probs.shape
    # calculate the edge for each option and each forecaster
    edges = forecast_probs - implied_probs # shape (n, d)
    edge_max = np.argmax(edges, axis=1) # shape (n,)
    # calculate the number of contracts to buy for each forecaster
    bet_values = 1 / implied_probs[edge_max] # shape (n,)
    # create a (n, d) one-hot vector for the bets
    bets_one_hot = np.zeros((n, d))
    bets_one_hot[np.arange(n), edge_max] = bet_values
    return bets_one_hot

def _get_risk_averse_log_bets(forecast_probs: np.ndarray, implied_probs: np.ndarray) -> np.ndarray:
    """
    Calculate the number of bets to each option that a log-risk-averse investor would make.
    From simple calculation, we know that no matter the implied probs, the log-risk-averse investor
    would bet proportionally to its own forecast probabilities.

    Args:
        forecast_probs: a (n x d) numpy array of forecast probabilities for n forecasters and d options.
        implied_probs: a (d,) numpy array of implied probabilities for d options.
    Returns:
        The number of bets to each option that a log-risk-averse investor would make.
    """    
    return forecast_probs / implied_probs # shape (n, d)

def _get_risk_generic_crra_bets(forecast_probs: np.ndarray, implied_probs: np.ndarray, risk_aversion: float) -> np.ndarray:
    """
    Calculate the number of bets to each option that an investor with a certain CRRA utility 
    (defined by the risk_aversion parameter) would make.
    """
    d = forecast_probs.shape[1]
    assert implied_probs.shape == (d,), \
        f"implied_probs must have shape (d,), but got {implied_probs.shape}"
    # calculate the unnormalized fraction (shape (n, d))
    unnormalized_frac = implied_probs ** (1 - 1 / risk_aversion) * forecast_probs ** (1 / risk_aversion)
    # normalize the fraction (shape (n, d)) of total money
    normalized_frac = unnormalized_frac / np.sum(unnormalized_frac, axis=1, keepdims=True)
    # turn the fraction into the actual number of $1 bets
    return normalized_frac / implied_probs # shape (n, d)

class MarketEarning(object):
    def __init__(self, num_money_per_round: int = 1, risk_aversion: float = 0.0):
        self.num_money_per_round = num_money_per_round
        assert risk_aversion >= 0 and risk_aversion <= 1, \
            f"risk_aversion must be between 0 and 1, but got {risk_aversion}"
        self.risk_aversion = risk_aversion

    def _process_problem(self, problem: ForecastProblem, forecaster_data: Dict[str, List[float]]) -> None:
        """Process a single problem and update forecaster_data with earnings."""
        if not problem.has_odds:
            return
        
        # concatenate the forecast probs for all forecasters
        forecast_probs = np.array([forecast.probs for forecast in problem.forecasts])
        # concatenate the implied probs for all forecasters
        implied_probs = np.array(problem.odds)
        # check shape
        assert forecast_probs.shape[1] == implied_probs.shape[0], \
            f"forecast probs and implied probs must have the same shape, but got {forecast_probs.shape} and {implied_probs.shape}"
        
        if self.risk_aversion == 0:
            bets = _get_risk_neutral_bets(forecast_probs, implied_probs)
        elif self.risk_aversion == 1:
            bets = _get_risk_averse_log_bets(forecast_probs, implied_probs)
        else:
            bets = _get_risk_generic_crra_bets(forecast_probs, implied_probs, self.risk_aversion)
        
        correct_idx = problem.options.index(problem.correct_option)
        earnings = bets[:, correct_idx] * self.num_money_per_round
        for i, forecast in enumerate(problem.forecasts):
            username = forecast.username
            if username not in forecaster_data:
                forecaster_data[username] = []
            forecaster_data[username].append(earnings[i])

    def fit(self, problems: List[ForecastProblem], include_scores: bool = True) -> \
        Tuple[Dict[str, Any], Dict[str, int]] | Dict[str, int]:
        """Fit the market earning model to the problems."""
        forecaster_data = {}
        for problem in problems:
            self._process_problem(problem, forecaster_data)
        
        return forecaster_data_to_rankings(forecaster_data, include_scores=include_scores, ascending=False, aggregate="mean")

    def fit_stream(self, problem_iter: Iterator[List[ForecastProblem]], include_scores: bool = True) -> \
        Dict[int, Tuple[Dict[str, Any], Dict[str, int]] | Dict[str, int]]:
        """Return the fitted scores and rankings as problems are streamed in instead of all given at once."""
        forecaster_data = {}
        batch_results = {}
        batch_id = 0
        
        for batch in problem_iter:
            for problem in batch:
                self._process_problem(problem, forecaster_data)

            batch_results[batch_id] = forecaster_data_to_rankings(forecaster_data, include_scores=include_scores, ascending=False, aggregate="mean")
            batch_id += 1
        
        return batch_results