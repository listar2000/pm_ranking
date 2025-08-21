"""
Average Return Model for Prediction Market Ranking.

This module implements ranking algorithms based on the average returns that forecasters
can achieve from prediction markets. The model calculates expected earnings based on
different risk aversion profiles and market odds.

Note: The forecast problem needs to have the field `odds` in order to use this
model for evaluation.

IMPORTANT DEFINITIONS:

- `implied_probs`: The implied probabilities calculated from the market odds across
  all functions below. In our setting, a $p_i$ implied prob for the outcome $i$ signifies
  that a buy contract will cost $p_i$ dollars and pay out 1 dollar if the outcome is $i$.

- `number of bets`: The number of contracts (see above) to buy for each outcome.
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Iterator, Callable, Literal
from collections import OrderedDict
from dataclasses import dataclass, field
from pm_rank.data.base import ForecastProblem, ForecastChallenge
from pm_rank.model.utils import forecaster_data_to_rankings, get_logger, log_ranking_table, DEFAULT_BOOTSTRAP_CI_CONFIG, BootstrapCIConfig
import logging


@dataclass
class AverageReturnConfig:
    """Configuration class for AverageReturn model parameters.
    
    :param num_money_per_round: Amount of money to bet per round.
    :param risk_aversion: Risk aversion parameter between 0 and 1.
        - 0: Risk neutral
        - 1: Log risk averse  
        - 0 < x < 1: Intermediate risk aversion levels
    :param use_approximate: Whether to use the approximate CRRA betting strategy.
    :param break_tie_by_uniform: When edges are all the same, whether to break tie 
        by spending uniform money on each leg. Only effective when use_approximate is True.
    :param use_binary_reduction: Whether to use the binary reduction strategy.
    """
    num_money_per_round: int = 1
    risk_aversion: float = 0.0
    use_approximate: bool = False
    break_tie_by_uniform: bool = True
    use_binary_reduction: bool = False
    # need this trick since otherwise the default object is mutable.
    bootstrap_ci_config: BootstrapCIConfig = field(default_factory=lambda: DEFAULT_BOOTSTRAP_CI_CONFIG)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not (0 <= self.risk_aversion <= 1):
            raise ValueError(f"risk_aversion must be between 0 and 1, but got {self.risk_aversion}")
    
    def __getitem__(self, key):
        """Allow dict-like access to config parameters."""
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """Allow dict-like setting of config parameters."""
        setattr(self, key, value)
    
    def get(self, key, default=None):
        """Get config parameter with default value."""
        return getattr(self, key, default)
    
    def keys(self):
        """Return config parameter names."""
        return self.__dataclass_fields__.keys()
    
    def items(self):
        """Return config parameter name-value pairs."""
        return [(k, getattr(self, k)) for k in self.keys()]
    
    @classmethod
    def default(cls) -> 'AverageReturnConfig':
        """Create a default configuration."""
        return cls()


def _get_risk_neutral_bets(forecast_probs: np.ndarray, implied_probs: np.ndarray) -> np.ndarray:
    """Calculate the number of bets to each option that a risk-neutral investor would make.

    From simple calculation, we know that in this case the investor would "all-in" to the 
    outcome with the largest `edge`, i.e. where `forecast_probs - implied_probs` is the largest.

    :param forecast_probs: A (n x d) numpy array of forecast probabilities for n forecasters and d options.
    :param implied_probs: A (d,) numpy array of implied probabilities for d options. This might have shape 
    (n, d) in certain cases, e.g. when we are using the binary reduction strategy.

    :returns: The number of bets to each option that a risk-neutral investor would make.
              Shape (n, d) where n is number of forecasters, d is number of options.
    """
    n, d = forecast_probs.shape
    # Calculate the edge for each option and each forecaster
    edges = forecast_probs / implied_probs  # shape (n, d)
    edge_max = np.argmax(edges, axis=1)  # shape (n,)
    # Calculate the number of contracts to buy for each forecaster

    if implied_probs.shape == (d,):
        bet_values = 1 / implied_probs[edge_max]  # shape (n,)
    else:
        bet_values = 1 / implied_probs[np.arange(n), edge_max]  # shape (n,)
    # Create a (n, d) one-hot vector for the bets
    bets_one_hot = np.zeros((n, d))
    bets_one_hot[np.arange(n), edge_max] = bet_values

    return bets_one_hot


def _get_risk_averse_log_bets(forecast_probs: np.ndarray, implied_probs: np.ndarray) -> np.ndarray:
    """Calculate the number of bets to each option that a log-risk-averse investor would make.

    From simple calculation, we know that no matter the implied probs, the log-risk-averse investor
    would bet proportionally to its own forecast probabilities.

    :param forecast_probs: A (n x d) numpy array of forecast probabilities for n forecasters and d options.
    :param implied_probs: A (d,) numpy array of implied probabilities for d options.

    :returns: The number of bets to each option that a log-risk-averse investor would make.
              Shape (n, d) where n is number of forecasters, d is number of options.
    """
    return forecast_probs / implied_probs  # shape (n, d)


def _get_risk_generic_crra_bets(forecast_probs: np.ndarray, implied_probs: np.ndarray, risk_aversion: float) -> np.ndarray:
    """Calculate the number of bets to each option that an investor with a certain CRRA utility 
    (defined by the risk_aversion parameter) would make.

    This function implements the Constant Relative Risk Aversion (CRRA) utility function
    to determine optimal betting strategies for different risk aversion levels.

    :param forecast_probs: A (n x d) numpy array of forecast probabilities for n forecasters and d options.
    :param implied_probs: A (d,) numpy array of implied probabilities for d options.
    :param risk_aversion: A float between 0 and 1 representing the risk aversion parameter.
                         - 0: Risk neutral (equivalent to _get_risk_neutral_bets)
                         - 1: Log risk averse (equivalent to _get_risk_averse_log_bets)
                         - 0 < x < 1: Intermediate risk aversion levels

    :returns: The number of bets to each option for the given risk aversion level.
              Shape (n, d) where n is number of forecasters, d is number of options.

    :raises AssertionError: If implied_probs shape doesn't match the number of options.
    """
    d = forecast_probs.shape[1]
    assert implied_probs.shape == (d,), \
        f"implied_probs must have shape (d,), but got {implied_probs.shape}"

    # Calculate the unnormalized fraction (shape (n, d))
    unnormalized_frac = implied_probs ** (1 - 1 / risk_aversion) * \
        forecast_probs ** (1 / risk_aversion)
    # Normalize the fraction (shape (n, d)) of total money
    normalized_frac = unnormalized_frac / \
        np.sum(unnormalized_frac, axis=1, keepdims=True)
    # Turn the fraction into the actual number of $1 bets
    return normalized_frac / implied_probs  # shape (n, d)


def _get_risk_generic_crra_bets_approximate(
    forecast_probs: np.ndarray,
    implied_probs:  np.ndarray,
    risk_aversion:  float,
    break_tie_by_uniform: bool = True,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Allocate a fixed 1-dollar budget across *overlapping* binary legs of a market for a CRRA utility U(w) = w^(1-γ) / (1-γ).
    Returns the *number of contracts* to buy on each leg (= dollars spent / leg price).

    :param forecast_probs: A (n x d) numpy array of forecast probabilities for n forecasters and d options.
    :param implied_probs: A (d,) numpy array of implied probabilities for d options. 
        This might have shape (n, d) in certain cases, e.g. when we are using the binary reduction strategy.
    :param risk_aversion: A float between 0 and 1 representing the risk aversion parameter.
    :param eps: A float representing the numerical floor to avoid division by zero when p is 0 or 1.

    :returns: The number of bets to each option for the given risk aversion level.
              Shape (n, d) where n is number of forecasters, d is number of options.
    """
    n, d = forecast_probs.shape
    assert implied_probs.shape == (d,) or implied_probs.shape == (n, d), "implied_probs must be shape (d,) or (n, d)"
    assert 0.0 <= risk_aversion <= 1.0, "risk_aversion must be in [0, 1]"

    implied_probs = implied_probs.astype(float).clip(eps, 1.0 - eps)   # m_i
    contracts = np.zeros((n, d))
    γ = risk_aversion

    for k in range(n):
        if implied_probs.shape == (n, d):
            m = implied_probs[k]
        else:
            m = implied_probs
        # preprocess this forecaster's numbers
        p = forecast_probs[k].astype(float).clip(eps, 1.0 - eps)  # p_{k,i}
        a = p / m - 1.0                                           # edge a_i
        b = p * (1.0 - p) / m**2                                  # variance b_i

        # handle special case, if all the edges are the same, we break tie by spending uniform money on each leg
        if break_tie_by_uniform and np.allclose(a, a[0]):
            per_leg_money = 1.0 / d
            contracts[k] = per_leg_money / m
            continue

        # risk-neutral (γ → 0)
        if γ < eps:
            idx_star = int(np.argmax(a))          # best (maybe negative) edge
            contracts[k, idx_star] = 1.0 / m[idx_star]
            continue

        # collect positive-edge legs
        pos_mask = a > 0
        if not np.any(pos_mask):
            # all edges ≤ 0 → forced to spend on the least-bad leg
            idx_star = int(np.argmax(a))
            contracts[k, idx_star] = 1.0 / m[idx_star]
            continue

        # sort positive-edge legs by descending edge
        idx_sorted = np.argsort(-a[pos_mask])
        pos_idx    = np.where(pos_mask)[0][idx_sorted]

        # cumulative sums needed for λ
        inv_b_cum    = 0.0
        a_over_b_cum = 0.0

        # active set that currently satisfies the water-filling condition
        active = []

        for t, j in enumerate(pos_idx):
            inv_b_cum    += 1.0 / b[j]
            a_over_b_cum += a[j] / b[j]
            active.append(j)

            # candidate water level
            lam = (a_over_b_cum - γ) / inv_b_cum

            # look-ahead: will the next edge still be ≥ λ ?
            next_is_ok = (
                t == len(pos_idx) - 1          # no next leg
                or a[pos_idx[t + 1]] <= lam    # next edge below λ
            )

            if next_is_ok:
                # compute dollar stakes for the active set
                x = np.zeros(d)
                for j_act in active:
                    stake = (a[j_act] - lam) / (γ * b[j_act])
                    if stake > 0:
                        x[j_act] = stake

                # numerical safety: ensure the sum is strictly positive
                total = x.sum()
                if total <= eps:
                    # fallback: shove the whole dollar into the top edge leg
                    j_best = active[0]
                    x[j_best] = 1.0
                else:
                    x /= total   # force ∑ x_i = 1 exactly

                contracts[k] = x / m      # convert $ → #contracts
                break

        else:
            # Failsafe (shouldn’t happen): use the single best positive edge
            j_best = pos_idx[0]
            contracts[k, j_best] = 1.0 / m[j_best]

    return contracts


class AverageReturn:
    """Average Return Model for ranking forecasters based on their expected market returns.

    This class implements a ranking algorithm that evaluates forecasters based on how much
    money they could earn from prediction markets using different risk aversion strategies.
    The model calculates expected returns for each forecaster and ranks them accordingly.
    """

    def __init__(self, num_money_per_round: int = None, risk_aversion: float = None, 
                 use_approximate: bool = None, break_tie_by_uniform: bool = None,
                 use_binary_reduction: bool = None, verbose: bool = False, 
                 config: AverageReturnConfig = None, bootstrap_ci_config: BootstrapCIConfig = DEFAULT_BOOTSTRAP_CI_CONFIG):
        """Initialize the AverageReturn model.

        :param num_money_per_round: Amount of money to bet per round (default: 1).
        :param risk_aversion: Risk aversion parameter between 0 and 1 (default: 0.0).
        :param use_approximate: Whether to use the approximate CRRA betting strategy (default: False).
        :param break_tie_by_uniform: When the edges are all the same, 
            whether to break tie by spending uniform money on each leg. Only effective when use_approximate is True (default: True).
        :param use_binary_reduction: Whether to use the binary reduction strategy (default: False).
        :param verbose: Whether to enable verbose logging (default: False).
        :param config: Configuration object containing model parameters. If provided, individual parameters are ignored.

        :raises ValueError: If risk_aversion is not between 0 and 1.
        """
        # If config is provided, use it; otherwise create from individual parameters
        if config is not None:
            final_config = config
        else:
            # Use individual parameters or their defaults
            final_config = AverageReturnConfig(
                num_money_per_round=1 if num_money_per_round is None else num_money_per_round,
                risk_aversion=0.0 if risk_aversion is None else risk_aversion,
                use_approximate=False if use_approximate is None else use_approximate,
                break_tie_by_uniform=True if break_tie_by_uniform is None else break_tie_by_uniform,
                use_binary_reduction=True if use_binary_reduction is None else use_binary_reduction,
                bootstrap_ci_config=bootstrap_ci_config
            )
        
        # Deconstruct config into individual attributes to avoid self.config.xxx usage
        self.num_money_per_round = final_config.num_money_per_round
        self.risk_aversion = final_config.risk_aversion
        self.use_approximate = final_config.use_approximate
        self.break_tie_by_uniform = final_config.break_tie_by_uniform
        self.use_binary_reduction = final_config.use_binary_reduction
        self.bootstrap_ci_config = final_config.bootstrap_ci_config
        self.verbose = verbose
        
        self.logger = get_logger(f"pm_rank.model.{self.__class__.__name__}")
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)
        self.logger.info(f"Initialized {self.__class__.__name__} with config: \n" +
                         f"num_money_per_round={self.num_money_per_round}, risk_aversion={self.risk_aversion} \n" +
                         f"use_approximate={self.use_approximate}, break_tie_by_uniform={self.break_tie_by_uniform}, use_binary_reduction={self.use_binary_reduction}")

        # determine the process_problem_fn based on the use_binary_reduction flag
        self.process_problem_fn = self._process_problem_with_binary_reduction if self.use_binary_reduction else self._process_problem

    def _process_problem(self, problem: ForecastProblem, forecaster_data: Dict[str, List[float]]) -> np.ndarray:
        """Process a single problem and update forecaster_data with earnings.

        This method calculates the expected earnings for each forecaster based on their
        predictions and the actual outcome, then updates the forecaster_data dictionary.

        :param problem: A ForecastProblem instance containing the problem data and forecasts.
        :param forecaster_data: Dictionary mapping usernames to lists of earnings.

        :note: This method only processes problems that have odds data available.
        """
        if not problem.has_odds:
            return

        # Concatenate the forecast probs for all forecasters
        forecast_probs = np.array(
            [forecast.unnormalized_probs for forecast in problem.forecasts])
        # Concatenate the implied probs for all forecasters
        implied_probs = np.array(problem.odds)

        # Check shape consistency
        assert forecast_probs.shape[1] == implied_probs.shape[0], \
            f"forecast probs and implied probs must have the same shape, but got {forecast_probs.shape} and {implied_probs.shape}"

        if self.use_approximate:
            bets = _get_risk_generic_crra_bets_approximate(forecast_probs, implied_probs, self.risk_aversion, self.break_tie_by_uniform)
        else:
            if self.risk_aversion == 0:
                bets = _get_risk_neutral_bets(forecast_probs, implied_probs)
            elif self.risk_aversion == 1:
                bets = _get_risk_averse_log_bets(forecast_probs, implied_probs)
            else:
                bets = _get_risk_generic_crra_bets(
                    forecast_probs, implied_probs, self.risk_aversion)

        # check that bets have value close to 1
        assert np.allclose(bets @ implied_probs, 1.0)
        earnings = np.sum(bets[:, problem.correct_option_idx] * self.num_money_per_round, axis=1)

        # Update forecaster data with earnings
        for i, forecast in enumerate(problem.forecasts):
            username = forecast.username
            if username not in forecaster_data:
                forecaster_data[username] = []
            # we will weight the earnings by the forecast weight for this event.
            forecaster_data[username].append(earnings[i] * forecast.weight)

        outcomes = np.zeros_like(forecast_probs)
        outcomes[:, problem.correct_option_idx] = 1
        return {"bets": bets, "effective_outcomes": outcomes, "choose_yes_contract": np.ones_like(bets).astype(bool)}

    def _process_problem_with_binary_reduction(self, problem: ForecastProblem, forecaster_data: Dict[str, List[float]]) -> np.ndarray:
        """Process a single problem and update forecaster_data with earnings.
        
        The main difference between this helper method and the `_process_problem` method is that
        this method will reduce the options/markets within a problem to individual binary markets, so that we now consider a
        two-level optimization problem for optimal decision making.

        1. On the first (inner) level, for each binary market i, let the implied prob be q_i and the forecast prob be p_i -- which
        corresponds to the belief that market i will be realized. On the contrary, we also allow buying the `No` option, in which
        case the forecast prob will be 1 - p_i and the implied prob will be 1 - q_i.

        The problem then boils down to choosing whether to buy the `Yes` option or the `No` option for each binary market. And the
        solution is that we buy the `Yes` option if the edge `p_i / q_i` is greater than `(1 - p_i) / (1 - q_i)`. Otherwise, we buy
        the `No` option.

        2. On the second (outer) level, we still preserve the previous way of calculating how much of the `num_money_per_round` to 
        spend on each option. However, this time, instead of using `p_i / q_i` as the edge for each option, we will use whether we
        will buy the `Yes` option or the `No` option for each binary market, i.e. we take the edge to be the maximum between
        `p_i / q_i` and `(1 - p_i) / (1 - q_i)`.

        :returns: A (n, d) numpy array of booleans, indicating whether to buy the `Yes` option or the `No` option for each binary market.
        """
        if not problem.has_odds or not problem.has_no_odds:
            return

        forecast_probs = np.array(
            [forecast.unnormalized_probs for forecast in problem.forecasts]) # shape (n, d)
        # Concatenate the implied probs for all forecasters
        implied_probs = np.array(problem.odds) # shape (d,)
        implied_no_probs = np.array(problem.no_odds) # shape (d,)
        
        # Step 1: calculate the per-outcome YES & NO bet edges
        yes_edges = forecast_probs / implied_probs # shape (n, d)
        no_edges = (1 - forecast_probs) / implied_no_probs # shape (n, d)

        # If any forecaster in this problem is the market-baseline, we need to handle it separately
        market_baseline_mask = np.array([forecast.username == "market-baseline" for forecast in problem.forecasts])
        if np.any(market_baseline_mask):
            market_baseline_idx = np.argmax(market_baseline_mask)
            # adjust the no_edge
            no_edges[market_baseline_idx, :] = (1 - forecast_probs[market_baseline_idx, :]) / (1 - implied_probs[market_baseline_idx])
        
        # replace the original forecast_probs with the max of yes_edges and no_edges
        effective_forecast_probs = forecast_probs.copy() # shape (n, d)
        effective_implied_probs = np.tile(implied_probs.copy(), (forecast_probs.shape[0], 1)) # shape (d,) -> (n, d) by expanding the dimension
        # doing 1 - x to the forecast_probs and implied_probs when yes_edges < no_edges
        effective_forecast_probs[no_edges > yes_edges] = 1 - effective_forecast_probs[no_edges > yes_edges]
        effective_implied_probs[no_edges > yes_edges] = np.tile(implied_no_probs, (forecast_probs.shape[0], 1))[no_edges > yes_edges]

        # Step 2: use the older way to incorporate risk aversion and do the outside (approximate) solution
        if self.use_approximate:
            bets = _get_risk_generic_crra_bets_approximate(effective_forecast_probs, effective_implied_probs, self.risk_aversion, self.break_tie_by_uniform)

            if np.any(market_baseline_mask):
                # we will use risk_aversion = 1 to calculate the bets for the market-baseline
                market_baseline_bets = _get_risk_generic_crra_bets_approximate(
                    effective_forecast_probs[market_baseline_idx, :].reshape(1, -1), effective_implied_probs[market_baseline_idx, :].reshape(1, -1), \
                        0, self.break_tie_by_uniform)
                bets[market_baseline_idx, :] = market_baseline_bets.squeeze()
        else:
            if self.risk_aversion == 0:
                bets = _get_risk_neutral_bets(effective_forecast_probs, effective_implied_probs)
            elif self.risk_aversion == 1:
                bets = _get_risk_averse_log_bets(effective_forecast_probs, effective_implied_probs)
            else:
                bets = _get_risk_generic_crra_bets(effective_forecast_probs, effective_implied_probs, self.risk_aversion)

        assert np.allclose(np.sum(bets * effective_implied_probs, axis=1), 1.0)

        # Step 3: calculate the total earnings. Remember that any prob that we flipped will have the realization also inverted
        effective_outcomes = np.zeros_like(forecast_probs) # shape (n, d)
        effective_outcomes[:, problem.correct_option_idx] = 1
        # invert the outcomes when yes_edges < no_edges
        effective_outcomes[no_edges > yes_edges] = 1 - effective_outcomes[no_edges > yes_edges]

        effective_earnings = np.sum(effective_outcomes * bets * self.num_money_per_round, axis=1)

        # update the forecaster data with the effective earnings
        for i, forecast in enumerate(problem.forecasts):
            username = forecast.username
            if username not in forecaster_data:
                forecaster_data[username] = []
            # we will weight the earnings by the forecast weight for this event.
            forecaster_data[username].append(effective_earnings[i] * forecast.weight)

        return {"bets": bets, "effective_outcomes": effective_outcomes, "choose_yes_contract": (yes_edges > no_edges)}

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
                self.process_problem_fn(problem, forecaster_data)

            # Generate rankings for this batch
            # TODO: support bootstrap CI for streaming fit as well.
            batch_results[key] = forecaster_data_to_rankings(
                forecaster_data, include_scores=include_scores, ascending=False, aggregate="mean"
            )
            if self.verbose:
                log_ranking_table(self.logger, batch_results[key])

        return batch_results

    def fit(self, problems: List[ForecastProblem], include_scores: bool = True, include_bootstrap_ci: bool = False, include_per_problem_info: bool = False) -> \
            Tuple[Dict[str, Any], Dict[str, int]] | Dict[str, int]:
        """Fit the average return model to the given problems.

        This method processes all problems at once and returns the final rankings
        based on average returns across all problems.

        :param problems: List of ForecastProblem instances to process.
        :param include_scores: Whether to include scores in the results (default: True).
        :param include_bootstrap_ci: Whether to include bootstrap confidence intervals in the results (default: False).
        :param include_per_problem_info: Whether to include per-problem info in the results (default: False).

        :returns: Ranking results, either as a tuple of (scores, rankings) or just rankings.
                  If include_per_problem_info is True, returns a tuple of (scores, rankings, per_problem_info).
        """
        forecaster_data = {}
        if include_per_problem_info:
            per_problem_info = []

        for problem in problems:
            process_info = self.process_problem_fn(problem, forecaster_data)
            if include_per_problem_info:
                for i, forecast in enumerate(problem.forecasts):
                    # TODO: make this into a HOOK where the caller can customize the info
                    info = {
                        "forecast_id": forecast.forecast_id,
                        "username": forecast.username,
                        "problem_title": problem.title,
                        "problem_id": problem.problem_id,
                        "problem_category": problem.category,
                        "score": forecaster_data[forecast.username][-1],
                        "probs": forecast.unnormalized_probs,
                        "bets": process_info["bets"][i].tolist(),
                        "effective_outcomes": process_info["effective_outcomes"][i].tolist(),
                        "choose_yes_contract": process_info["choose_yes_contract"][i].tolist()
                    }
                    # add `submission_id` if it exists; TODO: remove this hard-coded check
                    if hasattr(forecast, "submission_id"):
                        info["submission_id"] = forecast.submission_id
                    per_problem_info.append(info)

        result = forecaster_data_to_rankings(
            forecaster_data, include_scores=include_scores, include_bootstrap_ci=include_bootstrap_ci, ascending=False, aggregate="mean", bootstrap_ci_config=self.bootstrap_ci_config)
        if self.verbose:
            log_ranking_table(self.logger, result)
        
        return (*result, per_problem_info) if include_per_problem_info else result

    def fit_stream(self, problem_iter: Iterator[List[ForecastProblem]], include_scores: bool = True) -> \
            Dict[int, Tuple[Dict[str, Any], Dict[str, int]] | Dict[str, int]]:
        """Fit the model to streaming problems and return incremental results.

        This method processes problems as they arrive and returns rankings after each batch,
        allowing for incremental analysis of forecaster performance.

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
        """Fit the model to streaming problems with timestamps and return incremental results.

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
        """Fit the average return model to the given problems by category.

        This method processes all problems at once and returns the final rankings
        based on average returns across all problems.

        :param problems: List of ForecastProblem instances to process.
        :param include_scores: Whether to include scores in the results (default: True).
        :param stream_with_timestamp: Whether to stream problems with timestamps (default: False).
        :param stream_increment_by: The increment by which to stream problems (default: "day").
        :param min_bucket_size: The minimum number of problems to include in a bucket (default: 1).
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

            