import logging
import sys
from typing import Dict, List, Literal, Tuple, Callable
import numpy as np
from tqdm import tqdm
import random
from pydantic import BaseModel, Field


def get_logger(name: str = "pm_rank.model"):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

AGGREGATE_FNS = {
    "mean": np.mean,
    "median": np.median,
    "max": np.max,
    "min": np.min
}

class BootstrapCIConfig(BaseModel):
    """Configuration for bootstrap confidence interval computation.
    
    This configuration class defines parameters for computing bootstrap confidence
    intervals used in forecaster ranking evaluations.
    
    :param num_bootstrap_samples: The number of bootstrap samples to draw (default: 1000).
    :param bootstrap_ci_level: The confidence level for the bootstrap confidence interval (default: 0.95).
    :param random_seed: The random seed for reproducible bootstrap sampling. Set to None for random seeding (default: 42).
    :param symmetric: Whether to use symmetric confidence intervals (default: True).
    """
    num_bootstrap_samples: int = Field(
        default=1000, description="The number of bootstrap samples to draw."
    )
    bootstrap_ci_level: float = Field(
        default=0.95, description="The confidence level for the bootstrap confidence interval."
    )
    random_seed: int | None = Field(
        default=42, description="The random seed for reproducible bootstrap sampling. Set to None for random seeding."
    )
    symmetric: bool = Field(
        default=True, description="Whether to use symmetric confidence intervals."
    )


# Default configuration instance
DEFAULT_BOOTSTRAP_CI_CONFIG = BootstrapCIConfig()

# TODO: (1) implement a multi-processing version of this. Currently we do a vanilla version.
# (2) do a "clustering-based" bootstrap to correctly account for inter-problem correlations.
# Reference: https://github.com/EleutherAI/lm-evaluation-harness/blob/3bc7cc8a72c66bac8d5b830cb3ccec9a5f691b12/lm_eval/api/metrics.py
def _bootstrap_ci_single_process(forecaster_data: Dict[str, List[float]], aggregate_fn: Callable, bootstrap_ci_config: BootstrapCIConfig):
    """
    Bootstrap the forecaster data to get the confidence intervals.
    """
    num_bootstrap_samples = bootstrap_ci_config.num_bootstrap_samples
    bootstrap_ci_level = bootstrap_ci_config.bootstrap_ci_level
    
    if bootstrap_ci_config.random_seed is not None:
        random.seed(bootstrap_ci_config.random_seed)

    fitted_scores_over_resamples = {}
    for _ in tqdm(range(num_bootstrap_samples), desc="Bootstrapping"):
        resampled_forecaster_data = {k: random.choices(v, k=len(v)) for k, v in forecaster_data.items()}
        fitted_scores = {k: aggregate_fn(v) for k, v in resampled_forecaster_data.items()}
        for k, v in fitted_scores.items():
            if k not in fitted_scores_over_resamples:
                fitted_scores_over_resamples[k] = []
            fitted_scores_over_resamples[k].append(v)
    
    # compute the confidence intervals
    confidence_intervals = {}

    for k, v in fitted_scores_over_resamples.items():
        if bootstrap_ci_config.symmetric:
            point_estimate = aggregate_fn(forecaster_data[k])
            # we build a symmetric confidence interval around the point estimate
            deviations = np.abs(np.array(v) - point_estimate)
            deviations = np.sort(deviations)

            idx = int(np.ceil(bootstrap_ci_level * len(deviations)))
            confidence_intervals[k] = [point_estimate - deviations[idx], point_estimate + deviations[idx]]
        else:
            # we simply use the (1 - bootstrap_ci_level) / 2 and (1 + bootstrap_ci_level) / 2 percentile
            confidence_intervals[k] = np.percentile(v, [(1 - bootstrap_ci_level) / 2, (1 + bootstrap_ci_level) / 2])

    return confidence_intervals


def forecaster_data_to_rankings(forecaster_data: Dict[str, List[float]], include_scores: bool = True, include_bootstrap_ci: bool = False,
    ascending: bool = True, aggregate: Literal["mean", "median", "max", "min"] = "mean", bootstrap_ci_config: BootstrapCIConfig = DEFAULT_BOOTSTRAP_CI_CONFIG):
    """
    Convert the forecaster data to rankings.
    A forecaster data is a dictionary that maps forecaster name to a list of scores.

    Args:
        forecaster_data: a dictionary that maps forecaster name to a list of scores.
        include_scores: whether to include the scores in the rankings.
        include_bootstrap_ci: whether to include the bootstrap confidence intervals in the rankings.
        ascending: if true, the score is smaller, the better; otherwise, the score is larger, the better.
    Returns:
        A dictionary that maps forecaster name to a list of rankings.
    """
    aggregate_fn = AGGREGATE_FNS[aggregate]
    fitted_scores = {k: aggregate_fn(v) for k, v in forecaster_data.items()}

    sorted_forecasters = sorted(fitted_scores.keys(), key=lambda x: fitted_scores[x], reverse=not ascending)
    forecastor_rankings = {forecaster: rank + 1 for rank, forecaster in enumerate(sorted_forecasters)}

    return_dicts = [forecastor_rankings] if not include_scores else [fitted_scores, forecastor_rankings]
    
    if include_bootstrap_ci:
        bootstrap_cis = _bootstrap_ci_single_process(forecaster_data, aggregate_fn, bootstrap_ci_config)
        return_dicts.append(bootstrap_cis)

    return (*return_dicts,)

"""
Diagnostic/Analysis functions
"""
def _prepare_ranks(rank_dict_a: Dict[str, int], rank_dict_b: Dict[str, int]):
    common_keys = list(set(rank_dict_a) & set(rank_dict_b))
    common_keys.sort()
    ranks_a = np.array([rank_dict_a[k] for k in common_keys])
    ranks_b = np.array([rank_dict_b[k] for k in common_keys])
    return ranks_a, ranks_b


def spearman_correlation(rank_dict_a: Dict[str, int], rank_dict_b: Dict[str, int]) -> float:
    """
    Compute the Spearman correlation between two rankings.
    Reference: https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    """
    x, y = _prepare_ranks(rank_dict_a, rank_dict_b)
    x_mean = x.mean()
    y_mean = y.mean()
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    return numerator / denominator if denominator != 0 else 0.0


def kendall_correlation(rank_dict_a: Dict[str, int], rank_dict_b: Dict[str, int]) -> float:
    """
    Compute the Kendall correlation between two rankings.
    Reference: https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
    """
    x, y = _prepare_ranks(rank_dict_a, rank_dict_b)
    n = len(x)
    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            concordant += (dx * dy) > 0
            discordant += (dx * dy) < 0

    total_pairs = n * (n - 1) / 2
    return (concordant - discordant) / total_pairs if total_pairs != 0 else 0.0


def _format_ranking_table(
    rankings: dict,
    scores: dict | None = None,
    bootstrap_cis: dict | None = None,
    max_rows: int = 25
) -> str:
    """
    Format a table of forecaster scores and rankings for logging.
    Shows up to max_rows entries, sorted by rank.
    """
    # Wider columns: Rank (6), User (40), Score (14), Bootstrap CI (14 each)
    RANK_WIDTH = 6
    USER_WIDTH = 40
    SCORE_WIDTH = 14
    CI_WIDTH = 14

    items = sorted(rankings.items(), key=lambda x: rankings[x[0]])
    header = f"{'Rank':>{RANK_WIDTH}}  {'User':<{USER_WIDTH}}"
    if scores is not None:
        header += f"  {'Score':>{SCORE_WIDTH}}"
    if bootstrap_cis is not None:
        header += f"  {'CI Lower':>{CI_WIDTH}}  {'CI Upper':>{CI_WIDTH}}"
    lines = [header, '-' * len(header)]
    for _, (user, rank) in enumerate(items[:max_rows]):
        if scores is not None and bootstrap_cis is not None:
            lines.append(
                f"{rank:>{RANK_WIDTH}}  {user:<{USER_WIDTH}}  {scores[user]:>{SCORE_WIDTH}.4f}  {bootstrap_cis[user][0]:>{CI_WIDTH}.4f}  {bootstrap_cis[user][1]:>{CI_WIDTH}.4f}"
            )
        elif scores is not None:
            lines.append(
                f"{rank:>{RANK_WIDTH}}  {user:<{USER_WIDTH}}  {scores[user]:>{SCORE_WIDTH}.4f}"
            )
        else:
            lines.append(
                f"{rank:>{RANK_WIDTH}}  {user:<{USER_WIDTH}}"
            )
    if len(items) > max_rows:
        lines.append(f"... ({len(items) - max_rows} more)")
    return '\n'.join(lines)


def log_ranking_table(logger: logging.Logger, ranking_result: Tuple | Dict[str, int], max_rows: int = 25):
    """
    Take in any model's ranking result, which might or might not include scores, and log the table.
    """
    if isinstance(ranking_result, tuple):
        if len(ranking_result) == 2:
            scores, rankings = ranking_result
            bootstrap_ci = None
        elif len(ranking_result) == 3:
            scores, rankings, bootstrap_cis = ranking_result
        else:
            raise ValueError(f"Invalid ranking result: {ranking_result}")
    else:
        scores, rankings = None, ranking_result
        bootstrap_cis = None

    logger.info("\n" + _format_ranking_table(rankings, scores, bootstrap_cis, max_rows=max_rows))