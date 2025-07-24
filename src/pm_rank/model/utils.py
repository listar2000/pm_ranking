import logging
import sys
from typing import Dict, List, Literal, Any, Tuple
import numpy as np

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

def forecaster_data_to_rankings(forecaster_data: Dict[str, List[float]], include_scores: bool = True,
    ascending: bool = True, aggregate: Literal["mean", "median", "max", "min"] = "mean"):
    """
    Convert the forecaster data to rankings.
    A forecaster data is a dictionary that maps forecaster name to a list of scores.

    Args:
        forecaster_data: a dictionary that maps forecaster name to a list of scores.
        include_scores: whether to include the scores in the rankings.
        ascending: if true, the score is smaller, the better; otherwise, the score is larger, the better.
    Returns:
        A dictionary that maps forecaster name to a list of rankings.
    """
    aggregate_fn = AGGREGATE_FNS[aggregate]
    fitted_scores = {k: aggregate_fn(v) for k, v in forecaster_data.items()}

    sorted_forecasters = sorted(fitted_scores.keys(), key=lambda x: fitted_scores[x], reverse=not ascending)
    forecastor_rankings = {forecaster: rank + 1 for rank, forecaster in enumerate(sorted_forecasters)}
    
    if include_scores:
        return fitted_scores, forecastor_rankings
    else:
        return forecastor_rankings


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


def _format_ranking_table(rankings: dict, scores: dict | None = None, max_rows: int = 20) -> str:
    """
    Format a table of forecaster scores and rankings for logging.
    Shows up to max_rows entries, sorted by rank.
    """
    # Sort by rank
    items = sorted(rankings.items(), key=lambda x: rankings[x[0]])
    header = f"{'Rank':>4}  {'User':<20}"
    if scores is not None:
        header += f"  {'Score':>10}"
    lines = [header, '-' * len(header)]
    for _, (user, rank) in enumerate(items[:max_rows]):
        if scores is not None:
            lines.append(f"{rank:>4}  {user:<20}  {scores[user]:>10.4f}")
        else:
            lines.append(f"{rank:>4}  {user:<20}")
    if len(items) > max_rows:
        lines.append(f"... ({len(items) - max_rows} more)")
    return '\n'.join(lines)


def log_ranking_table(logger: logging.Logger, ranking_result: Tuple[Dict[str, Any], Dict[str, int]] | Dict[str, int], max_rows: int = 20):
    """
    Take in any model's ranking result, which might or might not include scores, and log the table.
    """
    if isinstance(ranking_result, tuple):
        scores, rankings = ranking_result
    else:
        scores, rankings = None, ranking_result
    logger.info(_format_ranking_table(rankings, scores, max_rows))