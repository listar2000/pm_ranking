from typing import Dict, List, Literal
import numpy as np

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