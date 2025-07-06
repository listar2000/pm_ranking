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
    forecastor_rankings = {forecaster: rank for rank, forecaster in enumerate(sorted_forecasters)}
    
    if include_scores:
        return fitted_scores, forecastor_rankings
    else:
        return forecastor_rankings