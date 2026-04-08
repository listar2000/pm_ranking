"""
Bootstrap confidence interval calculation for forecaster rankings.

This module provides simplified bootstrap CI estimation for the nightly API,
focusing on symmetric confidence intervals around point estimates.

Key principle: Bootstrap resampling is done SEPARATELY for each forecaster,
sampling with replacement from their own predictions only. This properly
estimates the uncertainty in each forecaster's individual score.

Granularity: The resampling unit depends on the input data. When the input
contains per-market rows (one row per individual market outcome), bootstrap
resamples at the market level. When the input contains per-event rows (one
row per problem/event with scores averaged across markets), bootstrap resamples
at the event level. Market-level resampling is the default and captures both
within-event and across-event variance.

Performance: the inner resampling loop is fully vectorized — for each
forecaster we draw a single ``(num_samples, n_rows)`` index matrix and
compute all bootstrap statistics in one batch of numpy operations, instead
of running a Python-level for loop over ``num_samples`` iterations. This
makes the dominant cost of the nightly recompute small enough that bootstrap
CI is no longer the bottleneck.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple


def compute_bootstrap_ci(
    result_df: pd.DataFrame,
    score_col: str,
    adjusted_weights: np.ndarray,
    bootstrap_config: Dict = None,
    aggregation: str = 'mean',
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    """
    Compute bootstrap confidence intervals for forecaster scores.

    This function performs weighted bootstrap resampling of individual predictions
    to estimate confidence intervals for forecaster scores. It uses symmetric CIs
    around the point estimate.

    IMPORTANT: Resampling is done SEPARATELY for each forecaster. For each
    forecaster we draw an entire ``(num_samples, n_rows)`` index matrix in one
    ``np.random.choice`` call and compute the bootstrap statistic for every
    sample at once via vectorized numpy ops, instead of running a Python loop
    over the bootstrap iterations.

    Args:
        result_df: DataFrame with columns (forecaster, score_col, adjusted_weight)
                  where each row is an individual prediction
        score_col: Name of the score column ('brier_score' or 'average_return')
        adjusted_weights: Array of adjusted weights for each prediction (same length as result_df)
        bootstrap_config: Dictionary with bootstrap parameters:
            - num_samples: Number of bootstrap samples (default: 1000)
            - ci_level: Confidence level (default: 0.95)
            - num_se: Number of standard errors for CI bounds (default: None, uses ci_level)
            - random_seed: Random seed for reproducibility (default: 42)
            - show_progress: Ignored (kept for backwards compatibility — the
              vectorized implementation has no per-iteration progress to show)
        aggregation: Aggregation mode for forecaster scores.
            - 'mean': weighted mean of score_col
            - 'sum': sum of score_col
            - 'roi': sum(score_col) / sum(cost), requiring a 'cost' column in result_df.

    Returns:
        Tuple of (standard_errors, confidence_intervals) where:
        - standard_errors: Dict mapping forecaster -> SE of the score
        - confidence_intervals: Dict mapping forecaster -> (lower, upper) bounds
    """
    # Default configuration
    default_config = {
        'num_samples': 1000,
        'ci_level': 0.95,
        'num_se': None,  # If None, use ci_level for symmetric CI
        'random_seed': 42,
        'show_progress': True,
    }

    if bootstrap_config is None:
        bootstrap_config = {}

    # Merge with defaults
    config = {**default_config, **bootstrap_config}

    num_samples = config['num_samples']
    ci_level = config['ci_level']
    num_se = config['num_se']
    random_seed = config['random_seed']

    if aggregation == 'roi' and 'cost' not in result_df.columns:
        raise ValueError("ROI bootstrap aggregation requires a 'cost' column in result_df")

    # Use a local Generator so concurrent callers don't race on the global RNG.
    rng = np.random.default_rng(random_seed)

    forecasters = result_df['forecaster'].unique()

    # Pre-compute forecaster-specific arrays once.
    forecaster_data: Dict[str, Dict[str, np.ndarray]] = {}
    for forecaster in forecasters:
        forecaster_mask = (result_df['forecaster'] == forecaster).to_numpy()
        forecaster_data[forecaster] = {
            'scores': result_df.loc[forecaster_mask, score_col].to_numpy(),
            'weights': adjusted_weights[forecaster_mask],
        }
        if aggregation == 'roi':
            forecaster_data[forecaster]['costs'] = result_df.loc[forecaster_mask, 'cost'].to_numpy()

    point_estimates: Dict[str, float] = {}
    standard_errors: Dict[str, float] = {}
    confidence_intervals: Dict[str, Tuple[float, float]] = {}

    for forecaster in forecasters:
        data = forecaster_data[forecaster]
        scores = data['scores']
        weights = data['weights']
        n_rows = scores.shape[0]

        # Point estimate from the original (un-resampled) data.
        if n_rows == 0:
            point_estimates[forecaster] = np.nan
            standard_errors[forecaster] = np.nan
            confidence_intervals[forecaster] = (np.nan, np.nan)
            continue

        if aggregation == 'sum':
            point = float(np.sum(scores))
        elif aggregation == 'roi':
            costs = data['costs']
            total_cost = float(np.sum(costs))
            point = float(np.sum(scores) / total_cost) if total_cost > 0 else 0.0
        else:  # 'mean'
            weight_sum_total = float(np.sum(weights))
            point = (
                float(np.average(scores, weights=weights))
                if weight_sum_total > 0
                else float(np.mean(scores))
            )
        point_estimates[forecaster] = point

        # Vectorized resampling: draw ALL bootstrap iterations at once.
        weight_sum = float(np.sum(weights))
        sampling_probs = weights / weight_sum if weight_sum > 0 else None
        sampled_idx = rng.choice(
            n_rows,
            size=(num_samples, n_rows),
            replace=True,
            p=sampling_probs,
        )

        # All gathers and aggregations happen across the rows axis.
        sampled_scores = scores[sampled_idx]  # (num_samples, n_rows)

        if aggregation == 'sum':
            samples = sampled_scores.sum(axis=1)
        elif aggregation == 'roi':
            costs = data['costs']
            sampled_costs = costs[sampled_idx]  # (num_samples, n_rows)
            cost_sums = sampled_costs.sum(axis=1)
            score_sums = sampled_scores.sum(axis=1)
            samples = np.where(cost_sums > 0, score_sums / np.where(cost_sums == 0, 1, cost_sums), 0.0)
        else:  # weighted mean
            sampled_weights = weights[sampled_idx]  # (num_samples, n_rows)
            weight_row_sums = sampled_weights.sum(axis=1)
            # Guard against zero-weight rows (shouldn't happen given weight_sum > 0
            # but the bootstrap could in principle draw all-zero-weight subsets).
            safe_denom = np.where(weight_row_sums == 0, 1, weight_row_sums)
            samples = (sampled_scores * sampled_weights).sum(axis=1) / safe_denom
            samples = np.where(weight_row_sums == 0, np.nan, samples)

        valid_samples = samples[~np.isnan(samples)]
        if valid_samples.size == 0:
            standard_errors[forecaster] = np.nan
            confidence_intervals[forecaster] = (np.nan, np.nan)
            continue

        se = float(np.std(valid_samples, ddof=1)) if valid_samples.size > 1 else 0.0
        standard_errors[forecaster] = se

        if num_se is not None:
            margin = num_se * se
        else:
            deviations = np.abs(valid_samples - point)
            deviations_sorted = np.sort(deviations)
            idx = int(np.ceil(ci_level * deviations_sorted.size)) - 1
            idx = max(0, min(idx, deviations_sorted.size - 1))
            margin = float(deviations_sorted[idx])

        confidence_intervals[forecaster] = (point - margin, point + margin)

    return standard_errors, confidence_intervals
