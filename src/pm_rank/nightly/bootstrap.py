"""
Bootstrap confidence interval calculation for forecaster rankings.

This module provides simplified bootstrap CI estimation for the nightly API,
focusing on symmetric confidence intervals around point estimates.

Key principle: Bootstrap resampling is done SEPARATELY for each forecaster,
sampling with replacement from their own predictions only. This properly
estimates the uncertainty in each forecaster's individual score.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from tqdm import tqdm


def compute_bootstrap_ci(
    result_df: pd.DataFrame,
    score_col: str,
    adjusted_weights: np.ndarray,
    bootstrap_config: Dict = None
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    """
    Compute bootstrap confidence intervals for forecaster scores.
    
    This function performs weighted bootstrap resampling of individual predictions
    to estimate confidence intervals for forecaster scores. It uses symmetric CIs
    around the point estimate.
    
    IMPORTANT: Resampling is done SEPARATELY for each forecaster. At each bootstrap
    iteration, we sample (with replacement) from each forecaster's own predictions
    using their adjusted weights. This properly captures the uncertainty in each
    forecaster's individual score estimate.
    
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
            - show_progress: Whether to show progress bar (default: True)
    
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
        'show_progress': True
    }
    
    if bootstrap_config is None:
        bootstrap_config = {}
    
    # Merge with defaults
    config = {**default_config, **bootstrap_config}
    
    num_samples = config['num_samples']
    ci_level = config['ci_level']
    num_se = config['num_se']
    random_seed = config['random_seed']
    show_progress = config['show_progress']
    
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Get unique forecasters
    forecasters = result_df['forecaster'].unique()
    
    # Store bootstrap samples for each forecaster
    bootstrap_samples = {forecaster: [] for forecaster in forecasters}
    
    # Perform bootstrap resampling - SEPARATELY FOR EACH FORECASTER
    iterator = range(num_samples)
    if show_progress:
        iterator = tqdm(iterator, desc="Bootstrap sampling")
    
    # TODO: implement a multi-processing version of this. Currently we do a vanilla version.
    
    # Pre-compute forecaster-specific data for efficiency
    forecaster_data = {}
    for forecaster in forecasters:
        forecaster_mask = result_df['forecaster'] == forecaster
        forecaster_data[forecaster] = {
            'indices': np.where(forecaster_mask)[0],
            'scores': result_df.loc[forecaster_mask, score_col].values,
            'weights': adjusted_weights[forecaster_mask]
        }
    
    for _ in iterator:
        # For each forecaster, sample from THEIR OWN predictions
        for forecaster in forecasters:
            data = forecaster_data[forecaster]
            n_forecaster_rows = len(data['indices'])
            
            if n_forecaster_rows == 0:
                bootstrap_samples[forecaster].append(np.nan)
                continue
            
            # Normalize weights to sum to 1 for this forecaster's predictions
            sampling_probs = data['weights'] / data['weights'].sum()
            
            # Sample with replacement from this forecaster's predictions
            sampled_indices = np.random.choice(
                n_forecaster_rows, 
                size=n_forecaster_rows, 
                replace=True, 
                p=sampling_probs
            )
            
            # Get the resampled scores and weights for this forecaster
            resampled_scores = data['scores'][sampled_indices]
            resampled_weights = data['weights'][sampled_indices]
            
            # Calculate weighted average score for this forecaster in this resample
            weighted_score = np.average(resampled_scores, weights=resampled_weights)
            bootstrap_samples[forecaster].append(weighted_score)
    
    # Calculate point estimates (original weighted averages)
    point_estimates = {}
    for forecaster in forecasters:
        forecaster_mask = result_df['forecaster'] == forecaster
        forecaster_scores = result_df.loc[forecaster_mask, score_col].values
        forecaster_weights = adjusted_weights[forecaster_mask]
        point_estimates[forecaster] = np.average(forecaster_scores, weights=forecaster_weights)
    
    # Calculate standard errors and confidence intervals
    standard_errors = {}
    confidence_intervals = {}
    
    for forecaster in forecasters:
        samples = np.array([s for s in bootstrap_samples[forecaster] if not np.isnan(s)])
        point_estimate = point_estimates[forecaster]
        
        if len(samples) == 0:
            standard_errors[forecaster] = np.nan
            confidence_intervals[forecaster] = (np.nan, np.nan)
            continue
        
        # Calculate standard error
        se = np.std(samples, ddof=1)
        standard_errors[forecaster] = se
        
        # Calculate symmetric confidence interval
        if num_se is not None:
            # Use specified number of standard errors
            margin = num_se * se
            lower = point_estimate - margin
            upper = point_estimate + margin
        else:
            # Use symmetric CI based on deviations from point estimate
            deviations = np.abs(samples - point_estimate)
            deviations_sorted = np.sort(deviations)
            
            # Find the ci_level percentile of deviations
            idx = int(np.ceil(ci_level * len(deviations_sorted))) - 1
            idx = max(0, min(idx, len(deviations_sorted) - 1))
            margin = deviations_sorted[idx]
            
            lower = point_estimate - margin
            upper = point_estimate + margin
        
        confidence_intervals[forecaster] = (lower, upper)
    
    return standard_errors, confidence_intervals

