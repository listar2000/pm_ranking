import pandas as pd
import numpy as np
from typing import Literal, Dict, Optional
from pm_rank.model.calibration import _bin_stats, _calculate_ece
from pm_rank.nightly.bootstrap import compute_bootstrap_ci
from tqdm import tqdm


DEFAULT_BOOTSTRAP_CONFIG = {
    'num_samples': 1000,
    'ci_level': 0.9,
    'num_se': None,
    'random_seed': 42,
    'show_progress': True
}


def add_individualized_market_baselines_to_scores(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add individualized market baseline scores for each forecaster at aggregation time.
    
    This function takes per-forecast scores (e.g., from compute_brier_score or compute_average_return_neutral)
    and creates "{forecaster}-market-baseline" entries by filtering the market-baseline scores to only
    the (event_ticker, round) combinations where each forecaster participated.
    
    This is efficient because it reuses the already-computed market-baseline scores rather than
    creating duplicate prediction rows.
    
    Args:
        result_df: DataFrame with columns (forecaster, event_ticker, round, weight, <score_col>)
                   Must contain a 'market-baseline' forecaster.
    
    Returns:
        DataFrame with added "{forecaster}-market-baseline" rows for each real forecaster.
    """
    if 'market-baseline' not in result_df['forecaster'].values:
        return result_df  # No market-baseline to work with
    
    # Get the market-baseline scores
    market_baseline_scores = result_df[result_df['forecaster'] == 'market-baseline'].copy()
    
    # Get unique real forecasters (excluding market-baseline and any existing individualized baselines)
    real_forecasters = result_df[
        ~result_df['forecaster'].str.contains('-market-baseline', na=False) & 
        (result_df['forecaster'] != 'market-baseline')
    ]['forecaster'].unique()
    
    individualized_baselines = []
    
    for forecaster in tqdm(real_forecasters, desc="Adding individualized market baselines"):
        # Get this forecaster's (event_ticker, round) combinations
        forecaster_data = result_df[result_df['forecaster'] == forecaster]
        forecaster_keys = forecaster_data[['event_ticker', 'round']].drop_duplicates()
        
        # Filter market-baseline scores to only these combinations
        individualized = market_baseline_scores.merge(
            forecaster_keys,
            on=['event_ticker', 'round'],
            how='inner'
        ).copy()
        
        # Also copy the weight from the original forecaster's data
        # This ensures proper weighting when aggregating
        weight_map = forecaster_data.set_index(['event_ticker', 'round'])['weight'].to_dict()
        individualized['weight'] = individualized.apply(
            lambda row: weight_map.get((row['event_ticker'], row['round']), row['weight']),
            axis=1
        )
        
        # Set the forecaster name
        individualized['forecaster'] = f'{forecaster}-market-baseline'
        
        individualized_baselines.append(individualized)
    
    # Concatenate all individualized baselines with original result_df
    if individualized_baselines:
        result = pd.concat([result_df] + individualized_baselines, ignore_index=True)
    else:
        result = result_df
    
    return result


def rank_forecasters_by_score(result_df: pd.DataFrame, normalize_by_round: bool = False, 
                              score_col: str = None, ascending: bool = None,
                              bootstrap_config: Optional[Dict] = None,
                              add_individualized_baselines: bool = False) -> pd.DataFrame:
    """
    Return a rank_df with columns (forecaster, rank, score).
    
    Args:
        result_df: DataFrame containing forecaster scores
        normalize_by_round: If True, downweight by the number of rounds per (forecaster, event_ticker) group
                           (ignored for ECE scores which are already aggregated)
        score_col: Name of the score column to rank by. If None, auto-detects from {'brier_score', 'average_return', 'ece_score'}
        ascending: Whether lower scores are better (True for Brier/ECE, False for returns). If None, auto-detects.
        bootstrap_config: Optional dict with bootstrap parameters for CI estimation:
            - num_samples: Number of bootstrap samples (default: 1000)
            - ci_level: Confidence level (default: 0.95)
            - num_se: Number of standard errors for CI bounds (default: None, uses ci_level)
            - random_seed: Random seed for reproducibility (default: 42)
            - show_progress: Whether to show progress bar (default: True)
            Only supported for 'brier_score' and 'average_return', not 'ece_score'.
        add_individualized_baselines: If True, create "{forecaster}-market-baseline" entries for each
            forecaster by filtering market-baseline scores to their participated (event_ticker, round)
            combinations. Only works for Brier score and average return (not ECE/Sharpe).
            Requires 'market-baseline' forecaster to be present in result_df.
    
    Returns:
        DataFrame with rank as index and columns (forecaster, score).
        If bootstrap_config is provided, also includes (se, lower, upper) columns.
    """
    df = result_df.copy()
    
    # Auto-detect score column if not provided
    if score_col is None:
        if 'brier_score' in df.columns:
            score_col = 'brier_score'
        elif 'average_return' in df.columns:
            score_col = 'average_return'
        elif 'ece_score' in df.columns:
            score_col = 'ece_score'
        elif 'sharpe_ratio' in df.columns:
            score_col = 'sharpe_ratio'
        else:
            raise ValueError("Could not find score column. Please specify 'score_col' parameter.")
    
    # Auto-detect ascending if not provided
    if ascending is None:
        # Lower is better for Brier score and ECE score, higher is better for average return
        ascending = (score_col in ['brier_score', 'ece_score'])
    
    # Special handling for ECE scores and Sharpe ratio scores: they're already aggregated per forecaster
    if score_col in ['ece_score', 'sharpe_ratio']:
        if bootstrap_config is not None:
            raise ValueError(f"Bootstrap CI is not supported for {score_col} scores (already aggregated)")
        
        # These scores are already computed per forecaster, just need to rank and format
        forecaster_scores = df[['forecaster', score_col]].copy()
        forecaster_scores.columns = ['forecaster', 'score']
        
        # Rank forecasters (ascending=True means lower score = better rank)
        forecaster_scores['rank'] = forecaster_scores['score'].rank(method='min', ascending=ascending).astype(int)
        
        # Sort by rank and select required columns, then set rank as index
        rank_df = forecaster_scores[['forecaster', 'rank', 'score']].sort_values('rank')
        rank_df = rank_df.set_index('rank')[['forecaster', 'score']]
        
        return rank_df

    # Optionally add individualized market baselines before aggregation
    if add_individualized_baselines:
        df = add_individualized_market_baselines_to_scores(df)
    
    # For other metrics (Brier, average return), perform weighted aggregation
    if normalize_by_round:
        # For each (forecaster, event_ticker) group, downweight by number of rounds
        # First, count the number of rounds per (forecaster, event_ticker)
        round_counts = df.groupby(['forecaster', 'event_ticker']).size().reset_index(name='round_count')
        
        # Merge back to get round_count for each row
        df = df.merge(round_counts, on=['forecaster', 'event_ticker'])
        
        # Adjust weights by dividing by round_count
        df['adjusted_weight'] = df['weight'] / df['round_count']
    else:
        df['adjusted_weight'] = df['weight']
    
    # Calculate weighted average score for each forecaster
    # Group by forecaster and compute weighted mean
    forecaster_scores = df.groupby('forecaster').apply(
        lambda group: np.dot(group[score_col], group['adjusted_weight']) / np.sum(group['adjusted_weight']),
        include_groups=False
    ).reset_index(name='score')
    
    # Count predictions per forecaster
    prediction_counts = df.groupby('forecaster').size()
    forecaster_scores['# predictions'] = forecaster_scores['forecaster'].map(prediction_counts)
    
    ci_col_name = None
    if bootstrap_config is not None:
        # Compute bootstrap confidence intervals if requested
        ci_col_name = f'{bootstrap_config["ci_level"] * 100}% ci'
        standard_errors, confidence_intervals = compute_bootstrap_ci(
            df[['forecaster', score_col]].copy(),
            score_col,
            df['adjusted_weight'].values,
            bootstrap_config
        )
        
        # Add SE and CI columns to forecaster_scores
        # forecaster_scores['se'] = forecaster_scores['forecaster'].map(standard_errors)
        # forecaster_scores['lower'] = forecaster_scores['forecaster'].map(lambda f: confidence_intervals[f][0])
        # forecaster_scores['upper'] = forecaster_scores['forecaster'].map(lambda f: confidence_intervals[f][1])
        forecaster_scores[ci_col_name] = \
            forecaster_scores['forecaster'].map(lambda f: f"±{(confidence_intervals[f][1] - confidence_intervals[f][0]) / 2:.4f}")
    
    # Rank forecasters (ascending=True means lower score = better rank)
    forecaster_scores['rank'] = forecaster_scores['score'].rank(method='min', ascending=ascending).astype(int)
    
    # Sort by rank and select required columns, then set rank as index
    if bootstrap_config is not None:
        rank_df = forecaster_scores[['forecaster', 'rank', 'score', ci_col_name, '# predictions']].sort_values('rank')
        rank_df = rank_df.set_index('rank')[['forecaster', 'score', ci_col_name, '# predictions']]
    else:
        rank_df = forecaster_scores[['forecaster', 'rank', 'score', '# predictions']].sort_values('rank')
        rank_df = rank_df.set_index('rank')[['forecaster', 'score', '# predictions']]
    
    return rank_df


def add_market_baseline_predictions(forecasts: pd.DataFrame, reference_forecaster: str = None, use_both_sides: bool = False) -> pd.DataFrame:
    """
    We turn the forecasts from a certain forecaster into market baseline predictions.
    If use_both_sides is True, we will add the market baseline predictions for both YES and NO sides.

    Args:
        forecasts: DataFrame with columns (forecaster, event_ticker, round, prediction, outcome, weight)
        reference_forecaster: The forecaster to use as the reference for the market baseline predictions
        use_both_sides: If True, we will add the market baseline predictions for both YES and NO sides
    """
    if reference_forecaster is None:
        # if no reference forecaster is provided, we take the union of all forecasts from all forecasters
        market_baseline_forecasts = forecasts.groupby(['event_ticker', 'round'], as_index=False).first()
    else:
        market_baseline_forecasts = forecasts[forecasts['forecaster'] == reference_forecaster].copy()

    market_baseline_forecasts['forecaster'] = 'market-baseline'

    def turn_odds_to_prediction(row: pd.Series) -> np.ndarray:
        odds, no_odds = row['odds'], row['no_odds']
        if use_both_sides:
            return np.array([(odds[i] + (1 - no_odds[i])) / 2.0 for i in range(len(odds))])
        else:
            return np.array([odds[i] for i in range(len(odds))])

    market_baseline_forecasts['prediction'] = market_baseline_forecasts.apply(turn_odds_to_prediction, axis=1)

    # concat market_baseline_forecasts with forecasts (take care of the pd index as well)
    forecasts = pd.concat([forecasts, market_baseline_forecasts]).reset_index(drop=True)
    return forecasts


def compute_brier_score(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Brier score for the forecasts. We will proceed by grouping by `event_ticker`, as each resulting group
    will have the same shape (i.e. number of markets), and we can manually construct a np matrix to accelerate the computation.

    The result will be a DataFrame containing (forecaster, event_ticker, round, time_rank, brier_score) 

    Args:
        forecasts: DataFrame with columns (forecaster, event_ticker, round, prediction, outcome, weight)
    """
    result_df = forecasts.copy()
    
    # Initialize brier_score column
    result_df['brier_score'] = np.nan
    
    # Group by event_ticker and process each group
    for _, event_group in result_df.groupby('event_ticker'):
        # Get indices for this group
        group_indices = event_group.index
        
        # prepare the predictions matrix with shape (num_group_elements, num_markets)
        prediction_matrix = np.stack(event_group['prediction'].values)
        
        # prepare the outcome vector with shape (num_markets,) since it's the same for all group elements
        outcome_vector = event_group['outcome'].iloc[0]
        
        # Calculate Brier score: mean squared difference between predictions and outcomes
        # Brier score = mean((prediction - outcome)^2) for each forecast
        squared_diffs = (prediction_matrix - outcome_vector) ** 2
        brier_scores = np.mean(squared_diffs, axis=1)
        
        # Assign brier scores back to the result dataframe
        result_df.loc[group_indices, 'brier_score'] = brier_scores
    
    # Select only the required columns
    result_df = result_df[['forecaster', 'event_ticker', 'weight', 'round', 'brier_score']]
    return result_df


def compute_average_return_neutral(forecasts: pd.DataFrame, num_money_per_round: float = 1.0, 
                                   spread_market_even: bool = False) -> pd.DataFrame:
    """
    Calculate the average return for forecasters with risk-neutral utility using binary reduction strategy.
    
    This implementation uses:
    - Risk-neutral betting (all-in on best edge, or spread evenly)
    - Binary reduction (can bet YES or NO on each market)
    - Approximate CRRA betting strategy for risk_aversion=0
    
    For each market, we compare:
    - YES edge: forecast_prob / yes_odds
    - NO edge: (1 - forecast_prob) / no_odds
    
    If spread_market_even is False (default):
        We choose the better edge for each market, then allocate all money to the market with the best edge.
    
    If spread_market_even is True:
        We spread the budget evenly across all markets (budget/m per market), and bet on the better edge
        (YES or NO) in each market.
    
    Args:
        forecasts: DataFrame with columns (forecaster, event_ticker, round, prediction, outcome, odds, no_odds, weight)
        num_money_per_round: Amount of money to bet per round (default: 1.0)
        spread_market_even: If True, spread budget evenly across markets instead of all-in on best market
    
    Returns:
        DataFrame with columns (forecaster, event_ticker, round, weight, average_return)
    """
    result_df = forecasts.copy()
    result_df['average_return'] = np.nan
    
    # Group by event_ticker and process each event
    for _, event_group in result_df.groupby('event_ticker'):
        group_indices = event_group.index
        
        # Stack predictions and odds into matrices: shape (n_forecasters, n_markets)
        forecast_probs = np.stack(event_group['prediction'].values)  # p_i
        implied_yes_probs = np.stack(event_group['odds'].values)    # q_i (YES odds)
        implied_no_probs = np.stack(event_group['no_odds'].values)  # q'_i (NO odds)
        
        # Outcome is the same for all forecasters in this event
        outcome_vector = event_group['outcome'].iloc[0]  # shape (n_markets,)
        
        # Step 1: Calculate edges for YES and NO bets on each market
        # YES edge: p_i / q_i (ratio of forecast prob to YES price)
        # NO edge: (1 - p_i) / q'_i (ratio of forecast NO prob to NO price)
        yes_edges = forecast_probs / implied_yes_probs
        no_edges = (1 - forecast_probs) / implied_no_probs
        
        # Step 2: Choose YES or NO for each market based on which has better edge
        choose_yes = yes_edges > no_edges  # boolean mask: shape (n_forecasters, n_markets)
        
        # Create effective probabilities and prices based on choice
        effective_forecast_probs = np.where(choose_yes, forecast_probs, 1 - forecast_probs)
        effective_implied_probs = np.where(choose_yes, implied_yes_probs, implied_no_probs)
        
        # Step 3: Risk-neutral betting strategy
        n_forecasters, n_markets = forecast_probs.shape
        bets = np.zeros((n_forecasters, n_markets))
        
        if spread_market_even:
            # Spread budget evenly across all markets
            money_per_market = num_money_per_round / n_markets
            # For each market, buy contracts with the allocated money at the effective price
            # Number of contracts = money_per_market / effective_price
            bets = money_per_market / effective_implied_probs
        else:
            # All-in on market with best edge
            # For risk-neutral, we find the market with max edge and bet everything there
            effective_edges = effective_forecast_probs / effective_implied_probs
            best_market_idx = np.argmax(effective_edges, axis=1)  # shape (n_forecasters,)
            
            for i in range(n_forecasters):
                market_idx = best_market_idx[i]
                # Number of contracts = money / price
                bets[i, market_idx] = num_money_per_round / effective_implied_probs[i, market_idx]
        
        # Step 4: Calculate effective outcomes (flip outcome if we chose NO)
        effective_outcomes = np.where(choose_yes, 
                                     outcome_vector[np.newaxis, :],  # broadcast outcome
                                     1 - outcome_vector[np.newaxis, :])

        # Sanity Check: the sum of money spent should be equal to num_money_per_round
        assert np.allclose(np.sum(bets * effective_implied_probs, axis=1), num_money_per_round)
        
        # Step 5: Calculate earnings
        # Earnings = sum over markets of (bets * effective_outcomes * num_money_per_round / num_money_per_round)
        # Since bets already incorporates the money amount, we don't multiply by num_money_per_round again
        # Each contract pays out 1 if it wins, 0 otherwise
        earnings = np.sum(bets * effective_outcomes, axis=1)
        
        # Assign earnings to result dataframe
        result_df.loc[group_indices, 'average_return'] = earnings
    
    # Select only the required columns
    result_df = result_df[['forecaster', 'event_ticker', 'round', 'weight', 'average_return']]
    return result_df


def compute_calibration_ece(forecasts: pd.DataFrame, num_bins: int = 10, 
                           strategy: Literal["uniform", "quantile"] = "uniform",
                           weight_event: bool = True, return_details: bool = False) -> pd.DataFrame:
    """
    Calculate the Expected Calibration Error (ECE) for each forecaster.
    
    The ECE measures how well-calibrated a forecaster's probability predictions are.
    For perfectly calibrated predictions, when a forecaster predicts probability p,
    the actual outcome should occur with frequency p.
    
    This function combines two types of weights:
    1. Prediction-level weight: from the 'weight' column (assigned by weight_fn in data loading)
    2. Market-level weight: either uniform (1.0) or inverse of number of markets per prediction
    
    The final weight for each market probability is: prediction_weight * market_weight
    
    Args:
        forecasts: DataFrame with columns (forecaster, event_ticker, round, prediction, outcome, weight)
        num_bins: Number of bins to use for discretization (default: 10)
        strategy: Strategy for discretization, either "uniform" or "quantile" (default: "uniform")
        weight_event: If True, weight each market by 1/num_markets within each prediction.
                     If False, all markets are weighted equally (default: True)
        return_details: If True, return the details of the ECE calculation for each forecaster. Useful for plotting.
    Returns:
        DataFrame with columns (forecaster, ece_score) containing the ECE for each forecaster
    """
    # Prepare data structures to collect probabilities and outcomes for each forecaster
    forecaster_data = {}
    
    for _, row in forecasts.iterrows():
        forecaster = row['forecaster']
        prediction_weight = row['weight']
        prediction_probs = row['prediction']  # numpy array of probabilities
        outcome_labels = row['outcome']  # numpy array of 0/1 outcomes
        
        # Initialize forecaster data if not already present
        if forecaster not in forecaster_data:
            forecaster_data[forecaster] = {
                'probs': [],
                'labels': [],
                'weights': []
            }
        
        # Calculate market-level weight
        num_markets = len(prediction_probs)
        if weight_event:
            # Each market within this prediction gets weight = prediction_weight / num_markets
            market_weight = prediction_weight / num_markets
        else:
            # Each market gets the full prediction weight
            market_weight = prediction_weight
        
        # Add each market's probability and outcome with combined weight
        forecaster_data[forecaster]['probs'].extend(prediction_probs)
        forecaster_data[forecaster]['labels'].extend(outcome_labels)
        forecaster_data[forecaster]['weights'].extend([market_weight] * num_markets)
    
    # Calculate ECE for each forecaster
    ece_results = []
    ece_details = {}
    
    for forecaster, data in forecaster_data.items():
        probs = data['probs']
        labels = data['labels']
        weights = np.array(data['weights'])
        
        # Normalize weights to sum to the number of samples (required by _bin_stats)
        if weights.sum() == 0:
            print(f"Warning: {forecaster} has no weights. Skipping ECE calculation...")
            continue
        
        weights = weights * len(probs) / weights.sum()
        
        # Calculate bin statistics using the helper function from the old API
        bin_centers, bin_widths, conf, acc, counts = _bin_stats(
            probs, labels, weights.tolist(), num_bins, strategy
        )
        
        # Calculate ECE using the helper function from the old API
        ece_score = _calculate_ece(conf, acc, counts, len(probs))
        
        ece_results.append({
            'forecaster': forecaster,
            'ece_score': ece_score
        })

        if return_details:
            ece_details[forecaster] = {
                'ece_score': ece_score,
                'bin_centers': bin_centers,
                'bin_widths': bin_widths,
                'conf': conf,
                'acc': acc,
                'counts': counts
            }
    
    # Create result DataFrame
    result_df = pd.DataFrame(ece_results)
    
    # Sort by ECE score (lower is better)
    result_df = result_df.sort_values('ece_score').reset_index(drop=True)
    
    if return_details:
        return result_df, ece_details
    else:
        return result_df


def compute_sharpe_ratio(average_return_results: pd.DataFrame, baseline_return: float = 1.0, 
                         normalize_by_round: bool = False) -> pd.DataFrame:
    """
    Calculate the Sharpe ratio for each forecaster.
    
    The Sharpe ratio is defined as: E[R - R_b] / std(R - R_b), where R is the return 
    and R_b is the baseline return (typically 1.0 for break-even).
    
    Args:
        average_return_results: DataFrame with columns (forecaster, event_ticker, round, weight, average_return)
        baseline_return: The baseline return to subtract from the average return (default: 1.0 for break-even)
        normalize_by_round: If True, first average returns within each (forecaster, event_ticker) group,
                           then calculate Sharpe ratio across events. This prevents events with more
                           rounds from dominating the calculation. (default: False)
    
    Returns:
        DataFrame with columns (forecaster, sharpe_ratio, mean_excess_return, std_excess_return)
        sorted by sharpe_ratio in descending order
    """
    df = average_return_results.copy()
    
    if normalize_by_round:
        # Step 1: For each (forecaster, event_ticker), compute weighted average return across all rounds
        # This gives us one return value per event per forecaster
        def weighted_mean(group):
            return np.average(group['average_return'], weights=group['weight'])
        
        event_returns = df.groupby(['forecaster', 'event_ticker']).apply(
            weighted_mean, include_groups=False
        ).reset_index(name='event_return')
        
        # Step 2: Calculate Sharpe ratio for each forecaster using event-level returns
        sharpe_results = []
        for forecaster in event_returns['forecaster'].unique():
            forecaster_data = event_returns[event_returns['forecaster'] == forecaster]
            returns = forecaster_data['event_return'].values
            
            # Calculate excess returns
            excess_returns = returns - baseline_return
            
            # Calculate mean and std of excess returns
            mean_excess = np.mean(excess_returns)
            std_excess = np.std(excess_returns, ddof=1)  # Use sample std (ddof=1)
            
            # Calculate Sharpe ratio (handle case where std is 0)
            if std_excess > 0:
                sharpe = mean_excess / std_excess
            else:
                sharpe = 0.0 if mean_excess == 0 else (np.inf if mean_excess > 0 else -np.inf)
            
            sharpe_results.append({
                'forecaster': forecaster,
                'sharpe_ratio': sharpe,
                'mean_excess_return': mean_excess,
                'std_excess_return': std_excess
            })
    else:
        # Calculate Sharpe ratio directly from all (event, round) pairs
        sharpe_results = []
        for forecaster in df['forecaster'].unique():
            forecaster_data = df[df['forecaster'] == forecaster]
            returns = forecaster_data['average_return'].values
            
            # Calculate excess returns
            excess_returns = returns - baseline_return
            
            # Calculate mean and std of excess returns
            mean_excess = np.mean(excess_returns)
            std_excess = np.std(excess_returns, ddof=1)  # Use sample std (ddof=1)
            
            # Calculate Sharpe ratio (handle case where std is 0)
            if std_excess > 0:
                sharpe = mean_excess / std_excess
            else:
                sharpe = 0.0 if mean_excess == 0 else (np.inf if mean_excess > 0 else -np.inf)
            
            sharpe_results.append({
                'forecaster': forecaster,
                'sharpe_ratio': sharpe,
                'mean_excess_return': mean_excess,
                'std_excess_return': std_excess
            })
    
    result_df = pd.DataFrame(sharpe_results)
    
    # Sort by Sharpe ratio (descending - higher is better)
    result_df = result_df.sort_values('sharpe_ratio', ascending=False).reset_index(drop=True)
    
    return result_df
        

"""
Helper functions to implement the generic category/streaming functionalities.
"""
def _iterate_over_categories(forecasts: pd.DataFrame) -> dict:
    categories = forecasts['category'].unique()
    for category in categories:
        category_forecasts = forecasts[forecasts['category'] == category]
        yield category, category_forecasts
    yield "overall", forecasts


def _stream_over_time(forecasts: pd.DataFrame, stream_every: int) -> dict:
    """
    Stream the forecasts each `stream_every` days, counting from the beginning.
    Yields cumulative forecasts up to each time window.
    
    Args:
        forecasts: DataFrame with 'close_time' column (string format: "2025-09-17T17:55:00+00:00")
        stream_every: Number of days between each stream window
    
    Yields:
        Tuple of (time_label, forecasts_up_to_time) for each window
    """
    # Create a copy and convert close_time to datetime
    forecasts = forecasts.copy()
    forecasts['close_time_dt'] = pd.to_datetime(forecasts['close_time'], format='ISO8601')
    
    # Find the earliest and latest close_time
    close_time_beg = forecasts['close_time_dt'].min()
    close_time_end = forecasts['close_time_dt'].max()
    
    # Calculate the number of days between start and end
    total_days = (close_time_end - close_time_beg).days
    
    # Stream forecasts every stream_every days
    current_days = 0
    while current_days <= total_days:
        # Calculate the cutoff time
        cutoff_time = close_time_beg + pd.Timedelta(days=current_days)
        
        # Get all forecasts up to this cutoff time (cumulative)
        stream_forecasts = forecasts[forecasts['close_time_dt'] <= cutoff_time]
        
        # Remove the helper column before yielding
        stream_forecasts = stream_forecasts.drop(columns=['close_time_dt'])
        
        # Create a label for this window
        time_label = str(cutoff_time.date())
        
        yield time_label, stream_forecasts
        
        current_days += stream_every
    
    forecasts.drop(columns=['close_time_dt'], inplace=True)


def compute_ranked_brier_score(forecasts: pd.DataFrame, by_category: bool = False, stream_every: int = -1, \
    normalize_by_round: bool = False, bootstrap_config: Optional[Dict] = None, 
    add_individualized_baselines: bool = False) -> dict:
    """
    Compute the ranked forecasters for the given score function.
    
    Args:
        forecasts: DataFrame with forecast data
        by_category: If True, compute rankings per category
        stream_every: If > 0, compute rankings at time intervals
        normalize_by_round: If True, downweight by number of rounds per (forecaster, event_ticker)
        bootstrap_config: Optional config for bootstrap CI estimation
        add_individualized_baselines: If True, create "{forecaster}-market-baseline" entries for each
            forecaster by filtering market-baseline scores to their participated (event_ticker, round).
            Requires 'market-baseline' forecaster to be present.
    """
    do_stream = stream_every > 0
    if not do_stream and not by_category:
        score = compute_brier_score(forecasts)
        return rank_forecasters_by_score(score, normalize_by_round=normalize_by_round, 
                                         bootstrap_config=bootstrap_config,
                                         add_individualized_baselines=add_individualized_baselines)
    
    if by_category:
        results = {}
        for category, category_forecasts in _iterate_over_categories(forecasts):
            if do_stream:
                results[category] = {}
                for time_label, time_forecasts in tqdm(_stream_over_time(category_forecasts, stream_every=stream_every), desc=f"Calculating Brier score for category {category}"):
                    results[category][time_label] = rank_forecasters_by_score(compute_brier_score(time_forecasts), \
                        normalize_by_round=normalize_by_round, bootstrap_config=bootstrap_config,
                        add_individualized_baselines=add_individualized_baselines)
            else:
                results[category] = rank_forecasters_by_score(compute_brier_score(category_forecasts), \
                    normalize_by_round=normalize_by_round, bootstrap_config=bootstrap_config,
                    add_individualized_baselines=add_individualized_baselines)
        return results
    else:  # do_stream might be True (otherwise handled by above)
        results = {}
        for time_label, time_forecasts in tqdm(_stream_over_time(forecasts, stream_every=stream_every), desc="Calculating overall"):
            results[time_label] = rank_forecasters_by_score(compute_brier_score(time_forecasts), \
                normalize_by_round=normalize_by_round, bootstrap_config=bootstrap_config,
                add_individualized_baselines=add_individualized_baselines)
        return results


def compute_ranked_average_return(forecasts: pd.DataFrame, by_category: bool = False, stream_every: int = -1, \
    spread_market_even: bool = False, num_money_per_round: float = 1.0, normalize_by_round: bool = False, 
    bootstrap_config: Optional[Dict] = None, add_individualized_baselines: bool = False) -> dict:
    """
    Compute the ranked forecasters for the given score function.
    
    Args:
        forecasts: DataFrame with forecast data
        by_category: If True, compute rankings per category
        stream_every: If > 0, compute rankings at time intervals
        spread_market_even: If True, spread budget evenly across markets
        num_money_per_round: Amount to bet per round
        normalize_by_round: If True, downweight by number of rounds per (forecaster, event_ticker)
        bootstrap_config: Optional config for bootstrap CI estimation
        add_individualized_baselines: If True, create "{forecaster}-market-baseline" entries for each
            forecaster by filtering market-baseline scores to their participated (event_ticker, round).
            Requires 'market-baseline' forecaster to be present.
    """
    do_stream = stream_every > 0
    if not do_stream and not by_category:
        score = compute_average_return_neutral(forecasts, spread_market_even=spread_market_even, num_money_per_round=num_money_per_round)
        return rank_forecasters_by_score(score, normalize_by_round=normalize_by_round, 
                                         bootstrap_config=bootstrap_config,
                                         add_individualized_baselines=add_individualized_baselines)
    if by_category:
        results = {}
        for category, category_forecasts in _iterate_over_categories(forecasts):
            if do_stream:
                results[category] = {}
                for time_label, time_forecasts in tqdm(_stream_over_time(category_forecasts, stream_every=stream_every), desc=f"Calculating average return for category {category}"):
                    results[category][time_label] = rank_forecasters_by_score(compute_average_return_neutral(time_forecasts, spread_market_even=spread_market_even, num_money_per_round=num_money_per_round), \
                        normalize_by_round=normalize_by_round, bootstrap_config=bootstrap_config,
                        add_individualized_baselines=add_individualized_baselines)
            else:
                results[category] = rank_forecasters_by_score(compute_average_return_neutral(category_forecasts, spread_market_even=spread_market_even, num_money_per_round=num_money_per_round), \
                    normalize_by_round=normalize_by_round, bootstrap_config=bootstrap_config,
                    add_individualized_baselines=add_individualized_baselines)
        return results
    else:  # do_stream might be True (otherwise handled by above)
        results = {}
        for time_label, time_forecasts in tqdm(_stream_over_time(forecasts, stream_every=stream_every), desc="Calculating overall"):
            results[time_label] = rank_forecasters_by_score(compute_average_return_neutral(time_forecasts, spread_market_even=spread_market_even, num_money_per_round=num_money_per_round), \
                normalize_by_round=normalize_by_round, bootstrap_config=bootstrap_config,
                add_individualized_baselines=add_individualized_baselines)
        return results


if __name__ == "__main__":
    predictions_csv = "slurm/predictions_12_31_to_01_01.csv"  # Your predictions CSV file
    submissions_csv = "slurm/submissions_12_31_to_01_01.csv"  # Your submissions CSV file

    from pm_rank.nightly.data import uniform_weighting, NightlyForecasts
    
    weight_fn = uniform_weighting()
    forecasts = NightlyForecasts.from_prophet_arena_csv(predictions_csv, submissions_csv, weight_fn)

    # from pm_rank.nightly.misc import get_rebalanced_forecasts
    # forecasts = get_rebalanced_forecasts(forecasts, balance_level='event', evenly_balanced=True, random_seed=42)
    # or you can do
    # desired_quota = {"Sports": 0.2, "Entertainment": 0.2, "Politics": 0.2, "Companies": 0.2, "Mentions": 0.2, "Economics": 0.2, "Climate and Weather": 0.2}
    # forecasts = get_rebalanced_forecasts(forecasts, balance_level='event', rebalance_quota=desired_quota, random_seed=42)

    forecasts.data = add_market_baseline_predictions(forecasts.data)

    brier_score = compute_brier_score(forecasts.data)
    rank_df = rank_forecasters_by_score(brier_score, normalize_by_round=True, bootstrap_config=None, add_individualized_baselines=True)
    print(rank_df)
    
    # filter out any row where the '# predictions' column is less than 100 or the forecaster name starts with "agent" or "AI"
    rank_df = rank_df[(rank_df['# predictions'] >= 100) & (~rank_df['forecaster'].str.startswith('agent'))]
    rank_df.to_csv("slurm/rank_df_12_31_to_01_01.csv")

    # Collect/stream the results for every 7 days, and also divide results by category.
    # ranked_brier_score = compute_ranked_brier_score(forecasts.data, by_category=True, stream_every=7, normalize_by_round=True, bootstrap_config=None)
    # ranked_average_return = compute_ranked_average_return(forecasts.data, by_category=True, stream_every=7, spread_market_even=False, num_money_per_round=1.0, normalize_by_round=True, bootstrap_config=None)

    # # Take the second time stamp of category "Sports", which remains a dataframe that's easy to work with.
    # example_sports_streams = ranked_brier_score["Sports"]
    # example_sports_second_stream = example_sports_streams[list(example_sports_streams.keys())[1]]
    # print(example_sports_second_stream)

    # # The streaming result for all the forecasts (no category division) is stored in the "overall" key.
    # example_overall_stream = ranked_brier_score["overall"]
    # example_overall_second_stream = example_overall_stream[list(example_overall_stream.keys())[1]]
    # print(example_overall_second_stream)

    # # If you want to take the overall result WITHOUT categorization or streaming, simply do not specify by_category or stream_every.
    # overall_brier_ranking = compute_ranked_brier_score(forecasts.data, normalize_by_round=True, bootstrap_config=None)
    # print(overall_brier_ranking)