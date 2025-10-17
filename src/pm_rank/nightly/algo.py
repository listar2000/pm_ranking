import pandas as pd
import numpy as np
from typing import Literal, Dict, Optional
from pm_rank.model.calibration import _bin_stats, _calculate_ece
from pm_rank.nightly.bootstrap import compute_bootstrap_ci


DEFAULT_BOOTSTRAP_CONFIG = {
    'num_samples': 1000,
    'ci_level': 0.9,
    'num_se': None,
    'random_seed': 42,
    'show_progress': True
}


def rank_forecasters_by_score(result_df: pd.DataFrame, normalize_by_round: bool = False, 
                              score_col: str = None, ascending: bool = None,
                              bootstrap_config: Optional[Dict] = None) -> pd.DataFrame:
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
        else:
            raise ValueError("Could not find score column. Please specify 'score_col' parameter.")
    
    # Auto-detect ascending if not provided
    if ascending is None:
        # Lower is better for Brier score and ECE score, higher is better for average return
        ascending = (score_col in ['brier_score', 'ece_score'])
    
    # Special handling for ECE scores: they're already aggregated per forecaster
    if score_col == 'ece_score':
        if bootstrap_config is not None:
            raise ValueError("Bootstrap CI is not supported for ECE scores (already aggregated)")
        
        # ECE scores are already computed per forecaster, just need to rank and format
        forecaster_scores = df[['forecaster', 'ece_score']].copy()
        forecaster_scores.columns = ['forecaster', 'score']
        
        # Rank forecasters (ascending=True means lower score = better rank)
        forecaster_scores['rank'] = forecaster_scores['score'].rank(method='min', ascending=ascending).astype(int)
        
        # Sort by rank and select required columns, then set rank as index
        rank_df = forecaster_scores[['forecaster', 'rank', 'score']].sort_values('rank')
        rank_df = rank_df.set_index('rank')[['forecaster', 'score']]
        
        return rank_df
    
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
    
    # Compute bootstrap confidence intervals if requested
    ci_col_name = f'{bootstrap_config["ci_level"] * 100}% ci'
    if bootstrap_config is not None:
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
        rank_df = forecaster_scores[['forecaster', 'rank', 'score', ci_col_name]].sort_values('rank')
        rank_df = rank_df.set_index('rank')[['forecaster', 'score', ci_col_name]]
    else:
        rank_df = forecaster_scores[['forecaster', 'rank', 'score']].sort_values('rank')
        rank_df = rank_df.set_index('rank')[['forecaster', 'score']]
    
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
        # if no reference forecaster is provided, we use the first forecaster
        reference_forecaster = forecasts['forecaster'].unique()[0]

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
                           weight_event: bool = True) -> pd.DataFrame:
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
    
    # Create result DataFrame
    result_df = pd.DataFrame(ece_results)
    
    # Sort by ECE score (lower is better)
    result_df = result_df.sort_values('ece_score').reset_index(drop=True)
    
    return result_df
    

if __name__ == "__main__":
    predictions_csv = "slurm/predictions_10_11_to_01_01.csv"
    submissions_csv = "slurm/submissions_10_11_to_01_01.csv"

    from pm_rank.nightly.data import uniform_weighting, exponential_weighting, first_n_weighting, NightlyForecasts
    
    weight_fn = uniform_weighting()
    # weight_fn = first_n_weighting(n=1)
    forecasts = NightlyForecasts.from_prophet_arena_csv(predictions_csv, submissions_csv, weight_fn)

    forecasts.data = add_market_baseline_predictions(forecasts.data)
    
    brier_score = compute_brier_score(forecasts.data)
    avg_returns = compute_average_return_neutral(forecasts.data, spread_market_even=False)
    ece_results = compute_calibration_ece(forecasts.data, num_bins=10, strategy="uniform", weight_event=False)

    # Test Brier score with Bootstrap CI
    print("\n" + "=" * 50)
    print("BRIER SCORE RANKINGS (with Bootstrap CI)")
    print("=" * 50)
    print(rank_forecasters_by_score(brier_score, normalize_by_round=True, bootstrap_config=DEFAULT_BOOTSTRAP_CONFIG))

    # Test Average Return with Bootstrap CI
    print("\n" + "=" * 50)
    print("AVERAGE RETURN RANKINGS (with Bootstrap CI)")
    print("=" * 50)
    print(rank_forecasters_by_score(avg_returns, normalize_by_round=True, bootstrap_config=DEFAULT_BOOTSTRAP_CONFIG))
    
    # Test Calibration (ECE)
    print("\n" + "=" * 50)
    print("CALIBRATION (ECE) RANKINGS")
    print("=" * 50)
    print(rank_forecasters_by_score(ece_results))