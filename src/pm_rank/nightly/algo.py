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

    # # Optionally add individualized market baselines before aggregation
    # if add_individualized_baselines:
    #     df = add_individualized_market_baselines_to_scores(df)
    
    # Drop rows with NaN scores (e.g. illiquid events skipped by average return)
    df = df.dropna(subset=[score_col])

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
    Calculate the Brier score for the forecasts using row-by-row processing.
    Handles predictions with different array lengths via key intersection.
    Automatically filters out illiquid events (yes_ask + no_ask > 1.03).

    The result will be a DataFrame containing (forecaster, event_ticker, round, weight, brier_score)

    Args:
        forecasts: DataFrame with columns (forecaster, event_ticker, round, prediction, outcome, weight, odds, no_odds)
    """
    MAX_SPREAD = 1.03

    # Filter out rows with mismatched array lengths
    def _arrays_same_length(row):
        try:
            return len(row['prediction']) == len(row['odds']) == len(row['no_odds']) == len(row['outcome'])
        except Exception:
            return False

    length_mask = forecasts.apply(_arrays_same_length, axis=1)
    filtered_count = (~length_mask).sum()
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} rows with mismatched array lengths")
    result_df = forecasts[length_mask].copy()

    # Initialize brier_score column
    result_df['brier_score'] = np.nan

    skipped_illiquid = 0

    for idx, row in result_df.iterrows():
        try:
            prediction = row['prediction']
            outcome = row['outcome']
            odds = row['odds']
            no_odds = row['no_odds']

            # Skip illiquid events (spread > 1.03 for any market)
            spreads = odds + no_odds
            if np.any(spreads > MAX_SPREAD):
                skipped_illiquid += 1
                continue

            if len(prediction) == 0:
                continue

            # Calculate Brier score: mean squared difference
            brier_score = np.mean((prediction - outcome) ** 2)
            result_df.at[idx, 'brier_score'] = brier_score

        except Exception:
            continue

    if skipped_illiquid > 0:
        print(f"Skipped {skipped_illiquid} illiquid events (spread > {MAX_SPREAD})")

    # Drop rows with NaN brier_score (illiquid events or failed computations)
    result_df = result_df.dropna(subset=['brier_score'])

    # Select only the required columns
    result_df = result_df[['forecaster', 'event_ticker', 'weight', 'round', 'brier_score']]
    return result_df


def compute_average_return_neutral(forecasts: pd.DataFrame, num_money_per_round: float = 1.0, spread_market_even: bool = False, max_spread: float = 1.03) -> pd.DataFrame:
    """
    Each forecaster is given a fixed $1000 budget spread across ALL markets they
    participate in. For each outcome within an event, the forecaster either bets
    YES or NO depending on their edge. 

    Betting logic (per outcome):
    diff = p - yes_ask

    If diff > 0: bet YES at price yes_ask 
    If diff < 0: bet NO  at price no_ask  
    If diff = 0: skip                     

    Budget allocation:
        amount_i = BUDGET * weight_i / sum(weight_j for all j across all events)

    Liquidity filter:
        Entire events are skipped if ANY outcome has yes_ask + no_ask > max_spread.
        This avoids betting into illiquid markets with excessive vig.

    Args:
        forecasts: DataFrame with columns:
            - forecaster: str, model/forecaster identifier
            - event_ticker: str, event identifier
            - round: int, forecast round number
            - prediction: np.ndarray, forecaster's probability for each outcome
            - outcome: np.ndarray, actual binary outcomes (0 or 1) per market
            - odds: np.ndarray, YES ask prices (implied probabilities) per market
            - no_odds: np.ndarray, NO ask prices per market
            - weight: float, external weight (passed through, not used in computation)
        num_money_per_round: Unused, kept for API compatibility.
        spread_market_even: Unused, kept for API compatibility.
        max_spread: Maximum allowed spread (yes_ask + no_ask) for liquidity filter.
            Events with any outcome exceeding this threshold are skipped entirely.

    Returns:
        DataFrame with columns (forecaster, event_ticker, round, weight, average_return)
        where average_return is the net profit (can be negative) for that
        forecaster on that event/round.
    """
    # Filter out rows with mismatched array lengths
    def _arrays_same_length(row):
        try:
            return len(row['prediction']) == len(row['odds']) == len(row['no_odds']) == len(row['outcome'])
        except Exception:
            return False

    length_mask = forecasts.apply(_arrays_same_length, axis=1)
    filtered_count = (~length_mask).sum()
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} rows with mismatched array lengths")
    forecasts = forecasts[length_mask].copy()

    # Deduplicate: keep only the median round per (forecaster, event_ticker)
    median_rounds = forecasts.groupby(['forecaster', 'event_ticker'])['round'].median().reset_index()
    median_rounds.columns = ['forecaster', 'event_ticker', 'median_round']
    forecasts = forecasts.merge(median_rounds, on=['forecaster', 'event_ticker'])
    # Pick the round closest to the median for each group
    forecasts['round_dist'] = (forecasts['round'] - forecasts['median_round']).abs()
    forecasts = forecasts.sort_values('round_dist').drop_duplicates(subset=['forecaster', 'event_ticker'], keep='first')
    forecasts = forecasts.drop(columns=['median_round', 'round_dist'])
    print(f"After median-round dedup: {len(forecasts)} rows (1 per forecaster-event)")

    TOTAL_BUDGET = 1000

    # PASS 1: Calculate total weights per forecaster (row by row)
    forecaster_total_weights = {}
    row_weights = {}  # {idx: (weight_sum, weights_array, ...)}

    for idx, row in forecasts.iterrows():
        forecaster = row['forecaster']

        try:
            prediction = row['prediction']
            odds = row['odds']
            no_odds = row['no_odds']

            if len(prediction) == 0:
                continue

            # Sanity check: skip outcomes with invalid prices (<=0 or >=1)
            valid_prices = (odds > 0) & (odds < 1) & (no_odds > 0) & (no_odds < 1)
            if not np.any(valid_prices):
                continue

            # Check liquidity for ALL outcomes
            spreads = odds + no_odds
            if np.any(spreads > max_spread):
                continue

            # Calculate weights: |p - yes_ask| * price
            diff = prediction - odds
            bet_yes = diff > 0
            bet_no = diff < 0

            # Price is yes_ask for YES bets, no_ask for NO bets
            price = np.where(bet_yes, odds, np.where(bet_no, no_odds, 1.0))

            # Weight = |diff| * price, but only for non-zero diffs and valid prices
            weights = np.where((diff != 0) & valid_prices, np.abs(diff) * price, 0.0)
            total_weight = np.sum(weights)

            if forecaster not in forecaster_total_weights:
                forecaster_total_weights[forecaster] = 0.0
            forecaster_total_weights[forecaster] += total_weight

            # Store for pass 2
            row_weights[idx] = {
                'weights': weights,
                'price': price,
                'bet_yes': bet_yes,
                'forecaster': forecaster
            }

        except Exception:
            continue

    # PASS 2: Calculate profit row by row
    result_data = []

    for idx, row in forecasts.iterrows():
        forecaster = row['forecaster']

        try:
            if idx in row_weights:
                data = row_weights[idx]
                weights = data['weights']
                price = data['price']
                bet_yes = data['bet_yes']

                outcome = row['outcome']

                # Get total weight for this forecaster
                total_weight = forecaster_total_weights.get(forecaster, 0.0)
                if total_weight == 0:
                    profit = 0.0
                else:
                    # Allocate budget proportionally
                    amount_per_market = TOTAL_BUDGET * weights / total_weight

                    # Calculate win/loss
                    is_yes_win = outcome == 1
                    is_no_win = outcome == 0
                    is_win = np.where(bet_yes, is_yes_win, is_no_win)

                    # Calculate profit per market
                    safe_prices = np.where(weights > 0, price, 1.0)
                    win_profit = (amount_per_market / safe_prices) - amount_per_market
                    lose_profit = -amount_per_market

                    profit_per_market = np.where(is_win, win_profit, lose_profit)
                    profit_per_market = np.where(weights > 0, profit_per_market, 0.0)

                    profit = np.sum(profit_per_market)

                result_data.append({
                    'forecaster': forecaster,
                    'event_ticker': row['event_ticker'],
                    'round': row['round'],
                    'weight': row['weight'],
                    'average_return': profit
                })
            else:
                # Row was skipped (illiquid or failed)
                result_data.append({
                    'forecaster': forecaster,
                    'event_ticker': row['event_ticker'],
                    'round': row['round'],
                    'weight': row['weight'],
                    'average_return': np.nan
                })

        except Exception:
            result_data.append({
                'forecaster': forecaster,
                'event_ticker': row['event_ticker'],
                'round': row['round'],
                'weight': row['weight'],
                'average_return': np.nan
            })

    result_df = pd.DataFrame(result_data)
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
