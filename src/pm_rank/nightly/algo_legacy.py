import pandas as pd
import numpy as np
import warnings


def compute_average_return_neutral_legacy(forecasts: pd.DataFrame, num_money_per_round: float = 1.0, 
                                   spread_market_even: bool = False) -> pd.DataFrame:
    """
    (This is a legacy implementation of the average return calculation. Now deprecated)

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
    warnings.warn(
        "This is a legacy implementation of the average return calculation. Now deprecated. "
        "Please use the new implementation in algo.py instead.", 
        DeprecationWarning
    )
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