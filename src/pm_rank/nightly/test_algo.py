"""
Test script to compare the new nightly average return implementation 
with the original average_return.py implementation.
"""
import numpy as np
import pandas as pd
from pm_rank.nightly.data import NightlyForecasts, last_n_weighting
from pm_rank.nightly.algo import compute_average_return_neutral, rank_forecasters_by_score

def test_simple_case():
    """Test with a simple synthetic case to validate the logic."""
    # Create a simple test case
    # Event 1: 2 markets, 2 forecasters
    data = {
        'forecaster': ['alice', 'bob', 'alice', 'bob'],
        'event_ticker': ['event1', 'event1', 'event2', 'event2'],
        'round': [1, 1, 1, 1],
        'prediction': [
            np.array([0.8, 0.2]),  # alice predicts 80% market 0, 20% market 1
            np.array([0.3, 0.7]),  # bob predicts 30% market 0, 70% market 1
            np.array([0.9, 0.1]),  # alice predicts 90% market 0, 10% market 1
            np.array([0.4, 0.6]),  # bob predicts 40% market 0, 60% market 1
        ],
        'outcome': [
            np.array([1, 0]),  # market 0 wins
            np.array([1, 0]),  # market 0 wins
            np.array([0, 1]),  # market 1 wins
            np.array([0, 1]),  # market 1 wins
        ],
        'odds': [
            np.array([0.5, 0.5]),  # 50-50 market odds
            np.array([0.5, 0.5]),
            np.array([0.6, 0.4]),
            np.array([0.6, 0.4]),
        ],
        'no_odds': [
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            np.array([0.4, 0.6]),
            np.array([0.4, 0.6]),
        ],
        'weight': [1.0, 1.0, 1.0, 1.0],
        'time_rank': [0, 0, 0, 0],
    }
    
    df = pd.DataFrame(data)
    
    # Calculate average returns
    returns = compute_average_return_neutral(df, num_money_per_round=1.0)
    
    print("=" * 60)
    print("SIMPLE TEST CASE")
    print("=" * 60)
    print("\nReturns DataFrame:")
    print(returns)
    
    # Rank forecasters
    rankings = rank_forecasters_by_score(returns, normalize_by_round=False)
    print("\nRankings:")
    print(rankings)
    
    # Manual verification for event1, alice:
    # YES edges: [0.8/0.5, 0.2/0.5] = [1.6, 0.4]
    # NO edges:  [0.2/0.5, 0.8/0.5] = [0.4, 1.6]
    # Choose: [YES, NO] (because 1.6 > 0.4 and 1.6 > 0.4)
    # Effective probs: [0.8, 0.8], prices: [0.5, 0.5], edges: [1.6, 1.6]
    # Risk-neutral: bet on market 0 (first max), buy 1/0.5 = 2 contracts
    # Outcome: market 0 wins (outcome=[1,0]), chose YES on market 0, so effective outcome = 1
    # Earnings: 2 * 1 = 2
    
    print("\n" + "=" * 60)
    print("MANUAL VERIFICATION")
    print("=" * 60)
    print("\nEvent 1, Alice:")
    print("  YES edges: [0.8/0.5, 0.2/0.5] = [1.6, 0.4]")
    print("  NO edges:  [0.2/0.5, 0.8/0.5] = [0.4, 1.6]")
    print("  Choose: [YES on mkt 0 (1.6 > 0.4), NO on mkt 1 (1.6 > 0.4)]")
    print("  Effective probs: [0.8, 0.8], prices: [0.5, 0.5]")
    print("  Effective edges: [1.6, 1.6] - TIE!")
    print("  Risk-neutral: bet on first max (market 0)")
    print("  Contracts: 1 / 0.5 = 2.0")
    print("  Outcome: market 0 wins, chose YES, effective=1")
    print("  Earnings: 2.0 * 1 = 2.0")
    print(f"  Computed: {returns[returns['forecaster'] == 'alice']['average_return'].iloc[0]}")
    

if __name__ == "__main__":
    test_simple_case()
    
    print("\n\n" + "=" * 60)
    print("RUNNING MAIN DATA TEST")
    print("=" * 60)
    
    # Test with actual data if available
    try:
        from pathlib import Path
        predictions_csv = "slurm/predictions_10_01.csv"
        submissions_csv = "slurm/submissions_10_01.csv"
        
        if Path(predictions_csv).exists() and Path(submissions_csv).exists():
            print(f"\nLoading data from {predictions_csv} and {submissions_csv}...")
            weight_fn = last_n_weighting(n=10)
            forecasts = NightlyForecasts.from_prophet_arena_csv(predictions_csv, submissions_csv, weight_fn)
            
            df = forecasts.data
            print(f"Loaded {len(df)} rows")
            
            avg_returns = compute_average_return_neutral(df)
            print(f"\nComputed returns for {len(avg_returns)} rows")
            
            rankings = rank_forecasters_by_score(avg_returns, normalize_by_round=False)
            print("\nTop 10 Rankings (by average return):")
            print(rankings.head(10))
        else:
            print("\nData files not found, skipping real data test")
    except Exception as e:
        print(f"\nError loading real data: {e}")

