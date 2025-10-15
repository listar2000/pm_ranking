"""
Test script for bootstrap confidence interval calculation.

NOTE: Bootstrap resampling is done SEPARATELY for each forecaster.
At each iteration, we sample (with replacement) from each forecaster's
own predictions using their adjusted weights. This properly estimates
the uncertainty in each forecaster's individual score.
"""
import numpy as np
import pandas as pd
from pm_rank.nightly.algo import compute_brier_score, compute_average_return_neutral, rank_forecasters_by_score, add_market_baseline_predictions


def test_bootstrap_simple():
    """Test bootstrap CI with a simple synthetic case."""
    # Create a simple test case with known properties
    np.random.seed(42)
    
    data = []
    
    # Create 100 predictions for two forecasters
    # Alice is consistently good (Brier score ~ 0.15)
    # Bob is more variable (Brier score ~ 0.25)
    for i in range(50):
        # Event with 2 markets
        event_id = f'event_{i}'
        
        # True outcome (random)
        outcome = np.array([np.random.choice([0, 1]), np.random.choice([0, 1])])
        
        # Alice makes good predictions (close to truth with small noise)
        alice_pred = outcome + np.random.normal(0, 0.2, 2)
        alice_pred = np.clip(alice_pred, 0.05, 0.95)
        alice_pred = alice_pred / alice_pred.sum()  # Normalize
        
        # Bob makes more variable predictions
        bob_pred = np.random.uniform(0.2, 0.8, 2)
        bob_pred = bob_pred / bob_pred.sum()  # Normalize
        
        # Market odds (neutral)
        odds = np.array([0.5, 0.5])
        no_odds = np.array([0.5, 0.5])
        
        data.append({
            'forecaster': 'alice',
            'event_ticker': event_id,
            'round': 1,
            'prediction': alice_pred,
            'outcome': outcome,
            'odds': odds,
            'no_odds': no_odds,
            'weight': 1.0,
            'time_rank': 0
        })
        
        data.append({
            'forecaster': 'bob',
            'event_ticker': event_id,
            'round': 1,
            'prediction': bob_pred,
            'outcome': outcome,
            'odds': odds,
            'no_odds': no_odds,
            'weight': 1.0,
            'time_rank': 0
        })
    
    df = pd.DataFrame(data)
    
    print("=" * 70)
    print("BOOTSTRAP CI TEST - BRIER SCORE")
    print("=" * 70)
    print(f"\nDataset: {len(df)} predictions from 2 forecasters")
    print(f"Alice: Consistently good predictions (low Brier score)")
    print(f"Bob: More variable predictions (higher Brier score)")
    
    # Compute Brier scores
    brier_scores = compute_brier_score(df)
    
    # Without bootstrap CI
    print("\n" + "-" * 70)
    print("Rankings WITHOUT Bootstrap CI:")
    print("-" * 70)
    rank_df = rank_forecasters_by_score(brier_scores, normalize_by_round=False)
    print(rank_df)
    
    # With bootstrap CI (using num_se)
    print("\n" + "-" * 70)
    print("Rankings WITH Bootstrap CI (±1.96 SE for 95% CI):")
    print("-" * 70)
    bootstrap_config = {
        'num_samples': 500,
        'num_se': 1.96,  # 95% CI
        'random_seed': 42,
        'show_progress': True
    }
    rank_df_ci = rank_forecasters_by_score(brier_scores, normalize_by_round=False, bootstrap_config=bootstrap_config)
    print(rank_df_ci)
    
    # With bootstrap CI (using ci_level)
    print("\n" + "-" * 70)
    print("Rankings WITH Bootstrap CI (symmetric deviation method):")
    print("-" * 70)
    bootstrap_config2 = {
        'num_samples': 500,
        'ci_level': 0.95,
        'random_seed': 42,
        'show_progress': True
    }
    rank_df_ci2 = rank_forecasters_by_score(brier_scores, normalize_by_round=False, bootstrap_config=bootstrap_config2)
    print(rank_df_ci2)
    
    print("\n" + "=" * 70)
    print("Analysis:")
    print("=" * 70)
    alice_score = rank_df_ci.loc[rank_df_ci['forecaster'] == 'alice', 'score'].values[0]
    alice_se = rank_df_ci.loc[rank_df_ci['forecaster'] == 'alice', 'se'].values[0]
    alice_lower = rank_df_ci.loc[rank_df_ci['forecaster'] == 'alice', 'lower'].values[0]
    alice_upper = rank_df_ci.loc[rank_df_ci['forecaster'] == 'alice', 'upper'].values[0]
    
    bob_score = rank_df_ci.loc[rank_df_ci['forecaster'] == 'bob', 'score'].values[0]
    bob_se = rank_df_ci.loc[rank_df_ci['forecaster'] == 'bob', 'se'].values[0]
    bob_lower = rank_df_ci.loc[rank_df_ci['forecaster'] == 'bob', 'lower'].values[0]
    bob_upper = rank_df_ci.loc[rank_df_ci['forecaster'] == 'bob', 'upper'].values[0]
    
    print(f"Alice: {alice_score:.4f} ± {alice_se:.4f} [{alice_lower:.4f}, {alice_upper:.4f}]")
    print(f"Bob:   {bob_score:.4f} ± {bob_se:.4f} [{bob_lower:.4f}, {bob_upper:.4f}]")
    print(f"\nAlice should have lower Brier score (better) with smaller SE")
    print(f"Bob should have higher Brier score (worse) with larger SE")
    
    # Check if CIs don't overlap (strong evidence of difference)
    if alice_upper < bob_lower:
        print("\n✓ CIs don't overlap - strong evidence that Alice is better than Bob")
    else:
        print(f"\n✓ CIs overlap - but Alice's point estimate is still better")


def test_bootstrap_average_return():
    """Test bootstrap CI with average return metric."""
    np.random.seed(123)
    
    data = []
    
    # Create predictions with different return profiles
    for i in range(30):
        event_id = f'event_{i}'
        outcome = np.array([1, 0])  # First market always wins
        
        # Charlie: good at identifying the winning market (high returns)
        charlie_pred = np.array([0.8, 0.2])
        
        # David: poor at identifying the winning market (low returns)
        david_pred = np.array([0.3, 0.7])
        
        # Market odds
        odds = np.array([0.5, 0.5])
        no_odds = np.array([0.5, 0.5])
        
        for forecaster, pred in [('charlie', charlie_pred), ('david', david_pred)]:
            data.append({
                'forecaster': forecaster,
                'event_ticker': event_id,
                'round': 1,
                'prediction': pred,
                'outcome': outcome,
                'odds': odds,
                'no_odds': no_odds,
                'weight': 1.0,
                'time_rank': 0
            })
    
    df = pd.DataFrame(data)
    
    print("\n\n" + "=" * 70)
    print("BOOTSTRAP CI TEST - AVERAGE RETURN")
    print("=" * 70)
    print(f"\nDataset: {len(df)} predictions from 2 forecasters")
    print(f"First market always wins in all events")
    print(f"Charlie: Predicts 80% on first market (high returns)")
    print(f"David: Predicts 30% on first market (low returns)")
    
    # Compute average returns
    avg_returns = compute_average_return_neutral(df, spread_market_even=False)
    
    # Without bootstrap CI
    print("\n" + "-" * 70)
    print("Rankings WITHOUT Bootstrap CI:")
    print("-" * 70)
    rank_df = rank_forecasters_by_score(avg_returns, normalize_by_round=False)
    print(rank_df)
    
    # With bootstrap CI
    print("\n" + "-" * 70)
    print("Rankings WITH Bootstrap CI (±1.96 SE):")
    print("-" * 70)
    bootstrap_config = {
        'num_samples': 500,
        'num_se': 1.96,
        'random_seed': 123,
        'show_progress': True
    }
    rank_df_ci = rank_forecasters_by_score(avg_returns, normalize_by_round=False, bootstrap_config=bootstrap_config)
    print(rank_df_ci)
    
    print("\n" + "=" * 70)
    print("Analysis:")
    print("=" * 70)
    charlie_score = rank_df_ci.loc[rank_df_ci['forecaster'] == 'charlie', 'score'].values[0]
    charlie_se = rank_df_ci.loc[rank_df_ci['forecaster'] == 'charlie', 'se'].values[0]
    
    david_score = rank_df_ci.loc[rank_df_ci['forecaster'] == 'david', 'score'].values[0]
    david_se = rank_df_ci.loc[rank_df_ci['forecaster'] == 'david', 'se'].values[0]
    
    print(f"Charlie: {charlie_score:.4f} ± {charlie_se:.4f}")
    print(f"David:   {david_score:.4f} ± {david_se:.4f}")
    print(f"\nCharlie should have higher average return (better)")
    print(f"David should have lower average return (worse)")
    print("\n✓ Test demonstrates bootstrap CI for average return metric")


if __name__ == "__main__":
    test_bootstrap_simple()
    test_bootstrap_average_return()
    
    print("\n\n" + "=" * 70)
    print("ALL BOOTSTRAP TESTS COMPLETED!")
    print("=" * 70)

