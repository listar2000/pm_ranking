"""
Test script for the new calibration metric implementation.
"""
import numpy as np
import pandas as pd
from pm_rank.nightly.algo import compute_calibration_ece, rank_forecasters_by_score


def test_simple_calibration():
    """Test calibration with a simple synthetic case."""
    # Create a simple test case
    # Forecaster "alice" makes well-calibrated predictions
    # Forecaster "bob" makes overconfident predictions
    
    data = {
        'forecaster': ['alice', 'alice', 'alice', 'alice', 
                      'bob', 'bob', 'bob', 'bob'],
        'event_ticker': ['event1', 'event2', 'event3', 'event4',
                        'event1', 'event2', 'event3', 'event4'],
        'round': [1, 1, 1, 1, 1, 1, 1, 1],
        'prediction': [
            # Alice: well-calibrated predictions (50-50)
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            # Bob: overconfident predictions (90-10)
            np.array([0.9, 0.1]),
            np.array([0.9, 0.1]),
            np.array([0.9, 0.1]),
            np.array([0.9, 0.1]),
        ],
        'outcome': [
            # Outcomes roughly 50-50 for both
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([1, 0]),
            np.array([0, 1]),
        ],
        'weight': [1.0] * 8,
        'time_rank': [0] * 8,
    }
    
    df = pd.DataFrame(data)
    
    print("=" * 60)
    print("SIMPLE CALIBRATION TEST")
    print("=" * 60)
    print("\nInput Data:")
    print(df[['forecaster', 'event_ticker', 'prediction', 'outcome']])
    
    # Calculate ECE with event weighting
    print("\n" + "=" * 60)
    print("ECE Results (weight_event=True)")
    print("=" * 60)
    ece_results = compute_calibration_ece(df, num_bins=10, strategy="uniform", weight_event=True)
    print("\nRanked:")
    print(rank_forecasters_by_score(ece_results))
    
    print("\n" + "=" * 60)
    print("ECE Results (weight_event=False)")
    print("=" * 60)
    ece_results_no_weight = compute_calibration_ece(df, num_bins=10, strategy="uniform", weight_event=False)
    print(ece_results_no_weight)
    print("\nRanked:")
    print(rank_forecasters_by_score(ece_results_no_weight))
    
    print("\n" + "=" * 60)
    print("Analysis:")
    print("=" * 60)
    print("Alice predicts 50% on all markets, and outcomes are 50-50.")
    print("Bob predicts 90% on all markets, but outcomes are still 50-50.")
    print("Expected: Alice should have lower ECE (better calibrated) than Bob.")
    print(f"Alice ECE: {ece_results[ece_results['forecaster'] == 'alice']['ece_score'].values[0]:.4f}")
    print(f"Bob ECE:   {ece_results[ece_results['forecaster'] == 'bob']['ece_score'].values[0]:.4f}")
    
    assert ece_results.iloc[0]['forecaster'] == 'alice', "Alice should be ranked first (better calibrated)"
    print("\n✓ Test passed! Alice is better calibrated than Bob.")


def test_perfect_calibration():
    """Test with perfectly calibrated predictions."""
    # Create perfectly calibrated predictions
    np.random.seed(42)
    
    forecasters_data = []
    
    # Perfect forecaster: predictions match actual frequencies
    for i in range(100):
        pred_prob = 0.7
        # Generate outcome with 70% probability
        outcome = 1 if np.random.random() < pred_prob else 0
        forecasters_data.append({
            'forecaster': 'perfect',
            'event_ticker': f'event_{i}',
            'round': 1,
            'prediction': np.array([pred_prob, 1 - pred_prob]),
            'outcome': np.array([outcome, 1 - outcome]),
            'weight': 1.0,
            'time_rank': 0
        })
    
    df = pd.DataFrame(forecasters_data)
    
    print("\n\n" + "=" * 60)
    print("PERFECT CALIBRATION TEST")
    print("=" * 60)
    
    ece_results = compute_calibration_ece(df, num_bins=10, strategy="uniform", weight_event=True)
    print("\nRaw ECE Results:")
    print(ece_results)
    print("\nRanked:")
    print(rank_forecasters_by_score(ece_results))
    
    perfect_ece = ece_results[ece_results['forecaster'] == 'perfect']['ece_score'].values[0]
    print(f"\nPerfect forecaster ECE: {perfect_ece:.4f}")
    print("Expected: ECE should be close to 0 for perfectly calibrated predictions.")
    print("(Note: Due to random sampling, it won't be exactly 0, but should be small)")


if __name__ == "__main__":
    test_simple_calibration()
    test_perfect_calibration()
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

