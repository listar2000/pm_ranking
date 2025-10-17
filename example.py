# IMPORTANT: use this to select submissions made within some datetime intervals
predictions_csv = "slurm/predictions_10_11_to_09_27.csv"
submissions_csv = "slurm/submissions_10_11_to_09_27.csv"

from pm_rank.nightly.data import uniform_weighting, time_to_last_weighting, NightlyForecasts
from pm_rank.nightly.algo import (
    compute_brier_score, 
    compute_average_return_neutral, 
    add_market_baseline_predictions, 
    compute_calibration_ece, 
    rank_forecasters_by_score, 
    DEFAULT_BOOTSTRAP_CONFIG
)

# IMPORTANT:use this to filter out predictions that are too close to the market close
weight_fn = time_to_last_weighting(min_hours=3.0)

exclude_forecasters = ['qwen/qwen3-235b-a22b-thinking-2507']
forecasts = NightlyForecasts.from_prophet_arena_csv(predictions_csv, submissions_csv, weight_fn, exclude_forecasters)

# check number of unique events and markets
unique_events = forecasts.data['event_ticker'].unique()
# unique market is the sum of length for the `odds` column for all unique event_tickers
unique_markets = forecasts.data.groupby('event_ticker')['odds'].apply(len).sum()
print(f"Number of unique events: {len(unique_events)}")
print(f"Number of unique markets: {unique_markets}")

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
