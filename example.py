predictions_csv = "slurm/predictions_10_11_to_01_01.csv"
submissions_csv = "slurm/submissions_10_11_to_01_01.csv"

from pm_rank.nightly.data import uniform_weighting, NightlyForecasts
from pm_rank.nightly.algo import (
    compute_brier_score, 
    compute_average_return_neutral, 
    add_market_baseline_predictions, 
    compute_calibration_ece, 
    rank_forecasters_by_score, 
    DEFAULT_BOOTSTRAP_CONFIG
)

weight_fn = uniform_weighting()
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
