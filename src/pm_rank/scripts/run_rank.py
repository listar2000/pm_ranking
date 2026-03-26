from pm_rank.nightly.algo import (
    add_market_baseline_predictions, 
    compute_ranked_brier_score,
    DEFAULT_BOOTSTRAP_CONFIG,
    compute_brier_score,
    rank_forecasters_by_score
)
from pm_rank.nightly.data import uniform_weighting, NightlyForecasts

predictions_csv = "slurm/predictions_04_01_to_02_01.csv"  # Your predictions CSV file
submissions_csv = "slurm/submissions_04_01_to_02_01.csv"  # Your submissions CSV file


weight_fn = uniform_weighting()
forecasts = NightlyForecasts.from_prophet_arena_csv(predictions_csv, submissions_csv, weight_fn)

# from pm_rank.nightly.misc import get_rebalanced_forecasts
# forecasts = get_rebalanced_forecasts(forecasts, balance_level='event', evenly_balanced=True, random_seed=42)
# or you can do
# desired_quota = {"Sports": 0.2, "Entertainment": 0.2, "Politics": 0.2, "Companies": 0.2, "Mentions": 0.2, "Economics": 0.2, "Climate and Weather": 0.2}
# forecasts = get_rebalanced_forecasts(forecasts, balance_level='event', rebalance_quota=desired_quota, random_seed=42)

forecasts.data = add_market_baseline_predictions(forecasts.data)

# rank_df = compute_ranked_brier_score(forecasts.data, bootstrap_config=DEFAULT_BOOTSTRAP_CONFIG, resample_level="market")
# rank_df = rank_df[rank_df['forecaster'].str.contains('foresight')]

# print("Market-level rank:")
# print(rank_df)

event_rank_df = compute_ranked_brier_score(forecasts.data, bootstrap_config=DEFAULT_BOOTSTRAP_CONFIG, resample_level="event")
event_rank_df = event_rank_df[event_rank_df['forecaster'].str.contains('foresight')]

print("Event-level rank:")
print(event_rank_df)