from pm_rank.nightly.data import NightlyForecasts
from typing import Literal
import pandas as pd

ALL_CATEGORIES = ["Sports", "Entertainment", "Politics", "Other"]

def _balance_by_event(original_forecast_df: pd.DataFrame, rebalance_quota: dict[str, float], random_seed: int = 42) -> pd.DataFrame:
    """
    Do a budget based sampling -- given the quota (i.e. desired distribution of event categories), we first need to determine
    how many events to sample by the "lower bound"
    """
    # step 1: get the counts of each category by summing up the category for unique event_tickers. Counts should sum to the total number of events.
    # Since each event_ticker has a unique category, we can simply get unique event_tickers and count their categories
    unique_events = original_forecast_df[['event_ticker', 'category']].drop_duplicates()

    # replace the category to 'Other' if it is not in the ALL_CATEGORIES
    unique_events['category'] = unique_events['category'].apply(lambda x: 'Other' if x not in ALL_CATEGORIES else x)

    category_counts = unique_events['category'].value_counts().to_dict()
    
    # step 2: for each category, divide the count by the quota, and takes the smallest result (integer) as "lower bound" across categories.
    lower_bounds = []
    for category, quota in rebalance_quota.items():
        if category in category_counts and quota > 0:
            lower_bounds.append(int(category_counts[category] / quota))
    
    if not lower_bounds:
        return original_forecast_df  # Return original if no valid categories
    
    lower_bound = min(lower_bounds)
    
    # step 3: determine the "actual quota" by multiplying the lower bound by the quota.
    actual_quota = {category: int(lower_bound * quota) for category, quota in rebalance_quota.items()}
    
    # step 4: actually sample the events by the actual quota.
    sampled_event_tickers = []
    for category, target_count in actual_quota.items():
        if target_count > 0 and category in category_counts:
            # Get all event_tickers for this category
            category_events = unique_events[unique_events['category'] == category]['event_ticker'].tolist()
            # Sample up to target_count events (or all if fewer available)
            sampled_count = min(target_count, len(category_events))
            sampled_events = pd.Series(category_events).sample(n=sampled_count, random_state=random_seed).tolist()
            sampled_event_tickers.extend(sampled_events)

    # print out some information of the resampled event distribution
    rebalanced_df = original_forecast_df[original_forecast_df['event_ticker'].isin(sampled_event_tickers)]
    rebalanced_unique_events = rebalanced_df[['event_ticker', 'category']].drop_duplicates()
    rebalanced_distribution = rebalanced_unique_events['category'].value_counts().to_dict()
    
    print("Rebalanced event distribution:", rebalanced_distribution)
    
    # step 5: return subset of the original dataframe, i.e. all rows that has the sampled event_tickers.
    return rebalanced_df


def get_rebalanced_forecasts(original_forecasts: NightlyForecasts, 
    balance_level: Literal['event', 'submission'] = 'event',
    evenly_balanced: bool = True, rebalance_quota: dict[str, float] = None, random_seed: int = 42) -> NightlyForecasts:

    assert evenly_balanced or rebalance_quota is not None, "Either evenly_balanced or rebalance_quota must be provided"

    if balance_level == 'submission':
        raise NotImplementedError("Balance by submission is not implemented yet")

    if rebalance_quota is not None:
        quota_sum = 0
        cleaned_quota = dict()
        for category, quota in rebalance_quota.items():
            if category in ALL_CATEGORIES:
                quota_sum += quota
                cleaned_quota[category] = quota
            else:
                raise ValueError(f"Quota must be positive and category must be in {ALL_CATEGORIES}, got {category} with quota {quota}")
        
        if quota_sum == 0:
            raise ValueError("Quota must sum to a positive value")
        
        cleaned_quota = {k: v / quota_sum for k, v in cleaned_quota.items()} # normalize the quota to sum to 1.0
    else:
        # evenly split
        cleaned_quota = {category: 1.0 / len(ALL_CATEGORIES) for category in ALL_CATEGORIES}


    new_forecast_df = _balance_by_event(original_forecasts.data, cleaned_quota, random_seed)
    new_forecasts = NightlyForecasts(new_forecast_df)

    return new_forecasts

    

