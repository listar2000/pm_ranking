import pandas as pd
import numpy as np
import json
from typing import Literal

WeightingStrategy = Literal['uniform', 'first_n', 'last_n', 'exponential']

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def uniform_weighting():
    # give each forecast a weight of 1
    def weight_fn(forecasts: pd.DataFrame) -> pd.DataFrame:
        forecasts = forecasts.copy()
        forecasts['weight'] = 1.0
        return forecasts
    return weight_fn


def first_n_weighting(n = 1, group_col: list[str] = ['forecaster', 'event_ticker'], time_col: str = 'round'):
    # give the first n forecasts a weight of 1, and the rest directly filtered out
    def weight_fn(forecasts: pd.DataFrame) -> pd.DataFrame:
        forecasts = forecasts.copy()
        forecasts = forecasts.sort_values(by=time_col, ascending=True)
        forecasts = forecasts.groupby(group_col).head(n)
        forecasts['weight'] = 1.0
        return forecasts
    return weight_fn


def last_n_weighting(n = 1, group_col: list[str] = ['forecaster', 'event_ticker'], time_col: str = 'round'):
    # give the last n forecasts a weight of 1, and the rest directly filtered out
    def weight_fn(forecasts: pd.DataFrame) -> pd.DataFrame:
        forecasts = forecasts.copy()
        forecasts = forecasts.sort_values(by=time_col, ascending=True)
        forecasts = forecasts.groupby(group_col).tail(n)
        forecasts['weight'] = 1.0
        return forecasts
    return weight_fn


def exponential_weighting(lambda_ = 0.1, time_col: str = 'time_rank'):
    # give the forecasts a weight of e^(-lambda * relative_time), where relative_time is the positional distance from the most recent forecast
    def weight_fn(forecasts: pd.DataFrame) -> pd.DataFrame:
        forecasts = forecasts.copy()
        forecasts['weight'] = np.exp(-lambda_ * forecasts[time_col])
        return forecasts
    return weight_fn


def time_to_last_weighting(min_hours: float = 0.0, max_hours: float = float('inf')):
    """
    Filter predictions based on their time gap to the market close time.
    
    This weighting function filters predictions based on how many hours before market close
    they were made. It automatically calculates 'time_to_last' if not present using the
    calculate_time_to_last_submission function from utils.py.
    
    Special handling for single-submission events:
    - Events with only one submission are ALWAYS included regardless of time range
    - This prevents filtering out events that had no opportunity for multiple predictions
    
    Args:
        min_hours: Minimum hours before market close (inclusive). Default: 0.0
        max_hours: Maximum hours before market close (exclusive). Default: inf (no upper limit)
    
    Returns:
        A weighting function that filters predictions within [min_hours, max_hours) and assigns weight=1.0
    
    Example:
        # Only keep predictions made 6-12 hours before market close
        weight_fn = time_to_last_weighting(min_hours=6.0, max_hours=12.0)
        
        # Only keep predictions made more than 24 hours before market close
        weight_fn = time_to_last_weighting(min_hours=24.0, max_hours=float('inf'))
        
        # Only keep predictions made within 3 hours of market close
        weight_fn = time_to_last_weighting(min_hours=0.0, max_hours=3.0)
    """
    def weight_fn(forecasts: pd.DataFrame) -> pd.DataFrame:        
        # Check if time_to_last column exists
        if 'time_to_last' not in forecasts.columns:
            from pm_rank.nightly.utils import calculate_time_to_last_submission
            forecasts = calculate_time_to_last_submission(forecasts)
        
        # Filter logic: for events with multiple submissions, only keep predictions within the time range
        mask = ((forecasts['time_to_last'] >= min_hours) & (forecasts['time_to_last'] < max_hours))
        
        num_before = len(forecasts)
        forecasts = forecasts[mask]
        num_after = len(forecasts)
        
        forecasts['weight'] = 1.0

        print(f"Time-based filtering: Retained {num_after}/{num_before} predictions")
        print(f"  - Time range: [{min_hours}, {max_hours}) hours before market close")
        return forecasts
    
    return weight_fn


class NightlyForecasts:

    PREDICTION_COLS = ['predictor_name', 'event_ticker', 'submission_count', 'prediction', 'market_outcome', 'category']
    SUBMISSION_COLS = ['event_ticker', 'submission_count', 'market_data', 'snapshot_time', 'close_time']

    RENAMES = {
        'predictor_name': 'forecaster',
        'submission_count': 'round',
        'market_outcome': 'outcome'
    }

    def __init__(self, forecasts: pd.DataFrame, exclude_forecasters: list[str] = None):
        prev_len = len(forecasts)
        if exclude_forecasters is not None:
            forecasts = forecasts[~forecasts['forecaster'].isin(exclude_forecasters)]
            print(f"Filtered out {prev_len - len(forecasts)} forecasts. Remaining {len(forecasts)}.")
        self.data = forecasts

    @staticmethod
    def turn_market_data_to_odds(market_data: dict) -> tuple[np.ndarray, np.ndarray]:
        # sort the list to ensure market consistency
        markets = sorted(list(market_data.keys()))
        yes_asks = np.array([market_data[mkt]['yes_ask'] / 100.0 for mkt in markets])
        no_asks = np.array([market_data[mkt]['no_ask'] / 100.0 for mkt in markets])
        return yes_asks, no_asks

    @staticmethod
    def simplify_prediction(prediction: dict) -> np.ndarray:
        prediction = {item['market']: item['probability'] for item in prediction['probabilities']}
        return np.array([prediction[mkt] for mkt in sorted(list(prediction.keys()))])

    @staticmethod
    def simplify_market_outcome(market_outcome: dict) -> np.ndarray:
        return np.array([market_outcome[mkt] for mkt in sorted(list(market_outcome.keys()))])

    @classmethod
    def from_prophet_arena_csv(cls, predictions_csv: str, submissions_csv: str, weight_fn = uniform_weighting(), exclude_forecasters: list[str] = None):
        logger.info(f"Loading forecasts from {predictions_csv} and {submissions_csv}")
        logger.info(f"Weighting function: {weight_fn}")
        # Load CSVs
        predictions_df = pd.read_csv(predictions_csv)[cls.PREDICTION_COLS]
        submissions_df = pd.read_csv(submissions_csv)[cls.SUBMISSION_COLS]
        
        # Parse JSON columns
        predictions_df['prediction'] = predictions_df['prediction'].apply(json.loads).apply(cls.simplify_prediction)
        predictions_df['market_outcome'] = predictions_df['market_outcome'].apply(json.loads).apply(cls.simplify_market_outcome)
        submissions_df['market_data'] = submissions_df['market_data'].apply(json.loads)

        # Convert the `market_data` in submissions_df to a list of odds & no_odds
        submissions_df['odds'], submissions_df['no_odds'] = zip(*submissions_df['market_data'].apply(cls.turn_market_data_to_odds))

        # Merge predictions with submissions for the odds and no_odds columns
        merged = predictions_df.merge(
            submissions_df[['event_ticker', 'submission_count', 'odds', 'no_odds', 'snapshot_time', 'close_time']],
            on=['event_ticker', 'submission_count'],
            how='inner'
        )

        # We leave only rows where the `odds`, `prediction`, `market_outcome` columns have the same length
        odds_len, prediction_len, market_outcome_len = merged['odds'].apply(len), merged['prediction'].apply(len), merged['market_outcome'].apply(len)
        merged = merged[(odds_len == prediction_len) & (odds_len == market_outcome_len)]

        # Rename predictor_name to forecaster
        merged = merged.rename(columns=cls.RENAMES)

        # Add `relative_round` column
        merged['time_rank'] = merged.groupby(['forecaster', 'event_ticker'])['round'].rank(ascending=False) - 1

        # Apply the weighting function
        merged = weight_fn(merged)

        logger.info(f"Loaded {len(merged)} rows")

        return cls(merged, exclude_forecasters)
