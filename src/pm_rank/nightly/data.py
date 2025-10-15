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


class NightlyForecasts:

    PREDICTION_COLS = ['predictor_name', 'event_ticker', 'submission_count', 'prediction', 'market_outcome', 'category']
    SUBMISSION_COLS = ['event_ticker', 'submission_count', 'market_data']

    RENAMES = {
        'predictor_name': 'forecaster',
        'submission_count': 'round',
        'market_outcome': 'outcome'
    }

    def __init__(self, forecasts: pd.DataFrame):
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
    def from_prophet_arena_csv(cls, predictions_csv: str, submissions_csv: str, weight_fn = uniform_weighting()):
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
            submissions_df[['event_ticker', 'submission_count', 'odds', 'no_odds']],
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

        return cls(merged)


if __name__ == "__main__":
    predictions_csv = "slurm/predictions_10_11_to_09_15.csv"
    submissions_csv = "slurm/submissions_10_11_to_09_15.csv"

    weight_fn = uniform_weighting()
    forecasts = NightlyForecasts.from_prophet_arena_csv(predictions_csv, submissions_csv, weight_fn)
    
    df = forecasts.data

    from pm_rank.nightly.algo import compute_brier_score, compute_average_return_neutral, rank_forecasters_by_score, add_market_baseline_predictions, compute_calibration_ece

    df = add_market_baseline_predictions(df)
    
    # Test Brier Score
    print("=" * 50)
    print("BRIER SCORE RANKINGS")
    print("=" * 50)
    brier_scores = compute_brier_score(df)
    print(rank_forecasters_by_score(brier_scores))
    print("\nWith round normalization:")
    print(rank_forecasters_by_score(brier_scores, normalize_by_round=True))
    
    # Test Average Return
    print("\n" + "=" * 50)
    print("AVERAGE RETURN RANKINGS (Risk Neutral)")
    print("=" * 50)
    avg_returns = compute_average_return_neutral(df, spread_market_even=True)
    print(rank_forecasters_by_score(avg_returns, normalize_by_round=False))
    print("\nWith round normalization:")
    print(rank_forecasters_by_score(avg_returns, normalize_by_round=True))
    print("\nWith Bootstrap CI (100 samples):")
    bootstrap_config = {'num_samples': 100, 'num_se': 1.96, 'random_seed': 42, 'show_progress': False}
    print(rank_forecasters_by_score(avg_returns, normalize_by_round=True, bootstrap_config=bootstrap_config))
    
    # Test Calibration (ECE)
    print("\n" + "=" * 50)
    print("CALIBRATION (ECE) RANKINGS")
    print("=" * 50)
    ece_results = compute_calibration_ece(df, num_bins=10, strategy="uniform", weight_event=True)
    print(rank_forecasters_by_score(ece_results))
