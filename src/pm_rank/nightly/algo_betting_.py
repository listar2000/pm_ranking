import numpy as np
import pandas as pd


def compute_individualized_market_brier(forecasts: pd.DataFrame) -> dict:
    """
    Compute per-forecaster individualized market baseline Brier scores.

    For each forecaster, uses the same filtered event set (dedup) and
    computes the Brier score as if the market prices were the predictions:
      - YES side: (yes_ask - outcome)^2
      - NO side: (no_ask - (1 - outcome))^2  [uses actual no_odds, NOT 1 - odds]
    Direction is determined by the forecaster's actual prediction vs odds.
    Forecaster-events with any market prediction inside the bid-ask spread are skipped.

    Returns:
        Dict mapping forecaster name -> individualized market baseline Brier score.
    """

    def _arrays_same_length(row):
        try:
            return len(row['prediction']) == len(row['odds']) == len(row['no_odds']) == len(row['outcome'])
        except Exception:
            return False

    length_mask = forecasts.apply(_arrays_same_length, axis=1)
    df = forecasts[length_mask].copy()

    median_rounds = df.groupby(['forecaster', 'event_ticker'])['round'].median().reset_index()
    median_rounds.columns = ['forecaster', 'event_ticker', 'median_round']
    df = df.merge(median_rounds, on=['forecaster', 'event_ticker'])
    df['round_dist'] = (df['round'] - df['median_round']).abs()
    df = df.sort_values('round_dist').drop_duplicates(subset=['forecaster', 'event_ticker'], keep='first')
    df = df.drop(columns=['median_round', 'round_dist'])

    forecaster_total_brier = {}
    forecaster_total_markets = {}
    skipped_within_spread = 0

    for _, row in df.iterrows():
        try:
            prediction = row['prediction']
            outcome = row['outcome']
            odds = row['odds']
            no_odds = row['no_odds']

            if len(prediction) == 0:
                continue

            market_yes_brier = (odds - outcome) ** 2
            market_no_brier = (no_odds - (1.0 - outcome)) ** 2
            per_market_brier = np.where(prediction > odds, market_yes_brier, market_no_brier)

            within_spread = (prediction >= (1 - no_odds)) & (prediction <= odds)
            if np.any(within_spread):
                skipped_within_spread += 1
                continue

            forecaster = row['forecaster']
            n_markets = len(per_market_brier)
            forecaster_total_brier[forecaster] = forecaster_total_brier.get(forecaster, 0.0) + np.sum(per_market_brier)
            forecaster_total_markets[forecaster] = forecaster_total_markets.get(forecaster, 0) + n_markets

        except Exception:
            continue

    if skipped_within_spread > 0:
        print(f"Skipped {skipped_within_spread} individualized market baseline rows with predictions inside the spread")

    result = {}
    for forecaster in forecaster_total_brier:
        if forecaster_total_markets[forecaster] > 0:
            result[forecaster] = forecaster_total_brier[forecaster] / forecaster_total_markets[forecaster]
    return result
