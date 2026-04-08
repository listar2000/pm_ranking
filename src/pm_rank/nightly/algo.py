import pandas as pd
import numpy as np
from typing import Literal, Dict, Optional, Tuple
from pm_rank.model.calibration import _bin_stats, _calculate_ece
from pm_rank.nightly.bootstrap import compute_bootstrap_ci
from tqdm import tqdm


DEFAULT_BOOTSTRAP_CONFIG = {
    'num_samples': 1000,
    'ci_level': 0.9,
    'num_se': None,
    'random_seed': 42,
    'show_progress': True
}
DEFAULT_MAX_SPREAD = 1.03


def add_individualized_market_baselines_to_scores(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add individualized market baseline scores for each forecaster at aggregation time.
    
    This function takes per-forecast scores (e.g., from compute_brier_score or compute_average_return_neutral)
    and creates "{forecaster}-market-baseline" entries by filtering the market-baseline scores to only
    the (event_ticker, round) combinations where each forecaster participated.
    
    This is efficient because it reuses the already-computed market-baseline scores rather than
    creating duplicate prediction rows.
    
    Args:
        result_df: DataFrame with columns (forecaster, event_ticker, round, weight, <score_col>)
                   Must contain a 'market-baseline' forecaster.
    
    Returns:
        DataFrame with added "{forecaster}-market-baseline" rows for each real forecaster.
    """
    if 'market-baseline' not in result_df['forecaster'].values:
        return result_df  # No market-baseline to work with
    
    # Get the market-baseline scores
    market_baseline_scores = result_df[result_df['forecaster'] == 'market-baseline'].copy()
    
    # Get unique real forecasters (excluding market-baseline and any existing individualized baselines)
    real_forecasters = result_df[
        ~result_df['forecaster'].str.contains('-market-baseline', na=False) & 
        (result_df['forecaster'] != 'market-baseline')
    ]['forecaster'].unique()
    
    individualized_baselines = []
    
    for forecaster in tqdm(real_forecasters, desc="Adding individualized market baselines"):
        # Get this forecaster's (event_ticker, round) combinations
        forecaster_data = result_df[result_df['forecaster'] == forecaster]
        forecaster_keys = forecaster_data[['event_ticker', 'round']].drop_duplicates()
        
        # Filter market-baseline scores to only these combinations
        individualized = market_baseline_scores.merge(
            forecaster_keys,
            on=['event_ticker', 'round'],
            how='inner'
        ).copy()
        
        # Also copy the weight from the original forecaster's data
        # This ensures proper weighting when aggregating
        weight_map = forecaster_data.set_index(['event_ticker', 'round'])['weight'].to_dict()
        individualized['weight'] = individualized.apply(
            lambda row: weight_map.get((row['event_ticker'], row['round']), row['weight']),
            axis=1
        )
        
        # Set the forecaster name
        individualized['forecaster'] = f'{forecaster}-market-baseline'
        
        individualized_baselines.append(individualized)
    
    # Concatenate all individualized baselines with original result_df
    if individualized_baselines:
        result = pd.concat([result_df] + individualized_baselines, ignore_index=True)
    else:
        result = result_df
    
    return result


def rank_forecasters_by_score(result_df: pd.DataFrame, normalize_by_round: bool = False,
                              score_col: str = None, ascending: bool = None,
                              bootstrap_config: Optional[Dict] = None,
                              add_individualized_baselines: bool = False,
                              bootstrap_scores_df: pd.DataFrame = None,
                              aggregation: str = 'mean',
                              analytical_ci: bool = False,
                              analytical_ci_level: float = 0.95) -> pd.DataFrame:
    """
    Return a rank_df with columns (forecaster, rank, score).

    Args:
        result_df: DataFrame containing forecaster scores (used for point estimates and ranking)
        normalize_by_round: If True, downweight by the number of rounds per (forecaster, event_ticker) group
                           (ignored for ECE scores which are already aggregated)
        score_col: Name of the score column to rank by. If None, auto-detects from {'brier_score', 'average_return', 'ece_score'}
        ascending: Whether lower scores are better (True for Brier/ECE, False for returns). If None, auto-detects.
        bootstrap_config: Optional dict with bootstrap parameters for CI estimation:
            - num_samples: Number of bootstrap samples (default: 1000)
            - ci_level: Confidence level (default: 0.95)
            - num_se: Number of standard errors for CI bounds (default: None, uses ci_level)
            - random_seed: Random seed for reproducibility (default: 42)
            - show_progress: Whether to show progress bar (default: True)
            Only supported for 'brier_score' and 'average_return', not 'ece_score'.
        add_individualized_baselines: If True, create "{forecaster}-market-baseline" entries for each
            forecaster by filtering market-baseline scores to their participated (event_ticker, round)
            combinations. Requires 'market-baseline' forecaster to be present in result_df.
        bootstrap_scores_df: Optional separate DataFrame for bootstrap resampling. When provided,
            the point estimate is computed from result_df (event-level) while bootstrap CI is
            computed from this DataFrame (e.g. market-level). Must contain the same score column
            and 'forecaster' column. If None, bootstrap uses result_df.
        aggregation: Aggregation mode for score_col.
            - 'mean': weighted mean of score_col (default)
            - 'sum': sum of score_col
            - 'roi': return on investment, computed as
              sum(score_col) / sum(cost). Requires a 'cost' column.
        analytical_ci: If True (and bootstrap_config is None and aggregation == 'mean'),
            compute a closed-form normal-approximation CI from the per-row score variance
            instead of bootstrap. Much cheaper than bootstrap; appropriate for proper
            scoring rules where the per-row scores are independent and the aggregator is
            the (weighted) sample mean. Mutually exclusive with ``bootstrap_config``.
        analytical_ci_level: Confidence level for the analytical CI (default 0.95). Maps
            to a normal-approximation z-score (e.g. 0.95 → 1.96).

    Returns:
        DataFrame with rank as index and columns (forecaster, score).
        If bootstrap_config or analytical_ci is set, also includes a "{level}% ci"
        column formatted as "±X.XXXX".
    """
    if bootstrap_config is not None and analytical_ci:
        raise ValueError("Pass either bootstrap_config or analytical_ci=True, not both")
    df = result_df.copy()

    if df.empty:
        return pd.DataFrame(columns=['forecaster', 'score']).rename_axis('rank')
    
    # Auto-detect score column if not provided
    if score_col is None:
        if 'brier_score' in df.columns:
            score_col = 'brier_score'
        elif 'average_return' in df.columns:
            score_col = 'average_return'
        elif 'ece_score' in df.columns:
            score_col = 'ece_score'
        elif 'sharpe_ratio' in df.columns:
            score_col = 'sharpe_ratio'
        else:
            raise ValueError("Could not find score column. Please specify 'score_col' parameter.")
    
    # Auto-detect ascending if not provided
    if ascending is None:
        # Lower is better for Brier score and ECE score, higher is better for average return
        ascending = (score_col in ['brier_score', 'ece_score'])
    
    # Special handling for ECE scores and Sharpe ratio scores: they're already aggregated per forecaster
    if score_col in ['ece_score', 'sharpe_ratio']:
        if bootstrap_config is not None:
            raise ValueError(f"Bootstrap CI is not supported for {score_col} scores (already aggregated)")
        
        # These scores are already computed per forecaster, just need to rank and format
        forecaster_scores = df[['forecaster', score_col]].copy()
        forecaster_scores.columns = ['forecaster', 'score']
        
        # Rank forecasters (ascending=True means lower score = better rank)
        forecaster_scores['rank'] = forecaster_scores['score'].rank(method='min', ascending=ascending).astype(int)
        
        # Sort by rank and select required columns, then set rank as index
        rank_df = forecaster_scores[['forecaster', 'rank', 'score']].sort_values('rank')
        rank_df = rank_df.set_index('rank')[['forecaster', 'score']]
        
        return rank_df

    # Optionally add individualized market baselines before aggregation.
    if add_individualized_baselines:
        df = add_individualized_market_baselines_to_scores(df)
    
    # Drop rows with NaN scores (e.g. illiquid events skipped by average return)
    df = df.dropna(subset=[score_col])
    if df.empty:
        return pd.DataFrame(columns=['forecaster', 'score']).rename_axis('rank')

    # For other metrics (Brier, average return), perform weighted aggregation
    if normalize_by_round:
        # For each (forecaster, event_ticker) group, downweight by number of unique rounds
        # Using nunique on 'round' ensures correctness for both event-level and market-level data
        round_counts = df.groupby(['forecaster', 'event_ticker'])['round'].nunique().reset_index(name='round_count')
        
        # Merge back to get round_count for each row
        df = df.merge(round_counts, on=['forecaster', 'event_ticker'])
        
        # Adjust weights by dividing by round_count
        df['adjusted_weight'] = df['weight'] / df['round_count']
    else:
        df['adjusted_weight'] = df['weight']
    
    # Calculate the aggregated score for each forecaster.
    if aggregation == 'roi':
        if 'cost' not in df.columns:
            raise ValueError("ROI aggregation requires a 'cost' column in result_df")
        forecaster_scores = df.groupby('forecaster').apply(
            lambda group: group[score_col].sum() / group['cost'].sum()
            if group['cost'].sum() > 0 else 0.0,
        ).reset_index(name='score')
    elif aggregation == 'sum':
        forecaster_scores = df.groupby('forecaster').apply(
            lambda group: group[score_col].sum()
        ).reset_index(name='score')
    else:
        forecaster_scores = df.groupby('forecaster').apply(
            lambda group: np.dot(group[score_col], group['adjusted_weight']) / np.sum(group['adjusted_weight'])
        ).reset_index(name='score')
    
    # Count predictions per forecaster
    prediction_counts = df.groupby('forecaster').size()
    forecaster_scores['# predictions'] = forecaster_scores['forecaster'].map(prediction_counts)
    
    ci_col_name = None
    if analytical_ci and aggregation != 'mean':
        # Closed-form CI only makes sense when the aggregator is the (weighted)
        # sample mean of independent per-row scores. ROI involves a ratio of
        # two random sums and would need the delta method; punt for now.
        raise ValueError("analytical_ci=True is only supported for aggregation='mean'")

    if analytical_ci:
        # Normal-approximation CI: SE = sample_std / sqrt(n_effective).
        # For uniform weights this is just std/sqrt(n); for non-uniform weights
        # we use Kish's effective sample size n_eff = (sum w)^2 / sum(w^2).
        from math import erf, sqrt as _sqrt

        ci_col_name = f'{analytical_ci_level * 100}% ci'
        # Approximate the inverse normal CDF for the requested level via a
        # binary search rather than pulling in scipy. The CI level is symmetric
        # so we want z such that P(|Z| < z) = analytical_ci_level.
        target = (1.0 + analytical_ci_level) / 2.0
        lo, hi = 0.0, 6.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            cdf_mid = 0.5 * (1.0 + erf(mid / _sqrt(2.0)))
            if cdf_mid < target:
                lo = mid
            else:
                hi = mid
        z_score = (lo + hi) / 2.0

        analytical_intervals: Dict[str, Tuple[float, float]] = {}
        for forecaster, group in df.groupby('forecaster'):
            scores_arr = group[score_col].to_numpy()
            weights_arr = group['adjusted_weight'].to_numpy()
            n = scores_arr.shape[0]
            if n == 0:
                analytical_intervals[str(forecaster)] = (np.nan, np.nan)
                continue
            weight_sum = float(np.sum(weights_arr))
            if weight_sum <= 0:
                analytical_intervals[str(forecaster)] = (np.nan, np.nan)
                continue
            point = float(np.average(scores_arr, weights=weights_arr))
            if n == 1:
                # Variance is undefined with a single sample.
                analytical_intervals[str(forecaster)] = (point, point)
                continue
            # Weighted sample variance (unbiased): see e.g. NIST handbook 4.3.6.
            weighted_sq = np.sum(weights_arr * (scores_arr - point) ** 2)
            # Effective sample size for weighted samples (Kish).
            n_eff = (weight_sum ** 2) / float(np.sum(weights_arr ** 2))
            if n_eff <= 1:
                analytical_intervals[str(forecaster)] = (point, point)
                continue
            variance = weighted_sq / weight_sum * (n_eff / (n_eff - 1.0))
            se = float(np.sqrt(variance / n_eff))
            margin = z_score * se
            analytical_intervals[str(forecaster)] = (point - margin, point + margin)

        forecaster_scores[ci_col_name] = forecaster_scores['forecaster'].map(
            lambda f: (
                f"±{(analytical_intervals[f][1] - analytical_intervals[f][0]) / 2:.4f}"
                if f in analytical_intervals
                and not np.isnan(analytical_intervals[f][0])
                and not np.isnan(analytical_intervals[f][1])
                else np.nan
            )
        )

    if bootstrap_config is not None:
        # Compute bootstrap confidence intervals if requested
        ci_level = bootstrap_config.get('ci_level', DEFAULT_BOOTSTRAP_CONFIG['ci_level'])
        ci_col_name = f'{ci_level * 100}% ci'

        # Use separate bootstrap DataFrame if provided (e.g. market-level data)
        if bootstrap_scores_df is not None:
            bs_df = bootstrap_scores_df.copy()
            if add_individualized_baselines:
                bs_df = add_individualized_market_baselines_to_scores(bs_df)
            bs_df = bs_df.dropna(subset=[score_col])
            if normalize_by_round:
                bs_round_counts = bs_df.groupby(['forecaster', 'event_ticker'])['round'].nunique().reset_index(name='round_count')
                bs_df = bs_df.merge(bs_round_counts, on=['forecaster', 'event_ticker'])
                bs_df['adjusted_weight'] = bs_df['weight'] / bs_df['round_count']
            else:
                bs_df['adjusted_weight'] = bs_df['weight']
            bs_cols = ['forecaster', score_col]
            if aggregation == 'roi':
                if 'cost' not in bs_df.columns:
                    raise ValueError("ROI aggregation requires a 'cost' column in bootstrap_scores_df")
                bs_cols.append('cost')
            bs_input = bs_df[bs_cols].copy()
            bs_weights = bs_df['adjusted_weight'].values
        else:
            bs_cols = ['forecaster', score_col]
            if aggregation == 'roi':
                bs_cols.append('cost')
            bs_input = df[bs_cols].copy()
            bs_weights = df['adjusted_weight'].values

        if bs_input.empty:
            confidence_intervals = {}
        else:
            _, confidence_intervals = compute_bootstrap_ci(
                bs_input,
                score_col,
                bs_weights,
                bootstrap_config,
                aggregation=aggregation,
            )
        
        # Add SE and CI columns to forecaster_scores
        # forecaster_scores['se'] = forecaster_scores['forecaster'].map(standard_errors)
        # forecaster_scores['lower'] = forecaster_scores['forecaster'].map(lambda f: confidence_intervals[f][0])
        # forecaster_scores['upper'] = forecaster_scores['forecaster'].map(lambda f: confidence_intervals[f][1])
        forecaster_scores[ci_col_name] = forecaster_scores['forecaster'].map(
            lambda f: (
                f"±{(confidence_intervals[f][1] - confidence_intervals[f][0]) / 2:.4f}"
                if f in confidence_intervals and not np.isnan(confidence_intervals[f][0]) and not np.isnan(confidence_intervals[f][1])
                else np.nan
            )
        )
    
    # Rank forecasters (ascending=True means lower score = better rank)
    forecaster_scores['rank'] = forecaster_scores['score'].rank(method='min', ascending=ascending).astype(int)

    # Sort by rank and select required columns, then set rank as index
    if ci_col_name is not None:
        rank_df = forecaster_scores[['forecaster', 'rank', 'score', ci_col_name, '# predictions']].sort_values('rank')
        rank_df = rank_df.set_index('rank')[['forecaster', 'score', ci_col_name, '# predictions']]
    else:
        rank_df = forecaster_scores[['forecaster', 'rank', 'score', '# predictions']].sort_values('rank')
        rank_df = rank_df.set_index('rank')[['forecaster', 'score', '# predictions']]

    return rank_df


def add_market_baseline_predictions(forecasts: pd.DataFrame, reference_forecaster: str = None, use_both_sides: bool = False) -> pd.DataFrame:
    """
    We turn the forecasts from a certain forecaster into market baseline predictions.
    If use_both_sides is True, we will add the market baseline predictions for both YES and NO sides.

    Args:
        forecasts: DataFrame with columns (forecaster, event_ticker, round, prediction, outcome, weight)
        reference_forecaster: The forecaster to use as the reference for the market baseline predictions
        use_both_sides: If True, we will add the market baseline predictions for both YES and NO sides
    """
    if reference_forecaster is None:
        # if no reference forecaster is provided, we take the union of all forecasts from all forecasters
        market_baseline_forecasts = forecasts.groupby(['event_ticker', 'round'], as_index=False).first()
    else:
        market_baseline_forecasts = forecasts[forecasts['forecaster'] == reference_forecaster].copy()

    market_baseline_forecasts['forecaster'] = 'market-baseline'

    def turn_odds_to_prediction(row: pd.Series) -> np.ndarray:
        odds, no_odds = row['odds'], row['no_odds']
        if use_both_sides:
            return np.array([(odds[i] + (1 - no_odds[i])) / 2.0 for i in range(len(odds))])
        else:
            return np.array([odds[i] for i in range(len(odds))])

    market_baseline_forecasts['prediction'] = market_baseline_forecasts.apply(turn_odds_to_prediction, axis=1)

    # concat market_baseline_forecasts with forecasts (take care of the pd index as well)
    forecasts = pd.concat([forecasts, market_baseline_forecasts]).reset_index(drop=True)
    return forecasts


def _filter_length_matched_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose prediction/odds/outcome arrays line up."""
    def _arrays_same_length(row):
        try:
            return len(row['prediction']) == len(row['odds']) == len(row['no_odds']) == len(row['outcome'])
        except Exception:
            return False

    length_mask = forecasts.apply(_arrays_same_length, axis=1)
    filtered_count = (~length_mask).sum()
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} rows with mismatched array lengths")
    return forecasts[length_mask].copy()


def _dedupe_to_median_round(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Keep the round closest to the median for each (forecaster, event_ticker)."""
    if forecasts.empty:
        return forecasts.copy()

    median_rounds = forecasts.groupby(['forecaster', 'event_ticker'])['round'].median().reset_index()
    median_rounds.columns = ['forecaster', 'event_ticker', 'median_round']
    deduped = forecasts.merge(median_rounds, on=['forecaster', 'event_ticker'])
    deduped['round_dist'] = (deduped['round'] - deduped['median_round']).abs()
    deduped = deduped.sort_values('round_dist').drop_duplicates(
        subset=['forecaster', 'event_ticker'],
        keep='first',
    )
    deduped = deduped.drop(columns=['median_round', 'round_dist'])
    print(f"After median-round dedup: {len(deduped)} rows (1 per forecaster-event)")
    return deduped


def compute_brier_score(forecasts: pd.DataFrame, per_market: bool = False, max_spread: float = DEFAULT_MAX_SPREAD) -> pd.DataFrame:
    """
    Calculate the Brier score for the forecasts using row-by-row processing.
    Handles predictions with different array lengths via key intersection.
    Automatically filters out illiquid events (yes_ask + no_ask > max_spread).

    The result will be a DataFrame containing (forecaster, event_ticker, round, weight, brier_score).
    If per_market=True, returns one row per individual market (outcome) within each event,
    with brier_score = (prediction_i - outcome_i)^2 and weight = event_weight / num_markets.

    Args:
        forecasts: DataFrame with columns (forecaster, event_ticker, round, prediction, outcome, weight, odds, no_odds)
        per_market: If True, return one row per market instead of one per event. Each market's
                    Brier score is the individual squared error, and its weight is the event weight
                    divided by the number of markets. This preserves the same weighted-average
                    point estimate as event-level scoring.
        max_spread: Maximum allowed spread for illiquidity filtering.
    """
    result_df = _dedupe_to_median_round(_filter_length_matched_forecasts(forecasts))

    skipped_illiquid = 0

    if per_market:
        # Output one row per market within each event
        market_rows = []
        for _, row in result_df.iterrows():
            try:
                prediction = np.asarray(row['prediction'], dtype=float)
                outcome = np.asarray(row['outcome'], dtype=float)
                odds = np.asarray(row['odds'], dtype=float)
                no_odds = np.asarray(row['no_odds'], dtype=float)

                spreads = odds + no_odds
                if np.any(spreads > max_spread):
                    skipped_illiquid += 1
                    continue

                if len(prediction) == 0:
                    continue

                num_markets = len(prediction)
                event_weight = float(num_markets)
                market_weight = event_weight / num_markets

                for i in range(num_markets):
                    market_rows.append({
                        'forecaster': row['forecaster'],
                        'event_ticker': row['event_ticker'],
                        'round': row['round'],
                        'market_index': i,
                        'weight': market_weight,
                        'brier_score': (prediction[i] - outcome[i]) ** 2
                    })
            except Exception:
                continue

        if skipped_illiquid > 0:
            print(f"Skipped {skipped_illiquid} illiquid events (spread > {max_spread})")

        result_df = pd.DataFrame(market_rows)
        return result_df
    else:
        # Original behavior: one row per event with averaged Brier score
        result_df['brier_score'] = np.nan

        for idx, row in result_df.iterrows():
            try:
                prediction = np.asarray(row['prediction'], dtype=float)
                outcome = np.asarray(row['outcome'], dtype=float)
                odds = np.asarray(row['odds'], dtype=float)
                no_odds = np.asarray(row['no_odds'], dtype=float)

                spreads = odds + no_odds
                if np.any(spreads > max_spread):
                    skipped_illiquid += 1
                    continue

                if len(prediction) == 0:
                    continue

                brier_score = np.mean((prediction - outcome) ** 2)
                result_df.at[idx, 'brier_score'] = brier_score
                result_df.at[idx, 'weight'] = float(len(prediction))

            except Exception:
                continue

        if skipped_illiquid > 0:
            print(f"Skipped {skipped_illiquid} illiquid events (spread > {max_spread})")

        result_df = result_df.dropna(subset=['brier_score'])
        result_df = result_df[['forecaster', 'event_ticker', 'weight', 'round', 'brier_score']]
        return result_df


def compute_global_market_brier(
    forecasts: pd.DataFrame,
    max_spread: float = DEFAULT_MAX_SPREAD,
) -> float:
    """
    Compute global market baseline Brier score: (yes_brier + no_brier) / 2.

    For each market across all events:
      - yes_brier = (odds - outcome)^2
      - no_brier = (no_odds - (1 - outcome))^2
      - market_brier = (yes_brier + no_brier) / 2

    Returns a single global average across all markets.
    """

    def _arrays_same_length(row):
        try:
            return len(row['odds']) == len(row['no_odds']) == len(row['outcome'])
        except Exception:
            return False

    length_mask = forecasts.apply(_arrays_same_length, axis=1)
    df = forecasts[length_mask].copy()

    # One row per event (market data is the same for all forecasters)
    df = df.drop_duplicates(subset=['event_ticker'], keep='first')

    total_brier = 0.0
    total_markets = 0
    skipped_illiquid = 0

    for _, row in df.iterrows():
        try:
            outcome = np.asarray(row['outcome'], dtype=float)
            odds = np.asarray(row['odds'], dtype=float)
            no_odds = np.asarray(row['no_odds'], dtype=float)

            if len(odds) == 0:
                continue

            spreads = odds + no_odds
            if np.any(spreads > max_spread):
                skipped_illiquid += 1
                continue

            market_yes_brier = (odds - outcome) ** 2
            market_no_brier = (no_odds - (1.0 - outcome)) ** 2
            per_market_brier = (market_yes_brier + market_no_brier) / 2.0

            total_brier += np.sum(per_market_brier)
            total_markets += len(per_market_brier)

        except Exception:
            continue

    if skipped_illiquid > 0:
        print(f"Skipped {skipped_illiquid} illiquid events (spread > {max_spread})")

    if total_markets == 0:
        return 0.0
    return total_brier / total_markets


def compute_per_forecaster_market_brier(
    forecasts: pd.DataFrame,
    max_spread: float = DEFAULT_MAX_SPREAD,
) -> Dict[str, float]:
    """
    Compute individualized market baseline Brier score per forecaster.

    For each forecaster, computes the market baseline Brier only over the events
    that forecaster participated in (using the median round per event).

    Returns:
        Dict mapping forecaster name -> market baseline Brier score (raw, not 1-brier).
    """

    def _arrays_same_length(row):
        try:
            return len(row['odds']) == len(row['no_odds']) == len(row['outcome'])
        except Exception:
            return False

    length_mask = forecasts.apply(_arrays_same_length, axis=1)
    df = forecasts[length_mask].copy()

    # Exclude the synthetic market-baseline forecaster
    df = df[df['forecaster'] != 'market-baseline']

    # Build per-event market brier lookup (one row per event)
    events_df = df.drop_duplicates(subset=['event_ticker'], keep='first')
    event_brier: Dict[str, float] = {}
    event_market_count: Dict[str, int] = {}
    skipped_illiquid = 0
    for _, row in events_df.iterrows():
        try:
            outcome = np.asarray(row['outcome'], dtype=float)
            odds = np.asarray(row['odds'], dtype=float)
            no_odds = np.asarray(row['no_odds'], dtype=float)
            if len(odds) == 0:
                continue
            spreads = odds + no_odds
            if np.any(spreads > max_spread):
                skipped_illiquid += 1
                continue
            yes_b = (odds - outcome) ** 2
            no_b = (no_odds - (1.0 - outcome)) ** 2
            per_market = (yes_b + no_b) / 2.0
            event_brier[row['event_ticker']] = float(np.sum(per_market))
            event_market_count[row['event_ticker']] = len(per_market)
        except Exception:
            continue

    # Deduplicate to one row per (forecaster, event_ticker) using median round
    deduped = _dedupe_to_median_round(df)

    result: Dict[str, float] = {}
    for forecaster, group in deduped.groupby('forecaster'):
        total_brier = 0.0
        total_markets = 0
        for evt in group['event_ticker'].unique():
            if evt in event_brier:
                total_brier += event_brier[evt]
                total_markets += event_market_count[evt]
        if total_markets > 0:
            result[str(forecaster)] = total_brier / total_markets
    if skipped_illiquid > 0:
        print(f"Skipped {skipped_illiquid} illiquid events (spread > {max_spread})")
    return result


def compute_avg_market_distance(
    forecasts: pd.DataFrame,
    max_spread: float = DEFAULT_MAX_SPREAD,
) -> dict:
    """
    Compute the average L2 distance |prediction - yes_ask| per forecaster across all markets.
    Uses the same dedup filters as compute_brier_score and skips illiquid events
    where any market has yes_ask + no_ask > max_spread.

    Returns:
        Dict mapping forecaster name -> average |prediction - odds| across all markets.
    """

    df = _dedupe_to_median_round(_filter_length_matched_forecasts(forecasts))

    # Accumulate per-forecaster: total distance and total market count
    forecaster_total_dist = {}
    forecaster_total_markets = {}
    skipped_illiquid = 0

    for _, row in df.iterrows():
        try:
            prediction = np.asarray(row['prediction'], dtype=float)
            odds = np.asarray(row['odds'], dtype=float)
            no_odds = np.asarray(row['no_odds'], dtype=float)

            if len(prediction) == 0:
                continue

            spreads = odds + no_odds
            if np.any(spreads > max_spread):
                skipped_illiquid += 1
                continue

            distances = np.where(
                prediction > odds,
                (prediction - odds) ** 2,
                (1 - prediction - no_odds) ** 2
            )

            within_spread = (prediction >= (1 - no_odds)) & (prediction <= odds)
            distances = distances[~within_spread]

            if len(distances) == 0:
                continue

            forecaster = row['forecaster']
            forecaster_total_dist[forecaster] = forecaster_total_dist.get(forecaster, 0.0) + np.sum(distances)
            forecaster_total_markets[forecaster] = forecaster_total_markets.get(forecaster, 0) + len(distances)

        except Exception:
            continue

    result = {}
    for forecaster in forecaster_total_dist:
        if forecaster_total_markets[forecaster] > 0:
            result[forecaster] = forecaster_total_dist[forecaster] / forecaster_total_markets[forecaster]
    if skipped_illiquid > 0:
        print(f"Skipped {skipped_illiquid} illiquid events (spread > {max_spread})")
    return result


def compute_average_return_neutral(forecasts: pd.DataFrame, num_money_per_round: float = 1.0, spread_market_even: bool = False, max_spread: float = DEFAULT_MAX_SPREAD, per_market: bool = False) -> pd.DataFrame:
    """
    Buy market shares directly from the forecaster's edge.

    Betting logic (per market):
        diff = p - yes_ask

        If diff > 0: buy YES at price yes_ask  with shares = p - yes_ask
        If diff < 0: buy NO  at price no_ask   with shares = (1 - p) - no_ask
        If diff = 0: skip

    Spread handling:
        Markets where prediction falls inside the bid-ask spread [1 - no_odds, odds]
        are treated as no-bet markets.

    Liquidity filter:
        Entire events are skipped if ANY market has yes_ask + no_ask > max_spread.

    Args:
        forecasts: DataFrame with columns:
            - forecaster: str, model/forecaster identifier
            - event_ticker: str, event identifier
            - round: int, forecast round number
            - prediction: np.ndarray, forecaster's probability for each outcome
            - outcome: np.ndarray, actual binary outcomes (0 or 1) per market
            - odds: np.ndarray, YES ask prices (implied probabilities) per market
            - no_odds: np.ndarray, NO ask prices per market
            - weight: float, external weight (passed through, not used in computation)
        num_money_per_round: Unused, kept for API compatibility.
        spread_market_even: Unused, kept for API compatibility.
        max_spread: Maximum allowed spread (yes_ask + no_ask) for liquidity filter.
            Events with any outcome exceeding this threshold are skipped entirely.
        per_market: If True, return one row per traded market instead of one per event.
            Each market row keeps the full event weight so ROI aggregation matches
            between event-level and market-level representations.

    Returns:
        DataFrame with columns (forecaster, event_ticker, round, weight, average_return, cost)
        where average_return is net profit and cost is total amount spent.
        If per_market=True, also includes market_index.
    """
    forecasts = _dedupe_to_median_round(_filter_length_matched_forecasts(forecasts))

    result_data = []

    for _, row in forecasts.iterrows():
        forecaster = row['forecaster']

        try:
            prediction = np.asarray(row['prediction'], dtype=float)
            outcome = np.asarray(row['outcome'], dtype=float)
            odds = np.asarray(row['odds'], dtype=float)
            no_odds = np.asarray(row['no_odds'], dtype=float)

            if len(prediction) == 0:
                continue

            spreads = odds + no_odds
            if np.any(spreads > max_spread):
                continue

            diff = prediction - odds
            within_spread = (prediction >= (1 - no_odds)) & (prediction <= odds)
            active_mask = ~within_spread

            if not np.any(active_mask):
                continue

            active_indices = np.where(active_mask)[0]
            prediction = prediction[active_mask]
            outcome = outcome[active_mask]
            odds = odds[active_mask]
            no_odds = no_odds[active_mask]
            diff = diff[active_mask]

            bet_yes = diff > 0
            bet_no = diff < 0
            has_bet = diff != 0
            shares = np.where(bet_yes, prediction - odds, (1 - prediction) - no_odds)
            price = np.where(bet_yes, odds, np.where(bet_no, no_odds, 1.0))

            cost_per_market = np.where(has_bet, shares * price, 0.0)
            is_win = np.where(bet_yes, outcome == 1, outcome == 0)
            payout_per_market = np.where(has_bet & is_win, shares, 0.0)
            profit_per_market = payout_per_market - cost_per_market

            num_markets_bet = int(np.sum(has_bet))

            if per_market:
                traded_indices = active_indices[has_bet]
                traded_profits = profit_per_market[has_bet]
                traded_costs = cost_per_market[has_bet]
                for market_index, market_profit, market_cost in zip(traded_indices, traded_profits, traded_costs):
                    result_data.append({
                        'forecaster': forecaster,
                        'event_ticker': row['event_ticker'],
                        'round': row['round'],
                        'market_index': int(market_index),
                        'weight': row['weight'],
                        'average_return': market_profit,
                        'cost': market_cost,
                        'num_markets_bet': 1,
                    })
            else:
                result_data.append({
                    'forecaster': forecaster,
                    'event_ticker': row['event_ticker'],
                    'round': row['round'],
                    'weight': row['weight'],
                    'average_return': np.sum(profit_per_market),
                    'cost': np.sum(cost_per_market),
                    'num_markets_bet': num_markets_bet,
                })

        except Exception:
            if per_market:
                continue
            result_data.append({
                'forecaster': forecaster,
                'event_ticker': row['event_ticker'],
                'round': row['round'],
                'weight': row['weight'],
                'average_return': np.nan,
                'cost': np.nan,
            })

    result_df = pd.DataFrame(result_data)
    return result_df


def compute_calibration_ece(forecasts: pd.DataFrame, num_bins: int = 10, 
                           strategy: Literal["uniform", "quantile"] = "uniform",
                           weight_event: bool = True, return_details: bool = False) -> pd.DataFrame:
    """
    Calculate the Expected Calibration Error (ECE) for each forecaster.
    
    The ECE measures how well-calibrated a forecaster's probability predictions are.
    For perfectly calibrated predictions, when a forecaster predicts probability p,
    the actual outcome should occur with frequency p.
    
    This function combines two types of weights:
    1. Prediction-level weight: from the 'weight' column (assigned by weight_fn in data loading)
    2. Market-level weight: either uniform (1.0) or inverse of number of markets per prediction
    
    The final weight for each market probability is: prediction_weight * market_weight
    
    Args:
        forecasts: DataFrame with columns (forecaster, event_ticker, round, prediction, outcome, weight)
        num_bins: Number of bins to use for discretization (default: 10)
        strategy: Strategy for discretization, either "uniform" or "quantile" (default: "uniform")
        weight_event: If True, weight each market by 1/num_markets within each prediction.
                     If False, all markets are weighted equally (default: True)
        return_details: If True, return the details of the ECE calculation for each forecaster. Useful for plotting.
    Returns:
        DataFrame with columns (forecaster, ece_score) containing the ECE for each forecaster
    """
    # Prepare data structures to collect probabilities and outcomes for each forecaster
    forecaster_data = {}
    
    for _, row in forecasts.iterrows():
        forecaster = row['forecaster']
        prediction_weight = row['weight']
        prediction_probs = row['prediction']  # numpy array of probabilities
        outcome_labels = row['outcome']  # numpy array of 0/1 outcomes
        
        # Initialize forecaster data if not already present
        if forecaster not in forecaster_data:
            forecaster_data[forecaster] = {
                'probs': [],
                'labels': [],
                'weights': []
            }
        
        # Calculate market-level weight
        num_markets = len(prediction_probs)
        if weight_event:
            # Each market within this prediction gets weight = prediction_weight / num_markets
            market_weight = prediction_weight / num_markets
        else:
            # Each market gets the full prediction weight
            market_weight = prediction_weight
        
        # Add each market's probability and outcome with combined weight
        forecaster_data[forecaster]['probs'].extend(prediction_probs)
        forecaster_data[forecaster]['labels'].extend(outcome_labels)
        forecaster_data[forecaster]['weights'].extend([market_weight] * num_markets)
    
    # Calculate ECE for each forecaster
    ece_results = []
    ece_details = {}
    
    for forecaster, data in forecaster_data.items():
        probs = data['probs']
        labels = data['labels']
        weights = np.array(data['weights'])
        
        # Normalize weights to sum to the number of samples (required by _bin_stats)
        if weights.sum() == 0:
            print(f"Warning: {forecaster} has no weights. Skipping ECE calculation...")
            continue
        
        weights = weights * len(probs) / weights.sum()
        
        # Calculate bin statistics using the helper function from the old API
        bin_centers, bin_widths, conf, acc, counts = _bin_stats(
            probs, labels, weights.tolist(), num_bins, strategy
        )
        
        # Calculate ECE using the helper function from the old API
        ece_score = _calculate_ece(conf, acc, counts, len(probs))
        
        ece_results.append({
            'forecaster': forecaster,
            'ece_score': ece_score
        })

        if return_details:
            ece_details[forecaster] = {
                'ece_score': ece_score,
                'bin_centers': bin_centers,
                'bin_widths': bin_widths,
                'conf': conf,
                'acc': acc,
                'counts': counts
            }
    
    # Create result DataFrame
    result_df = pd.DataFrame(ece_results)
    
    # Sort by ECE score (lower is better)
    result_df = result_df.sort_values('ece_score').reset_index(drop=True)
    
    if return_details:
        return result_df, ece_details
    else:
        return result_df


def compute_sharpe_ratio(average_return_results: pd.DataFrame, baseline_return: float = 0.0, 
                         normalize_by_round: bool = False) -> pd.DataFrame:
    """
    Calculate the Sharpe ratio for each forecaster using per-dollar returns (ROI).
    
    Args:
        average_return_results: DataFrame with columns
            (forecaster, event_ticker, round, weight, average_return, cost)
        baseline_return: The baseline ROI to subtract from realized ROI
            (default: 0.0 for break-even)
        normalize_by_round: If True, first average returns within each (forecaster, event_ticker) group,
                           then calculate Sharpe ratio across events. (default: False)
    
    Returns:
        DataFrame with columns (forecaster, sharpe_ratio, mean_roi, std_roi, num_events)
        sorted by sharpe_ratio in descending order
    """
    df = average_return_results.copy()

    if 'cost' not in df.columns:
        raise ValueError("Sharpe ratio requires a 'cost' column in average_return_results")

    df = df[df['cost'] > 0].copy()
    if df.empty:
        return pd.DataFrame(columns=['forecaster', 'sharpe_ratio', 'mean_roi', 'std_roi', 'num_events'])

    df['roi'] = df['average_return'] / df['cost']

    if normalize_by_round:
        def weighted_mean(group):
            return np.average(group['roi'], weights=group['weight'])

        event_rois = df.groupby(['forecaster', 'event_ticker']).apply(weighted_mean).reset_index(name='event_roi')

        sharpe_results = []
        for forecaster in event_rois['forecaster'].unique():
            forecaster_data = event_rois[event_rois['forecaster'] == forecaster]
            rois = forecaster_data['event_roi'].values

            excess = rois - baseline_return
            mean_excess = np.mean(excess)
            std_excess = np.std(excess, ddof=1) if len(excess) > 1 else 0.0

            if std_excess > 0:
                sharpe = mean_excess / std_excess
            else:
                sharpe = 0.0 if mean_excess == 0 else (np.inf if mean_excess > 0 else -np.inf)

            sharpe_results.append({
                'forecaster': forecaster,
                'sharpe_ratio': sharpe,
                'mean_roi': mean_excess,
                'std_roi': std_excess,
                'num_events': len(rois)
            })
    else:
        sharpe_results = []
        for forecaster in df['forecaster'].unique():
            forecaster_data = df[df['forecaster'] == forecaster]
            rois = forecaster_data['roi'].values

            excess = rois - baseline_return
            mean_excess = np.mean(excess)
            std_excess = np.std(excess, ddof=1) if len(excess) > 1 else 0.0

            if std_excess > 0:
                sharpe = mean_excess / std_excess
            else:
                sharpe = 0.0 if mean_excess == 0 else (np.inf if mean_excess > 0 else -np.inf)

            sharpe_results.append({
                'forecaster': forecaster,
                'sharpe_ratio': sharpe,
                'mean_roi': mean_excess,
                'std_roi': std_excess,
                'num_events': len(rois)
            })
    
    result_df = pd.DataFrame(sharpe_results)
    
    # Sort by Sharpe ratio (descending - higher is better)
    result_df = result_df.sort_values('sharpe_ratio', ascending=False).reset_index(drop=True)
    
    return result_df
        

"""
Helper functions to implement the generic category/streaming functionalities.
"""
def _iterate_over_categories(forecasts: pd.DataFrame) -> dict:
    categories = forecasts['category'].unique()
    for category in categories:
        category_forecasts = forecasts[forecasts['category'] == category]
        yield category, category_forecasts
    yield "overall", forecasts


def _stream_over_time(forecasts: pd.DataFrame, stream_every: int) -> dict:
    """
    Stream the forecasts each `stream_every` days, counting from the beginning.
    Yields cumulative forecasts up to each time window.
    
    Args:
        forecasts: DataFrame with 'close_time' column (string format: "2025-09-17T17:55:00+00:00")
        stream_every: Number of days between each stream window
    
    Yields:
        Tuple of (time_label, forecasts_up_to_time) for each window
    """
    # Create a copy and convert close_time to datetime
    forecasts = forecasts.copy()
    forecasts['close_time_dt'] = pd.to_datetime(forecasts['close_time'], format='ISO8601')
    
    # Find the earliest and latest close_time
    close_time_beg = forecasts['close_time_dt'].min()
    close_time_end = forecasts['close_time_dt'].max()
    
    # Calculate the number of days between start and end
    total_days = (close_time_end - close_time_beg).days
    
    # Stream forecasts every stream_every days
    current_days = 0
    while current_days <= total_days:
        # Calculate the cutoff time
        cutoff_time = close_time_beg + pd.Timedelta(days=current_days)
        
        # Get all forecasts up to this cutoff time (cumulative)
        stream_forecasts = forecasts[forecasts['close_time_dt'] <= cutoff_time]
        
        # Remove the helper column before yielding
        stream_forecasts = stream_forecasts.drop(columns=['close_time_dt'])
        
        # Create a label for this window
        time_label = str(cutoff_time.date())
        
        yield time_label, stream_forecasts
        
        current_days += stream_every
    
    forecasts.drop(columns=['close_time_dt'], inplace=True)


def compute_ranked_brier_score(forecasts: pd.DataFrame, by_category: bool = False, stream_every: int = -1, \
    normalize_by_round: bool = False, bootstrap_config: Optional[Dict] = None,
    resample_level: Literal["market", "event"] = "market",
    add_individualized_baselines: bool = False,
    max_spread: float = DEFAULT_MAX_SPREAD,
    analytical_ci: bool = False,
    analytical_ci_level: float = 0.95) -> dict:
    """
    Compute the ranked forecasters for the given score function.

    Args:
        forecasts: DataFrame with forecast data
        by_category: If True, compute rankings per category
        stream_every: If > 0, compute rankings at time intervals
        normalize_by_round: If True, downweight by number of rounds per (forecaster, event_ticker)
        bootstrap_config: Optional config for bootstrap CI estimation
        resample_level: Granularity for bootstrap resampling. "market" resamples individual markets
            (flattened across events), "event" resamples event-level aggregated scores. Default "market".
        add_individualized_baselines: If True, create "{forecaster}-market-baseline" entries for each
            forecaster by filtering market-baseline scores to their participated (event_ticker, round).
            Requires 'market-baseline' forecaster to be present.
        max_spread: Liquidity filter passed through to compute_brier_score. Events whose markets have
            yes_ask + no_ask exceeding this value are skipped. Defaults to DEFAULT_MAX_SPREAD (1.03).
        analytical_ci: If True (and bootstrap_config is None), compute a closed-form
            normal-approximation CI from per-event Brier variance instead of bootstrap.
            Mutually exclusive with bootstrap_config.
        analytical_ci_level: Confidence level for the analytical CI (default 0.95).
    """
    use_market_bootstrap = (resample_level == "market") and (bootstrap_config is not None)

    def _rank(fc):
        score = compute_brier_score(fc, per_market=False, max_spread=max_spread)
        bs_scores = compute_brier_score(fc, per_market=True, max_spread=max_spread) if use_market_bootstrap else None
        return rank_forecasters_by_score(score, normalize_by_round=normalize_by_round,
                                         bootstrap_config=bootstrap_config,
                                         add_individualized_baselines=add_individualized_baselines,
                                         bootstrap_scores_df=bs_scores,
                                         analytical_ci=analytical_ci,
                                         analytical_ci_level=analytical_ci_level)

    do_stream = stream_every > 0
    if not do_stream and not by_category:
        return _rank(forecasts)

    if by_category:
        results = {}
        for category, category_forecasts in _iterate_over_categories(forecasts):
            if do_stream:
                results[category] = {}
                for time_label, time_forecasts in tqdm(_stream_over_time(category_forecasts, stream_every=stream_every), desc=f"Calculating Brier score for category {category}"):
                    results[category][time_label] = _rank(time_forecasts)
            else:
                results[category] = _rank(category_forecasts)
        return results
    else:  # do_stream might be True (otherwise handled by above)
        results = {}
        for time_label, time_forecasts in tqdm(_stream_over_time(forecasts, stream_every=stream_every), desc="Calculating overall"):
            results[time_label] = _rank(time_forecasts)
        return results


def compute_ranked_average_return(forecasts: pd.DataFrame, by_category: bool = False, stream_every: int = -1, \
    spread_market_even: bool = False, num_money_per_round: float = 1.0, normalize_by_round: bool = False,
    bootstrap_config: Optional[Dict] = None,
    resample_level: Literal["market", "event"] = "market",
    add_individualized_baselines: bool = False,
    max_spread: float = DEFAULT_MAX_SPREAD) -> dict:
    """
    Compute the ranked forecasters for the given score function.

    Args:
        forecasts: DataFrame with forecast data
        by_category: If True, compute rankings per category
        stream_every: If > 0, compute rankings at time intervals
        spread_market_even: If True, spread budget evenly across markets
        num_money_per_round: Amount to bet per round
        normalize_by_round: If True, downweight by number of rounds per (forecaster, event_ticker)
        bootstrap_config: Optional config for bootstrap CI estimation
        resample_level: Granularity for bootstrap resampling. "market" resamples individual markets
            (flattened across events), "event" resamples event-level aggregated scores. Default "market".
        add_individualized_baselines: If True, create "{forecaster}-market-baseline" entries for each
            forecaster by filtering market-baseline scores to their participated (event_ticker, round).
            Requires 'market-baseline' forecaster to be present.
        max_spread: Liquidity filter passed through to compute_average_return_neutral. Events whose
            markets have yes_ask + no_ask exceeding this value are skipped. Defaults to
            DEFAULT_MAX_SPREAD (1.03).
    """
    use_market_bootstrap = (resample_level == "market") and (bootstrap_config is not None)

    def _rank(fc):
        score = compute_average_return_neutral(fc, spread_market_even=spread_market_even,
                                                num_money_per_round=num_money_per_round,
                                                max_spread=max_spread,
                                                per_market=False)
        bs_scores = compute_average_return_neutral(fc, spread_market_even=spread_market_even,
                                                    num_money_per_round=num_money_per_round,
                                                    max_spread=max_spread,
                                                    per_market=True) if use_market_bootstrap else None
        return rank_forecasters_by_score(score, normalize_by_round=normalize_by_round,
                                         bootstrap_config=bootstrap_config,
                                         add_individualized_baselines=add_individualized_baselines,
                                         bootstrap_scores_df=bs_scores,
                                         aggregation='roi')

    do_stream = stream_every > 0
    if not do_stream and not by_category:
        return _rank(forecasts)

    if by_category:
        results = {}
        for category, category_forecasts in _iterate_over_categories(forecasts):
            if do_stream:
                results[category] = {}
                for time_label, time_forecasts in tqdm(_stream_over_time(category_forecasts, stream_every=stream_every), desc=f"Calculating average return for category {category}"):
                    results[category][time_label] = _rank(time_forecasts)
            else:
                results[category] = _rank(category_forecasts)
        return results
    else:  # do_stream might be True (otherwise handled by above)
        results = {}
        for time_label, time_forecasts in tqdm(_stream_over_time(forecasts, stream_every=stream_every), desc="Calculating overall"):
            results[time_label] = _rank(time_forecasts)
        return results
