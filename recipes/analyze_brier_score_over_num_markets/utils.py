"""
Utility functions for analyzing the "sparsity" level of market outcomes.
"""
import pandas as pd
import numpy as np
import json_repair
from typing import Any


MARKET_NUM_KEY = "num_markets_per_event"
MARKET_OUTCOME_KEY = "market_outcome"
FORECASTER_KEY = "predictor_name"
PREDICTION_KEY = "prediction"
SELECTED_FORECASTERS = ["anthropic/claude-opus-4.1", "x-ai/grok-4", "o3", "gemini-2.0-flash"]

# Augmented columns
DEFAULT_DILUTION_KEY = "dilution"
DEFAULT_NUM_MARKETS_KEY = "num_markets"
DEFAULT_BRIER_SCORE_KEY = "brier_score"

# Columns left after cleaning
CLEAN_PREDICTIONS_DF_KEYS = ["event_ticker", "submission_id", "category", FORECASTER_KEY]

"""
Functions related to loading the dataframe and adding derived columns
"""
def derive_num_markets_and_dilution(row: pd.Series) -> tuple[float, float]:
    """Derive the number of markets and dilution for a given row. Dilution = number of 1 / total number of markets for market outcome"""
    try:
        market_outcome: dict = json_repair.loads(row[MARKET_OUTCOME_KEY])  # type: ignore
        num_markets = len(market_outcome.keys())
        num_true_markets = sum([1 if int(v) == 1 else 0 for v in market_outcome.values()])
        dilution = num_true_markets / num_markets
        return num_markets, dilution
    except:
        return float('nan'), float('nan')


def derive_brier_score(row: pd.Series) -> float:
    """Derive the Brier score for a given row. Brier score = mean((prediction - outcome)^2)"""
    try:
        prediction_raw: list[dict] = json_repair.loads(row[PREDICTION_KEY])["probabilities"]  # type: ignore
        # turn the prediction list of dict into a dict (similar to outcome)
        prediction = {item["market"]: item["probability"] for item in prediction_raw}
        outcome: dict = json_repair.loads(row[MARKET_OUTCOME_KEY])  # type: ignore

        market_keys = list(outcome.keys())
        outcome_vec = np.array([outcome[k] for k in market_keys])
        prediction_vec = np.array([prediction.get(k, 0.0) for k in market_keys]).clip(0.0, 1.0)
        brier_score = np.mean((prediction_vec - outcome_vec) ** 2)
        return float(brier_score)
    except:
        return float('nan')


def get_augmented_and_cleaned_df(
    predictions_path: str, 
    forecasters: list[str] | None = SELECTED_FORECASTERS, 
    num_markets_key: str = DEFAULT_NUM_MARKETS_KEY, 
    dilution_key: str = DEFAULT_DILUTION_KEY,
    brier_score_key: str = DEFAULT_BRIER_SCORE_KEY,
    clean_df_keys: list[str] = CLEAN_PREDICTIONS_DF_KEYS
) -> pd.DataFrame:
    """Get the cleaned dataframe by merging the submission and predictions dataframes."""
    predictions_df = pd.read_csv(predictions_path)
    loaded_len = len(predictions_df)
    # step 1: apply filtering by forecasters if needed
    if forecasters is not None:
        predictions_df = predictions_df[predictions_df[FORECASTER_KEY].isin(forecasters)]
        filtered_len = len(predictions_df)
        print(f"Filtered {loaded_len - filtered_len} rows out of {loaded_len} rows (remaining {filtered_len}) by limiting to forecasters\n{forecasters}")
    # step 2: add derived columns
    predictions_df[num_markets_key], predictions_df[dilution_key] = zip(*predictions_df.apply(derive_num_markets_and_dilution, axis=1))
    predictions_df[brier_score_key] = predictions_df.apply(derive_brier_score, axis=1)
    # step 3: remove any NaN values within the columns just added
    predictions_df = predictions_df.dropna(subset=[num_markets_key, dilution_key, brier_score_key])
    valid_len = len(predictions_df)
    print(f"Removed {valid_len - len(predictions_df)} rows out of {valid_len} rows (remaining {len(predictions_df)}) by dropping NaN values")
    # step 4: select only the columns we need
    clean_df_keys = list(set(clean_df_keys + [num_markets_key, dilution_key, brier_score_key]))
    predictions_df = predictions_df[clean_df_keys].reset_index(drop=True)
    return predictions_df


"""
Helper functions for part (i) of the analysis.
"""
def calc_num_markets_distribution(
    df: pd.DataFrame, 
    forecaster: str | None = None, 
    get_summary: bool = False,
    num_markets_key: str = DEFAULT_NUM_MARKETS_KEY
) -> pd.Series:
    """
    Calculate the distribution of the number of markets for events that a forecaster has submitted predictions for.
    If forecaster is not provided, we calculate for all forecasters.
    If get_summary is True, we return a summary of the distribution, rather than the distribution itself.
    """
    assert num_markets_key in df.columns, f"Column {num_markets_key} not found in dataframe"
    # filter the dataframe to only include predictions for the given forecaster
    if forecaster:
        df = df[df[FORECASTER_KEY] == forecaster]
    
    # calculate the distribution of the number of markets for events that a forecaster has submitted predictions for
    num_markets_distribution = df[num_markets_key].value_counts().sort_index()

    if get_summary:
        return num_markets_distribution.describe()
    else:
        return num_markets_distribution


def get_num_markets_by_forecaster(
    df: pd.DataFrame,
    forecasters: list[str] | None = None,
    num_markets_key: str = DEFAULT_NUM_MARKETS_KEY
) -> dict[str, np.ndarray]:
    """
    Get the number of markets distribution for each forecaster as a dict of arrays.
    Useful for creating comparative visualizations (e.g., box plots, violin plots).
    
    Returns:
        Dictionary mapping forecaster name to array of num_markets values.
    """
    if forecasters is None:
        forecasters = df[FORECASTER_KEY].unique().tolist()
    
    result = {}
    for forecaster in forecasters:
        forecaster_df = df[df[FORECASTER_KEY] == forecaster]
        result[forecaster] = forecaster_df[num_markets_key].values
    
    return result


"""
Helper functions for part (ii) of the analysis.
"""
def calc_avg_brier_score_over_num_markets(
    df: pd.DataFrame, 
    forecaster: str | None = None, 
    normalize_by_event: bool = False,
    brier_score_key: str = DEFAULT_BRIER_SCORE_KEY,
    num_markets_key: str = DEFAULT_NUM_MARKETS_KEY
) -> pd.DataFrame:
    """Calculate the average Brier score over the number of markets for a given forecaster.
    
    Returns a DataFrame with columns: num_markets, mean_brier, std_brier, count
    
    If normalize_by_event is False, we simply take the average over all "predictions" (i.e. rows) for a forecaster.
    Otherwise, we will first group by event_ticker and average within group, then take the average over all events.
    """
    assert brier_score_key in df.columns, f"Column {brier_score_key} not found in dataframe"
    assert num_markets_key in df.columns, f"Column {num_markets_key} not found in dataframe"
    
    if forecaster:
        df = df[df[FORECASTER_KEY] == forecaster].copy()
    else:
        df = df.copy()
    
    if normalize_by_event:
        # First average within each event, then aggregate by num_markets
        event_level = df.groupby("event_ticker").agg({
            brier_score_key: "mean",
            num_markets_key: "first"  # num_markets is the same for all rows in an event
        }).reset_index()
        grouped = event_level.groupby(num_markets_key)[brier_score_key]
    else:
        # Simple aggregation over all predictions
        grouped = df.groupby(num_markets_key)[brier_score_key]
    
    result = grouped.agg(["mean", "std", "count"]).reset_index()
    result.columns = [num_markets_key, "mean_brier", "std_brier", "count"]
    # Calculate standard error for error bars
    result["sem_brier"] = result["std_brier"] / np.sqrt(result["count"])
    return result


def calc_avg_brier_score_by_forecaster_over_num_markets(
    df: pd.DataFrame,
    forecasters: list[str] | None = None,
    normalize_by_event: bool = False,
    brier_score_key: str = DEFAULT_BRIER_SCORE_KEY,
    num_markets_key: str = DEFAULT_NUM_MARKETS_KEY
) -> dict[str, pd.DataFrame]:
    """
    Calculate average Brier score over number of markets for multiple forecasters.
    
    Returns:
        Dictionary mapping forecaster name to their Brier score statistics DataFrame.
    """
    if forecasters is None:
        forecasters = df[FORECASTER_KEY].unique().tolist()
    
    result = {}
    for forecaster in forecasters:
        result[forecaster] = calc_avg_brier_score_over_num_markets(
            df, forecaster=forecaster, normalize_by_event=normalize_by_event,
            brier_score_key=brier_score_key, num_markets_key=num_markets_key
        )
    
    return result
    

"""
Helper functions for part (iii) of the analysis.
"""
def get_dilution_num_markets_pairs(
    df: pd.DataFrame, 
    dilution_key: str = DEFAULT_DILUTION_KEY, 
    num_markets_key: str = DEFAULT_NUM_MARKETS_KEY
) -> pd.DataFrame:
    """
    Get unique event-level (dilution, num_markets) pairs for scatter plot visualization.
    Since dilution and num_markets are event-level properties (not forecaster-specific),
    we deduplicate by event_ticker.
    
    Returns:
        DataFrame with columns: event_ticker, num_markets, dilution
    """
    # Get unique events with their properties
    event_df = df.groupby("event_ticker").agg({
        num_markets_key: "first",
        dilution_key: "first"
    }).reset_index()
    
    return event_df


def calc_dilution_num_markets_correlation(
    df: pd.DataFrame,
    dilution_key: str = DEFAULT_DILUTION_KEY,
    num_markets_key: str = DEFAULT_NUM_MARKETS_KEY
) -> dict[str, float]:
    """
    Calculate correlation statistics between dilution and number of markets.
    
    Returns:
        Dictionary with pearson_r, pearson_p, spearman_r, spearman_p
    """
    from scipy import stats
    
    event_df = get_dilution_num_markets_pairs(df, dilution_key, num_markets_key)
    
    num_markets = event_df[num_markets_key].values
    dilution = event_df[dilution_key].values
    
    pearson_r, pearson_p = stats.pearsonr(num_markets, dilution)
    spearman_r, spearman_p = stats.spearmanr(num_markets, dilution)
    
    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "n_events": len(event_df)
    }


"""
Helper functions for part (iv) of the analysis.
"""
def calc_avg_brier_score_over_dilution(
    df: pd.DataFrame, 
    forecaster: str | None = None, 
    brier_score_key: str = DEFAULT_BRIER_SCORE_KEY, 
    dilution_key: str = DEFAULT_DILUTION_KEY,
    num_markets_key: str = DEFAULT_NUM_MARKETS_KEY
) -> pd.DataFrame:
    """
    For each event, average the Brier score across predictions (optionally filtered by forecaster),
    then return (dilution, brier_score, num_markets) triplets for each event.
    
    If forecaster is provided, only consider predictions from that forecaster.
    If forecaster is None, average across all forecasters' predictions for that event.
    
    Returns:
        DataFrame with columns: event_ticker, dilution, avg_brier, num_markets, n_predictions
    """
    if forecaster:
        df = df[df[FORECASTER_KEY] == forecaster].copy()
    else:
        df = df.copy()
    
    # Group by event and aggregate
    event_df = df.groupby("event_ticker").agg({
        brier_score_key: "mean",
        dilution_key: "first",  # Same for all rows in an event
        num_markets_key: "first",  # Same for all rows in an event
        FORECASTER_KEY: "count"  # Number of predictions
    }).reset_index()
    
    event_df.columns = ["event_ticker", "avg_brier", "dilution", "num_markets", "n_predictions"]
    return event_df


def calc_brier_dilution_by_forecaster(
    df: pd.DataFrame,
    forecasters: list[str] | None = None,
    brier_score_key: str = DEFAULT_BRIER_SCORE_KEY,
    dilution_key: str = DEFAULT_DILUTION_KEY,
    num_markets_key: str = DEFAULT_NUM_MARKETS_KEY
) -> dict[str, pd.DataFrame]:
    """
    Calculate event-level (dilution, brier_score) pairs for each forecaster.
    
    Returns:
        Dictionary mapping forecaster name to their event-level DataFrame.
    """
    if forecasters is None:
        forecasters = df[FORECASTER_KEY].unique().tolist()
    
    result = {}
    for forecaster in forecasters:
        result[forecaster] = calc_avg_brier_score_over_dilution(
            df, forecaster=forecaster, brier_score_key=brier_score_key,
            dilution_key=dilution_key, num_markets_key=num_markets_key
        )
    
    return result


def run_regression_brier_score_over_dilution(
    df: pd.DataFrame,
    forecaster: str | None = None,
    include_num_markets: bool = True,
    brier_score_key: str = DEFAULT_BRIER_SCORE_KEY,
    dilution_key: str = DEFAULT_DILUTION_KEY,
    num_markets_key: str = DEFAULT_NUM_MARKETS_KEY
) -> dict[str, Any]:
    """
    Run OLS regression to explain Brier score using dilution (and optionally num_markets).
    
    Model: brier_score ~ dilution + num_markets (if include_num_markets=True)
           brier_score ~ dilution (if include_num_markets=False)
    
    Args:
        df: DataFrame with prediction data
        forecaster: If provided, only use predictions from this forecaster
        include_num_markets: Whether to include num_markets as a feature
        
    Returns:
        Dictionary containing:
        - 'model': fitted statsmodels OLS result
        - 'summary': regression summary as string
        - 'coefficients': dict of coefficient names to (estimate, std_err, pvalue)
        - 'r_squared': R-squared value
        - 'data': the event-level DataFrame used for regression
    """
    import statsmodels.api as sm
    
    # Get event-level data
    event_df = calc_avg_brier_score_over_dilution(
        df, forecaster=forecaster, brier_score_key=brier_score_key,
        dilution_key=dilution_key, num_markets_key=num_markets_key
    )
    
    # Prepare features
    y = event_df["avg_brier"].values
    if include_num_markets:
        X = event_df[["dilution", "num_markets"]].values
        feature_names = ["const", "dilution", "num_markets"]
    else:
        X = event_df[["dilution"]].values
        feature_names = ["const", "dilution"]
    
    # Add constant term
    X = sm.add_constant(X)
    
    # Fit OLS
    model = sm.OLS(y, X).fit()
    
    # Extract coefficients
    coefficients = {}
    for i, name in enumerate(feature_names):
        coefficients[name] = {
            "estimate": model.params[i],
            "std_err": model.bse[i],
            "pvalue": model.pvalues[i],
            "ci_lower": model.conf_int()[i, 0],
            "ci_upper": model.conf_int()[i, 1]
        }
    
    return {
        "model": model,
        "summary": model.summary().as_text(),
        "coefficients": coefficients,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_pvalue": model.f_pvalue,
        "n_obs": int(model.nobs),
        "data": event_df
    }


def run_regression_comparison(
    df: pd.DataFrame,
    forecasters: list[str] | None = None,
    include_num_markets: bool = True,
    brier_score_key: str = DEFAULT_BRIER_SCORE_KEY,
    dilution_key: str = DEFAULT_DILUTION_KEY,
    num_markets_key: str = DEFAULT_NUM_MARKETS_KEY
) -> pd.DataFrame:
    """
    Run regression for multiple forecasters and compare results.
    
    Returns:
        DataFrame with one row per forecaster and columns for key regression statistics.
    """
    if forecasters is None:
        forecasters = df[FORECASTER_KEY].unique().tolist()
    
    results = []
    for forecaster in forecasters:
        reg_result = run_regression_brier_score_over_dilution(
            df, forecaster=forecaster, include_num_markets=include_num_markets,
            brier_score_key=brier_score_key, dilution_key=dilution_key, num_markets_key=num_markets_key
        )
        
        row = {
            "forecaster": forecaster,
            "n_events": reg_result["n_obs"],
            "r_squared": reg_result["r_squared"],
            "adj_r_squared": reg_result["adj_r_squared"],
            "dilution_coef": reg_result["coefficients"]["dilution"]["estimate"],
            "dilution_pvalue": reg_result["coefficients"]["dilution"]["pvalue"],
        }
        
        if include_num_markets:
            row["num_markets_coef"] = reg_result["coefficients"]["num_markets"]["estimate"]
            row["num_markets_pvalue"] = reg_result["coefficients"]["num_markets"]["pvalue"]
        
        results.append(row)
    
    return pd.DataFrame(results)



if __name__ == "__main__":
    # Example usage and testing
    predictions_path = "/net/scratch2/listar2000/pm_ranking/slurm/predictions_12_31_to_01_01.csv"
    df = get_augmented_and_cleaned_df(predictions_path)
    
    print("\n--- Part (i): Market distribution ---")
    print(calc_num_markets_distribution(df, get_summary=True))
    
    print("\n--- Part (ii): Brier score over num_markets ---")
    print(calc_avg_brier_score_over_num_markets(df))
    
    print("\n--- Part (iii): Dilution correlation ---")
    print(calc_dilution_num_markets_correlation(df))
    
    print("\n--- Part (iv): Regression ---")
    reg_result = run_regression_brier_score_over_dilution(df)
    print(reg_result["summary"])