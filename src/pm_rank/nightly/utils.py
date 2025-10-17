import pandas as pd
import numpy as np


def calculate_time_to_last_submission(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the time gap from each prediction to the market close time.
    
    This function calculates how many hours before the market close each prediction was made.
    It uses the 'close_time' column (actual market close time from database) if available,
    otherwise falls back to approximating with the last submission's snapshot_time.
    
    Args:
        forecasts_df: DataFrame with columns including ['event_ticker', 'snapshot_time']
                     Optionally includes 'close_time' for accurate close times
    
    Returns:
        DataFrame with an additional 'time_to_last' column (in hours) representing 
        the time gap from this prediction to the market close time
    """
    df = forecasts_df.copy()
    
    # Convert snapshot_time to datetime if it's not already
    df['snapshot_time'] = pd.to_datetime(df['snapshot_time'])
    
    # Check if close_time column exists
    if 'close_time' in df.columns:
        # Use actual close_time from database
        # Use format='ISO8601' to handle inconsistent datetime formats (some with/without milliseconds)
        df['close_time'] = pd.to_datetime(df['close_time'], format='ISO8601')
        
        # Calculate time difference in hours (close_time - snapshot_time)
        df['time_to_last'] = (df['close_time'] - df['snapshot_time']).dt.total_seconds() / 3600
    else:
        # Fall back to approximating with last submission time
        print("Warning: 'close_time' column not found. Approximating with last submission's snapshot_time.")
        
        # For each event_ticker, find the max (last) timestamp
        last_submission_times = df.groupby('event_ticker')['snapshot_time'].max().reset_index()
        last_submission_times.columns = ['event_ticker', 'last_submission_time']
        
        # Merge to get the last submission time for each row
        df = df.merge(last_submission_times, on='event_ticker')
        
        # Calculate time difference in hours
        df['time_to_last'] = (df['last_submission_time'] - df['snapshot_time']).dt.total_seconds() / 3600
        
        # Drop the temporary column
        df = df.drop(columns=['last_submission_time'])
    
    return df


def assign_time_bins(forecasts_df: pd.DataFrame, time_bins: list) -> pd.DataFrame:
    """
    Assign each forecast to a time bin based on time to market close.
    
    Args:
        forecasts_df: DataFrame with 'time_to_last' column (in hours) - time before market close
        time_bins: List of tuples (lower_bound, upper_bound, label) defining time bins.
                  Example: [(0, 6, "0-6h"), (6, 12, "6-12h"), ...]
    
    Returns:
        DataFrame with an additional 'time_bin' column containing the bin label.
        Rows that don't fit into any bin are filtered out.
    """
    df = forecasts_df.copy()
    
    def get_bin_label(hours):
        for lower, upper, label in time_bins:
            if lower <= hours < upper:
                return label
        return None  # Return None for anything outside defined bins
    
    df['time_bin'] = df['time_to_last'].apply(get_bin_label)
    
    # Filter out rows where time_bin is None
    df = df[df['time_bin'].notna()]
    
    # Create a categorical type with the correct order
    bin_labels = [label for _, _, label in time_bins if label is not None]
    df['time_bin'] = pd.Categorical(df['time_bin'], categories=bin_labels, ordered=True)
    
    return df

