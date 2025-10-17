import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import timedelta
from pm_rank.nightly.utils import calculate_time_to_last_submission, assign_time_bins

# Set theme and font styling (following plot_recall_rate_histogram.py)
sns.set_theme(style="ticks")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]

# Model name mapping for cleaner display
model_rename_dict = {
    "market-baseline": "Market Baseline",
    "gpt-5": "GPT-5$^\\mathrm{R}$",
    "x-ai/grok-4": "Grok 4",
    "anthropic/claude-sonnet-4-thinking": "Claude Sonnet 4$^\\mathrm{R}$",
    "google/gemini-2.5-flash-reasoning": "Gemini 2.5 Flash$^\\mathrm{R}$",
    "meta-llama/llama-4-maverick": "Llama 4 Scout"
}

# Define time bins in hours (reversed order for plotting - furthest from last submission on left)
TIME_BINS = [
    (96, float('inf'), ">4d"), # More than 4 days
    (48, 96, "2-4d"),         # 2-4 days
    (24, 48, "1-2d"),         # 1-2 days
    (12, 24, "12-24h"),         # 12-24 hours
    (6, 12, "6-12h"),         # 6-12 hours
    (3, 6, "3-6h"),         # 3-6 hours
    (0, 1, "0-3h")            # 0-3 hours
]


def plot_score_over_time_bins(result_df: pd.DataFrame, filename: str, score_col: str, 
                               forecasters: list[str] = None, time_bins: list = None):
    """
    Visualize each forecaster's average performance over time bins (time before market close).
    
    Args:
        result_df: DataFrame with columns ['forecaster', 'event_ticker', 'time_bin', 'weight', score_col]
        filename: filename to save the plot
        score_col: column to plot ('brier_score' or 'average_return')
        forecasters: list of forecasters to plot. If None, plot all forecasters.
        time_bins: List of tuples (lower, upper, label) defining time bins. If None, uses default TIME_BINS
    """
    if time_bins is None:
        time_bins = TIME_BINS
    
    # Ensure result_df has time_bin column
    if 'time_bin' not in result_df.columns:
        raise ValueError("result_df must have a 'time_bin' column. Use assign_time_bins() first.")
    
    # Get bin labels in order
    bin_labels = [label for _, _, label in time_bins]
    
    # Group by forecaster and time_bin, then compute mean score for each group
    aggregated_df = result_df.groupby(['forecaster', 'time_bin'])[score_col].mean().reset_index()
    
    # Get unique forecasters
    if forecasters is None:
        forecasters = aggregated_df['forecaster'].unique()
    else:
        forecasters = [forecaster for forecaster in forecasters if forecaster in aggregated_df['forecaster'].unique()]
    
    # Create the plot with paper-ready styling
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Color scheme for different models (distinct colors for visibility)
    model_colors = {
        "market-baseline": '#1f77b4',    # Blue
        "gpt-5": '#ff7f0e',              # Orange  
        "x-ai/grok-4": '#2ca02c',        # Green
        "anthropic/claude-sonnet-4-thinking": '#d62728',  # Red
        "google/gemini-2.5-flash-reasoning": '#9467bd',   # Purple
        "meta-llama/llama-4-maverick": '#8c564b'          # Brown
    }
    
    # Plot a curve for each forecaster
    for forecaster in forecasters:
        forecaster_data = aggregated_df[aggregated_df['forecaster'] == forecaster]
        
        # Ensure we have data for all bins (fill missing bins with NaN)
        forecaster_bins = []
        forecaster_scores = []
        
        for bin_label in bin_labels:
            bin_data = forecaster_data[forecaster_data['time_bin'] == bin_label]
            forecaster_bins.append(bin_label)
            if len(bin_data) > 0:
                forecaster_scores.append(bin_data[score_col].mean())
            else:
                forecaster_scores.append(np.nan)
        
        # Get display name and color for this forecaster
        display_name = model_rename_dict.get(forecaster, forecaster)
        color = model_colors.get(forecaster, '#000000')  # Default to black if not found
        
        # Plot the curve for this forecaster with paper-ready styling
        ax.plot(forecaster_bins, forecaster_scores, marker='o', linewidth=3, 
                markersize=8, label=display_name, color=color, alpha=0.9,
                markeredgecolor='white', markeredgewidth=1)
    
    # Customize the plot with paper-ready styling
    ax.set_xlabel('Prediction Time Before Market Resolution', fontsize=32, fontweight='normal')
    
    # Set appropriate y-label based on score type
    if score_col == "average_return":
        ax.set_ylabel('Average Return', fontsize=32, fontweight='normal')
        title = 'Average Return Performance Over Prediction Time'
    elif score_col == "brier_score":
        ax.set_ylabel('Brier Score', fontsize=32, fontweight='normal')
        title = 'Brier Score Performance Over Prediction Time'
    else:
        ax.set_ylabel(score_col.replace('_', ' ').title(), fontsize=32, fontweight='normal')
        title = f'{score_col.replace("_", " ").title()} Performance Over Time to Last Submission'
    
    ax.set_title(title, fontsize=35, fontweight='bold', pad=25)
    
    # Set x-axis to show all bin labels
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, fontsize=23, rotation=0)
    
    # Set y-axis properties
    ax.tick_params(axis='y', labelsize=23)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Position legend inside the plot at an optimal location
    legend_loc = 'lower left' if score_col == "brier_score" else 'upper left'
    
    ax.legend(loc=legend_loc, fontsize=20, framealpha=0.9, 
              fancybox=True, shadow=True, ncol=1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot with high quality
    plt.savefig(f"{filename}.pdf", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to {filename}.pdf")


def plot_time_gap_distribution(forecasts_df: pd.DataFrame, filename: str, 
                                forecasters: list[str] = None, time_bins: list = None):
    """
    Plot the empirical distribution of time gaps (time before market close) for each forecaster.
    
    Args:
        forecasts_df: DataFrame with columns ['forecaster', 'time_to_last', 'time_bin']
        filename: filename to save the plot
        forecasters: list of forecasters to plot. If None, plot all forecasters.
        time_bins: List of tuples (lower, upper, label) defining time bins. If None, uses default TIME_BINS
    """
    if time_bins is None:
        time_bins = TIME_BINS
    
    bin_labels = [label for _, _, label in time_bins]
    
    # Get unique forecasters
    if forecasters is None:
        forecasters = forecasts_df['forecaster'].unique()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Color scheme for different models
    model_colors = {
        "market-baseline": '#1f77b4',
        "gpt-5": '#ff7f0e',
        "x-ai/grok-4": '#2ca02c',
        "anthropic/claude-sonnet-4-thinking": '#d62728',
        "google/gemini-2.5-flash-reasoning": '#9467bd',
        "meta-llama/llama-4-maverick": '#8c564b'
    }
    
    # Count predictions in each bin for each forecaster
    for forecaster in forecasters:
        forecaster_data = forecasts_df[forecasts_df['forecaster'] == forecaster]
        
        # Count predictions in each bin
        bin_counts = []
        for bin_label in bin_labels:
            count = len(forecaster_data[forecaster_data['time_bin'] == bin_label])
            bin_counts.append(count)
        
        # Get display name and color
        display_name = model_rename_dict.get(forecaster, forecaster)
        color = model_colors.get(forecaster, '#000000')
        
        # Plot bar chart
        x_positions = np.arange(len(bin_labels))
        width = 0.12  # Width of bars
        forecaster_idx = list(forecasters).index(forecaster)
        offset = (forecaster_idx - len(forecasters) / 2) * width
        
        ax.bar(x_positions + offset, bin_counts, width, label=display_name, 
               color=color, alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Time to Last Submission', fontsize=32, fontweight='normal')
    ax.set_ylabel('Number of Predictions', fontsize=32, fontweight='normal')
    ax.set_title('Distribution of Time Gaps to Last Submission', fontsize=35, fontweight='bold', pad=25)
    
    # Set x-axis
    ax.set_xticks(np.arange(len(bin_labels)))
    ax.set_xticklabels(bin_labels, fontsize=23, rotation=0)
    
    # Set y-axis properties
    ax.tick_params(axis='y', labelsize=23)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=20, framealpha=0.9, 
              fancybox=True, shadow=True, ncol=1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{filename}.pdf", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved distribution plot to {filename}.pdf")


def plot_time_gap_histogram(forecasts_df: pd.DataFrame, filename: str, 
                             forecaster: str = None, max_hours: float = 200):
    """
    Plot a histogram of time gaps for a specific forecaster (for debugging/exploration).
    
    Args:
        forecasts_df: DataFrame with 'time_to_last' column
        filename: filename to save the plot
        forecaster: specific forecaster to plot. If None, plots all forecasters combined.
        max_hours: maximum hours to show on x-axis (default: 200)
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    if forecaster is not None:
        data = forecasts_df[forecasts_df['forecaster'] == forecaster]['time_to_last']
        title = f'Time Gap Distribution for {model_rename_dict.get(forecaster, forecaster)}'
    else:
        data = forecasts_df['time_to_last']
        title = 'Time Gap Distribution (All Forecasters)'
    
    # Filter to max_hours for better visualization
    data_filtered = data[data <= max_hours]
    
    # Create histogram
    ax.hist(data_filtered, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add vertical lines for bin boundaries
    for lower, upper, label in TIME_BINS[:-1]:  # Exclude the last bin (>1 week)
        if lower <= max_hours:
            ax.axvline(x=lower, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Customize plot
    ax.set_xlabel('Time to Last Submission (hours)', fontsize=24, fontweight='normal')
    ax.set_ylabel('Number of Predictions', fontsize=24, fontweight='normal')
    ax.set_title(title, fontsize=28, fontweight='bold', pad=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add statistics text
    stats_text = f'Total: {len(data)}\nMean: {data.mean():.1f}h\nMedian: {data.median():.1f}h'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=18, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f"{filename}.pdf", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved histogram to {filename}.pdf")


if __name__ == "__main__":
    predictions_csv = "slurm/predictions_10_01_to_09_01.csv"
    submissions_csv = "slurm/submissions_10_01_to_09_01.csv"
    
    from pm_rank.nightly.data import uniform_weighting, NightlyForecasts
    from pm_rank.nightly.algo import compute_brier_score, compute_average_return_neutral, add_market_baseline_predictions
    
    # Load forecasts
    forecasts = NightlyForecasts.from_prophet_arena_csv(predictions_csv, submissions_csv, uniform_weighting())
    
    # Calculate time to last submission
    print("Calculating time to last submission...")
    forecasts.data = calculate_time_to_last_submission(forecasts.data)
    
    # Assign time bins
    print("Assigning time bins...")
    forecasts.data = assign_time_bins(forecasts.data, TIME_BINS)
    
    # Print some statistics
    print("\nTime gap statistics:")
    print(forecasts.data.groupby('forecaster')['time_to_last'].describe())
    
    print("\nTime bin distribution:")
    print(forecasts.data.groupby(['forecaster', 'time_bin']).size().unstack(fill_value=0))
    
    # Add market baseline
    df = add_market_baseline_predictions(forecasts.data)
    
    # Calculate scores
    print("\nCalculating scores...")
    brier_score = compute_brier_score(df)
    average_return = compute_average_return_neutral(df, spread_market_even=True, num_money_per_round=1.0)
    
    # Merge time_bin information back to score dataframes
    time_bin_info = df[['forecaster', 'event_ticker', 'round', 'time_bin', 'time_to_last']].drop_duplicates()
    brier_score = brier_score.merge(time_bin_info, on=['forecaster', 'event_ticker', 'round'])
    average_return = average_return.merge(time_bin_info, on=['forecaster', 'event_ticker', 'round'])
    
    # Define forecasters to plot
    forecasters = ['market-baseline', 'gpt-5', 'x-ai/grok-4', 'anthropic/claude-sonnet-4-thinking', 
                   'google/gemini-2.5-flash-reasoning', 'meta-llama/llama-4-maverick']
    
    # Plot performance over time bins
    print("\nPlotting performance over time bins...")
    plot_score_over_time_bins(brier_score, "time_bins_brier_score", "brier_score", forecasters=forecasters)
    plot_score_over_time_bins(average_return, "time_bins_average_return", "average_return", forecasters=forecasters)
    
    # Plot time gap distributions
    print("\nPlotting time gap distributions...")
    plot_time_gap_distribution(df, "time_gap_distribution", forecasters=forecasters)
    
    # Plot histogram for a specific forecaster
    print("\nPlotting time gap histogram for gpt-5...")
    plot_time_gap_histogram(df, "time_gap_histogram_gpt5", forecaster="gpt-5", max_hours=200)
    
    print("\nDone!")

