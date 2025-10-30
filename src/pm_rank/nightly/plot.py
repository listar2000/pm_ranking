import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

def plot_score_over_round(result_df: pd.DataFrame, filename: str, score_col: str, forecasters: list[str] = None, merge_every: int = 1, max_round: int = None):
    """
    Visualize each forecaster's average return performance over rounds.
    
    For each round and each forecaster, we take the average of all rows (events) 
    that have this round and record the score. The final plot shows N curves 
    going from 1 to the max round number (N = number of forecasters).
    
    Args:
        result_df: DataFrame with columns ['forecaster', 'event_ticker', 'round', 'weight', score_col]
                  Output from compute_average_return_neutral function
        filename: filename to save the plot
        score_col: column to plot
        forecasters: list of forecasters to plot. If None, plot all forecasters.
        merge_every: merge every N rounds.
        max_round: max round to plot. If None, plot all rounds.
    """
    # Group by forecaster and round, then compute mean average_return for each group
    aggregated_df = result_df.groupby(['forecaster', 'round'])[score_col].mean().reset_index()
    
    # Get unique forecasters and rounds
    if forecasters is None:
        forecasters = aggregated_df['forecaster'].unique()
    else:
        forecasters = [forecaster for forecaster in forecasters if forecaster in aggregated_df['forecaster'].unique()]
        
    # If merge_every = 2, for instance, we will average all scores in round 1/2, round 3/4, etc. So we have len(rounds) // merge_every + 1 total rounds
    rounds = sorted(aggregated_df['round'].unique())
    round_ranges = [rounds[i:i+merge_every] for i in range(0, len(rounds), merge_every)]
    
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
        
        # Ensure we have data for all rounds (fill missing rounds with NaN)
        forecaster_rounds = []
        forecaster_returns = []
        
        for i, round_range in enumerate(round_ranges):
            round_data = forecaster_data[forecaster_data['round'].isin(round_range)]
            if len(round_data) > 0:
                forecaster_rounds.append(i + 1)
                forecaster_returns.append(round_data[score_col].mean())
            else:
                forecaster_rounds.append(i + 1)
                forecaster_returns.append(np.nan)
        
        if max_round is not None:
            forecaster_rounds = forecaster_rounds[:max_round]
            forecaster_returns = forecaster_returns[:max_round]

        # Get display name and color for this forecaster
        display_name = model_rename_dict.get(forecaster, forecaster)
        color = model_colors.get(forecaster, '#000000')  # Default to black if not found
        
        # Plot the curve for this forecaster with paper-ready styling
        ax.plot(forecaster_rounds, forecaster_returns, marker='o', linewidth=3, 
                markersize=8, label=display_name, color=color, alpha=0.9,
                markeredgecolor='white', markeredgewidth=1)
    
    # Customize the plot with paper-ready styling
    ax.set_xlabel('Round', fontsize=32, fontweight='normal')
    
    # Set appropriate y-label based on score type
    if score_col == "average_return":
        ax.set_ylabel('Average Return', fontsize=32, fontweight='normal')
        title = 'Average Return Performance Over Rounds'
    elif score_col == "brier_score":
        ax.set_ylabel('Brier Score', fontsize=32, fontweight='normal')
        title = 'Brier Score Performance Over Rounds'
    else:
        ax.set_ylabel(score_col.replace('_', ' ').title(), fontsize=32, fontweight='normal')
        title = f'{score_col.replace("_", " ").title()} Performance Over Rounds'
    
    ax.set_title(title, fontsize=35, fontweight='bold', pad=25)
    
    # Set x-axis to show all rounds
    x_ticks = range(1, len(round_ranges[:(max_round or len(round_ranges))]) + 1)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks], fontsize=23)
    
    # Set y-axis properties
    ax.tick_params(axis='y', labelsize=23)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add horizontal reference line
    if score_col == "average_return":
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Break-even')
    else:
        ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Random Predictions')
    
    # Position legend inside the plot at an optimal location
    legend_loc = 'lower left'
    
    ax.legend(loc=legend_loc, fontsize=20, framealpha=0.9, 
              fancybox=True, shadow=True, ncol=1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot with high quality
    plt.savefig(f"{filename}.pdf", dpi=300, bbox_inches='tight', facecolor='white')


def plot_market_versus_forecaster_probs(forecasts: pd.DataFrame, forecaster: str, forecaster_title: str, event_tickers: list[str] = None, outcome: str = "yes"):
    """
    We plot a scatter plot of the for the forecaster's probability versus the market-baseline one (conditioning on the event tickers and outcome) 
    """
    title_to_file_mapping = {
        "GPT-5$^\\mathrm{R}$": "gpt5-on-yes.pdf",
        "Grok 4$^\\mathrm{R}$": "grok4-on-yes.pdf",
        "Claude Sonnet 4$^\\mathrm{R}$": "claude4-on-yes.pdf",
        "Llama 4 Scout": "llama4-on-yes.pdf",
    }
    # Step 1: aggressive filtering based on forecaster and event tickers
    model_forecasts = forecasts[forecasts['forecaster'] == forecaster]
    baseline_forecasts = forecasts[forecasts['forecaster'] == "market-baseline"]

    if event_tickers is None:
        event_tickers = model_forecasts['event_ticker'].unique()
        print(f"Using the {len(event_tickers)} event tickers that the forecaster have a record on")

    model_forecasts = model_forecasts[model_forecasts['event_ticker'].isin(event_tickers)]
    baseline_forecasts = baseline_forecasts[baseline_forecasts['event_ticker'].isin(event_tickers)]

    # Step 2: go through each event ticker, make sure both model and baseline have it, and take the submission with time_rank = 1 from each
    point_x, point_y = [], []
    for i, event_ticker in enumerate(event_tickers):
        model_event = model_forecasts[model_forecasts['event_ticker'] == event_ticker]
        baseline_event = baseline_forecasts[baseline_forecasts['event_ticker'] == event_ticker]

        if i % 10 == 0:
            print("# model event: ", len(model_event))
            print("# baseline event: ", len(baseline_event))

        if len(model_event) == 0 or len(baseline_event) == 0:
            continue
        model_submission = model_event.iloc[0]
        baseline_submission = baseline_event.iloc[0]
        
        # get the indices `i`s of markets within the odds list, such that outcome[i] == 1 when outcome is "yes", and outcome[i] == 0 when outcome is "no"
        if outcome == "yes":
            market_indices = np.where(baseline_submission['outcome'] == 1)[0]
        else:
            market_indices = np.where(baseline_submission['outcome'] == 0)[0]

        # debugging:
        if i % 10 == 0:
            print("market indices: ", market_indices)
        
        model_probs = model_submission['prediction'][market_indices]
        baseline_probs = baseline_submission['prediction'][market_indices]
        
        # turn each pair of model_prob and baseline_prob into a point (x, y)
        for model_prob, baseline_prob in zip(model_probs, baseline_probs):
            # we treat 1 - prob as the actual prob if the outcome is "no"
            if outcome == "no":
                model_prob = 1 - model_prob
                baseline_prob = 1 - baseline_prob
            # rule out extreme cases
            if model_prob < 0.01 or model_prob > 0.99 or baseline_prob < 0.01 or baseline_prob > 0.99:
                continue
            point_x.append(baseline_prob)
            point_y.append(model_prob)
    
    print(f"Number of points: {len(point_x)}")

    # Step 3: plot the points
    plt.scatter(point_x, point_y, label=f'Markets resolved to {outcome.capitalize()}', alpha=0.6)
    plt.xlabel('Market-baseline Probability', fontsize=14)
    plt.xlim(0, 1)
    plt.ylabel('LLM Predicted Probability', fontsize=14)
    plt.ylim(0, 1)
    plt.title(forecaster_title, fontweight='bold', fontsize=18)
    # draw a 45 degree line
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2, label='Parity')
    # Add legend
    plt.legend(loc='upper left', fontsize=12, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(title_to_file_mapping[forecaster_title], dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved plot to {title_to_file_mapping[forecaster_title]}')


if __name__ == "__main__":
    event_tickers_df = pd.read_csv("slurm/subset_data_100.csv")
    event_tickers = event_tickers_df['event_ticker'].unique()

    predictions_csv = "slurm/predictions_10_11_to_01_01.csv"
    submissions_csv = "slurm/submissions_10_11_to_01_01.csv"
    # predictions_csv = "slurm/predictions_09_01_to_06_01.csv"
    # submissions_csv = "slurm/submissions_09_01_to_06_01.csv"

    from pm_rank.nightly.data import uniform_weighting, NightlyForecasts
    from pm_rank.nightly.algo import add_market_baseline_predictions
    
    forecasts = NightlyForecasts.from_prophet_arena_csv(predictions_csv, submissions_csv, uniform_weighting())

    data = add_market_baseline_predictions(forecasts.data)

    plot_market_versus_forecaster_probs(data, "gpt-5", "GPT-5$^\\mathrm{R}$", event_tickers, "yes")
    # clear plot
    plt.clf()
    plot_market_versus_forecaster_probs(data, "x-ai/grok-4", "Grok 4$^\\mathrm{R}$", event_tickers, "yes")
    # clear plot
    plt.clf()
    plot_market_versus_forecaster_probs(data, "anthropic/claude-sonnet-4-thinking", "Claude Sonnet 4$^\\mathrm{R}$", event_tickers, "yes")
    # clear plot
    plt.clf()
    plot_market_versus_forecaster_probs(data, "meta-llama/llama-4-scout", "Llama 4 Scout", event_tickers, "yes")
    # clear plot
    plt.clf()

    exit(0)

    # calculate the number of events and markets
    print(f"Number of events: {len(forecasts.data['event_ticker'].unique())}")
    
    # total markets = sum of length for the `odds` column for all unique event_tickers
    total_markets = forecasts.data.groupby('event_ticker')['odds'].apply(len).sum()
    print(f"Total markets: {total_markets}")

    # print a few odds samples
    print(forecasts.data['odds'].head())
    print(forecasts.data['odds'].apply(len).max())

    # print("Processing data...")
    
    # df = add_market_baseline_predictions(forecasts.data)

    # # calculate the average return and print rankings
    # brier_score = compute_brier_score(df)

    # average_return = compute_average_return_neutral(df, spread_market_even=True, num_money_per_round=1.0)

    # forecasters = ['market-baseline', 'gpt-5', 'x-ai/grok-4', 'anthropic/claude-sonnet-4-thinking', 'google/gemini-2.5-flash-reasoning', 'meta-llama/llama-4-maverick']

    # plot_score_over_round(brier_score, "09_01_to_06_01_over_round_brier_score", "brier_score", merge_every=1, max_round=12, forecasters=forecasters)
    # plot_score_over_round(average_return, "09_01_to_06_01_over_round_average_return", "average_return", merge_every=1, max_round=12, forecasters=forecasters)