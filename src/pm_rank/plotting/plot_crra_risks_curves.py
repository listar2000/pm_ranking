"""
We will choose a few different risk aversion levels (0, 0.5, 1) for the Prophet Arena challenge.
Then `fit_stream` to get the scores (utilities) over time (batch iterations) for each forecaster.
In total, we plot (1 x 3) plots, where in each plot for a certain risk aversion level:
- x-axis: batch iterations
- y-axis: scores (utilities)
- lines: different forecasters (use different colors for each forecaster)
- title: risk aversion level
"""
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

from pm_rank.data.loaders import ProphetArenaChallengeLoader
from pm_rank.model.average_return import AverageReturn

if __name__ == "__main__":
    # Set seaborn style for beautiful plots
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Set font sizes for better readability
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18
    })

    arena_file = "src/pm_rank/data/raw/prophet_arena_full.csv"
    prophet_arena_loader = ProphetArenaChallengeLoader(
        predictions_file=arena_file)

    prophet_arena_challenge = prophet_arena_loader.load_challenge()

    average_return_0 = AverageReturn(num_money_per_round=1, risk_aversion=0.0)
    average_return_05 = AverageReturn(num_money_per_round=1, risk_aversion=0.5)
    average_return_1 = AverageReturn(num_money_per_round=1, risk_aversion=1.0)

    results_0 = average_return_0.fit_stream(
        prophet_arena_challenge.stream_problems(increment=50, order="sequential"))
    results_05 = average_return_05.fit_stream(
        prophet_arena_challenge.stream_problems(increment=50, order="sequential"))
    results_1 = average_return_1.fit_stream(
        prophet_arena_challenge.stream_problems(increment=50, order="sequential"))

    # Create a 1 x 3 subplots with better spacing
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    fig.suptitle('Prophet Arena Challenge: Utility Scores Across Risk Aversion Levels',
                 fontsize=20, fontweight='bold', y=0.98)

    # Define a color palette for forecasters
    colors = sns.color_palette("husl", n_colors=4)

    # Plot the results
    for i, (risk_aversion, results) in enumerate(zip([0.0, 0.5, 1.0], [results_0, results_05, results_1])):
        ax = axs[i]

        # Extract cumulative scores over time for each forecaster
        forecaster_scores: Dict[str, list] = {}
        for batch_id, batch_result in results.items():
            # batch_result is a tuple of (scores, rankings) when include_scores=True
            if isinstance(batch_result, tuple):
                scores, _ = batch_result
            else:
                scores = batch_result

            for forecaster, score in scores.items():
                if forecaster not in forecaster_scores:
                    forecaster_scores[forecaster] = []
                forecaster_scores[forecaster].append(score)

        # Plot each forecaster with different colors
        for j, (forecaster, scores) in enumerate(forecaster_scores.items()):
            ax.plot(scores, label=forecaster, linewidth=2.5, alpha=0.8,
                    color=colors[j % len(colors)])

        # Customize the subplot
        ax.set_title(
            f'Risk Aversion: {risk_aversion}', fontweight='bold', pad=20)
        ax.set_xlabel('Batch Iterations', fontweight='semibold')
        ax.set_ylabel('Utility Scores', fontweight='semibold')

        # Only add legend to the last plot (rightmost)
        if i == 2:  # Last plot
            ax.legend(loc='upper right', frameon=True,
                      fancybox=True, shadow=True)

        ax.grid(True, alpha=0.3)

        # Add some styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # # Save to high DPI PDF
    # output_file = "prophet_arena_risk_curves_0717.png"
    # plt.savefig(output_file, dpi=300, bbox_inches='tight', format='png')
    # print(f"Plot saved as {output_file}")

    plt.show()
