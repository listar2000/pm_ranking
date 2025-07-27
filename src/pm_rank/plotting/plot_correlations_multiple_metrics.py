from pm_rank.model.bradley_terry import GeneralizedBT
from pm_rank.model.average_return import AverageReturn
from pm_rank.model.scoring_rule import BrierScoringRule
from pm_rank.model.irt import IRTModel, SVIConfig
from pm_rank.data.loaders import GJOChallengeLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
from pm_rank.model.utils import spearman_correlation, kendall_correlation


def _get_all_rankings():
    # load the data
    metadata_file = "data/raw/sports_challenge_metadata.json"
    predictions_file = "data/raw/all_predictions.json"
    gjo_loader = GJOChallengeLoader(
        metadata_file=metadata_file, predictions_file=predictions_file)

    gjo_challenge = gjo_loader.load_challenge(
        forecaster_filter=20, problem_filter=20)
    problems = gjo_challenge.get_problems()
    # the GJO data has no built-in odds, we offer a uniform odds for each problem
    for problem in problems:
        problem.odds = [1 / len(problem.options)] * \
            len(problem.options)  # type: ignore

    # get the IRT rankings
    device = "cpu"  # change to "cuda" if you have a GPU
    svi_config = SVIConfig(optimizer="Adam", num_steps=5000,
                           learning_rate=0.005, device=device)
    irt_model = IRTModel()
    irt_rankings = irt_model.fit(
        problems, method="SVI", config=svi_config, include_scores=False)

    # get the scoring rule rankings
    brier_rankings = BrierScoringRule().fit(problems, include_scores=False)

    # get the market earning rankings
    average_return_rankings = AverageReturn().fit(problems, include_scores=False)

    # get the Bradley-Terry rankings
    bt_rankings = GeneralizedBT(method="MM", num_iter=500).fit(
        problems, include_scores=False)

    # get the weighted Brier rankings
    problem_discrim_dict, _ = irt_model.get_problem_level_parameters()
    problem_discriminations = np.array(
        [problem_discrim_dict[problem.problem_id] for problem in problems])
    weighted_brier_rankings = BrierScoringRule().fit(
        problems, problem_discriminations=problem_discriminations, include_scores=False)

    return {
        "IRT": irt_rankings,
        "Brier": brier_rankings,
        "Market Earning": average_return_rankings,
        "Bradley-Terry": bt_rankings,
        "Weighted Brier": weighted_brier_rankings
    }


def plot_correlation_grid(rankings_dict, output_file="correlation_grid.png"):
    """
    Plot a grid of correlations (Spearman and Kendall) between ranking methods.
    Lower triangle: Spearman, Upper triangle: Kendall, Diagonal: method name.
    """
    methods = list(rankings_dict.keys())
    N = len(methods)

    # Set seaborn style for beautiful plots
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18
    })

    fig, ax = plt.subplots(figsize=(2.2*N, 2.2*N), dpi=300)
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    # Remove axis labels, only show method names on diagonal
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Remove grid and ticks
    ax.grid(False)
    ax.tick_params(axis=u'both', which=u'both', length=0)

    # Color maps for the two correlations
    cmap_spearman = plt.get_cmap('Blues')
    cmap_kendall = plt.get_cmap('Oranges')

    # Compute all correlations
    spearman_vals = np.zeros((N, N))
    kendall_vals = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            spearman_vals[i, j] = spearman_correlation(
                rankings_dict[methods[i]], rankings_dict[methods[j]])
            kendall_vals[i, j] = kendall_correlation(
                rankings_dict[methods[i]], rankings_dict[methods[j]])

    # Plot the grid
    for i in range(N):
        for j in range(N):
            if i == j:
                # Diagonal: method name
                ax.text(j, i, methods[i], ha='center', va='center',
                        fontsize=14, fontweight='bold', color='black')
            elif i > j:
                # Lower triangle: Spearman
                val = spearman_vals[i, j]
                color = cmap_spearman(0.5 + 0.5*val)  # val in [-1,1]
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1,
                             color=color, alpha=0.7, zorder=0))
                ax.text(j, i, f"{val:.2f}\nS", ha='center', va='center',
                        fontsize=13, color='black', fontweight='semibold')
            elif i < j:
                # Upper triangle: Kendall
                val = kendall_vals[i, j]
                color = cmap_kendall(0.5 + 0.5*val)
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1,
                             color=color, alpha=0.7, zorder=0))
                ax.text(j, i, f"{val:.2f}\nK", ha='center', va='center',
                        fontsize=13, color='black', fontweight='semibold')

    # Set limits and aspect
    ax.set_xlim(-0.5, N-0.5)
    ax.set_ylim(N-0.5, -0.5)
    ax.set_aspect('equal')

    # Title and layout
    fig.suptitle('Correlation Grid between Ranking Methods',
                 fontsize=20, fontweight='bold', y=1.05)
    plt.tight_layout()

    # Save and show
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='png')
    print(f"Plot saved as {output_file}")
    plt.show()


if __name__ == "__main__":
    rankings = _get_all_rankings()
    plot_correlation_grid(rankings)
