"""
Experimental Consistency Score: 
Give a score for each evaluated LLM in terms of the consistency of their predictions.
""" 

from typing import List, Dict, Tuple

from pm_rank.model.utils import forecaster_data_to_rankings, _format_ranking_table
from pm_rank.data import ProphetArenaChallengeLoader, ForecastProblem
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def score_logical_chain_consistency(probability_dict: Dict[str, float], chains: List[List[str]]) -> float:
    """
    Score the consistency of a set of predictions in terms of a logical chain of outcomes.
    """
    total_pairs, correct_pairs = 0, 0
    for chain in chains:
        for i in range(len(chain) - 1):
            total_pairs += 1
            assert chain[i] in probability_dict, f"Outcome {chain[i]} not in probability_dict"
            assert chain[i+1] in probability_dict, f"Outcome {chain[i+1]} not in probability_dict"
            prob_a, prob_b = probability_dict[chain[i]], probability_dict[chain[i+1]]
            # since we have "A" -> "B", the probability of "B" should be greater than or equal to the probability of "A"
            if prob_b >= prob_a:
                correct_pairs += 1

    return correct_pairs / total_pairs


def score_mutually_exclusive_consistency(probability_dict: Dict[str, float], exclusive_sets: List[List[str]], tolerance: float = 0.001) -> float:
    """
    Score the consistency of a set of predictions in terms of a mutually exclusive set of outcomes.
    """
    assert len(exclusive_sets) == 1, f"Only one mutually exclusive should be returned, for now. Get {exclusive_sets}"
    exclusive_set = exclusive_sets[0]
    assert len(set(exclusive_set)) == len(exclusive_set), f"Exclusive set {exclusive_set} has duplicate outcomes"

    exclusive_probs = [probability_dict[opt] for opt in exclusive_set if opt in probability_dict]
    return 0.0 if abs(sum(exclusive_probs) - 1.0) > tolerance else 1.0


def score_llm_consistency(problems: List[ForecastProblem], llm_dict: Dict[str, List[List[str]]], task: str) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Get a <llm_name, score> dictionary for each LLM in the problems. The score is for the logical chain consistency.
    """
    forecaster_data = {}
    for problem in problems:
        problem_id = problem.problem_id
        if problem_id not in llm_dict or len(problem.options) < 2: # skip problems with less than 2 options
            continue
        problem_chains = llm_dict[problem_id]
        for forecast in problem.forecasts:
            forecast_prob_dict = {opt: prob for opt, prob in zip(problem.options, forecast.probs)}
            if task == "logical-chain":
                score = score_logical_chain_consistency(forecast_prob_dict, problem_chains)
            elif task == "mutually-exclusive":
                score = score_mutually_exclusive_consistency(forecast_prob_dict, problem_chains)
            else:
                raise ValueError(f"Invalid task: {task}")
            if forecast.username not in forecaster_data:
                forecaster_data[forecast.username] = []
            forecaster_data[forecast.username].append(score)

    return forecaster_data_to_rankings(forecaster_data, include_scores=True, include_bootstrap_ci=False, ascending=False, aggregate="mean")


def test_logical_chain_consistency(problems: List[ForecastProblem], raw_dict):
    results = raw_dict["results"]
    llm_dict = {result["problem_id"]: result["result"]["chains"] for result in results if result["result"]["has_chain"]}

    print(f"Number of problems with logical chains: {len(llm_dict)}")

    llm_scores, llm_ranking = score_llm_consistency(problems, llm_dict, task="logical-chain")
    print(_format_ranking_table(llm_ranking, llm_scores))
    
    # Generate plot
    plot_consistency_scores(llm_scores, llm_ranking, "logical-chain")


def test_mutually_exclusive_consistency(problems: List[ForecastProblem], raw_dict):
    results = raw_dict["results"]
    llm_dict = {result["problem_id"]: result["result"]["exclusive_sets"] for result in results if result["result"]["has_exclusive_set"]}
    print(f"Number of problems with mutually exclusive sets: {len(llm_dict)}")
    llm_scores, llm_ranking = score_llm_consistency(problems, llm_dict, task="mutually-exclusive")
    print(_format_ranking_table(llm_ranking, llm_scores))
    
    # Generate plot
    plot_consistency_scores(llm_scores, llm_ranking, "mutually-exclusive")


def plot_consistency_scores(llm_scores: Dict[str, float], llm_ranking: Dict[str, int], task: str, 
                           output_dir: str = "/net/scratch2/listar2000/pm_ranking/slurm"):
    """
    Plot consistency scores for LLMs and save as publication-quality PDF.
    
    Args:
        llm_scores: Dictionary mapping LLM names to their consistency scores
        llm_ranking: Dictionary mapping LLM names to their ranking positions
        task: Task type ('logical-chain' or 'mutually-exclusive')
        output_dir: Directory to save the plot
    """
    
    def shorten_model_name(name: str) -> str:
        """Shorten model names by taking the part after '/' if it exists."""
        if '/' in name:
            return name.split('/')[-1]
        return name
    
    # Set up the plot style for publication quality
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure and axis with larger width to accommodate more models
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Sort LLMs by ranking for consistent ordering
    sorted_llms = sorted(llm_ranking.keys(), key=lambda x: llm_ranking[x])
    scores = [llm_scores[llm] for llm in sorted_llms]
    
    # Shorten model names for display
    short_names = [shorten_model_name(llm) for llm in sorted_llms]
    
    # Create bar plot
    bars = ax.bar(range(len(sorted_llms)), scores, 
                  color=sns.color_palette("husl", len(sorted_llms)),
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    ax.set_xlabel('Language Models', fontsize=14, fontweight='bold')
    ax.set_ylabel('Consistency Score', fontsize=14, fontweight='bold')
    
    # Set title based on task
    if task == "logical-chain":
        title = "Logical Chain Consistency Scores Across Language Models"
    elif task == "mutually-exclusive":
        title = "Mutually Exclusive Consistency Scores Across Language Models"
    else:
        title = f"{task.title()} Consistency Scores Across Language Models"
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis labels with shortened names
    ax.set_xticks(range(len(sorted_llms)))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=10)
    
    # Adapt y-axis to the data range
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    
    if score_range < 0.1:  # When scores are very close (e.g., all above 0.9)
        # Zoom in on the relevant range with more space for labels
        y_margin = max(0.015, score_range * 0.2)
        ax.set_ylim(min_score - y_margin, max_score)
    else:
        # Use standard range with some padding
        ax.set_ylim(0, max(1.0, max_score + 0.05))
    
    # Add value labels on top of bars with smart positioning to avoid overlap
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        
        # Calculate vertical offset to avoid overlap
        # For very close scores, stagger the labels
        if score_range < 0.1:
            # Alternate label positions for better visibility
            # if i % 2 == 0:
            #     v_offset = 0.003
            #     va = 'bottom'
            # else:
            v_offset = -0.008
            va = 'top'
        else:
            v_offset = 0.005
            va = 'bottom'
        
        ax.text(bar.get_x() + bar.get_width()/2., height + v_offset,
                f'{score:.3f}', ha='center', va=va, fontsize=9, fontweight='bold')
    
    # Add ranking information as text annotations inside bars
    # for i, llm in enumerate(sorted_llms):
    #     rank = llm_ranking[llm]
    #     # Position ranking text in the middle of the bar
    #     bar_height = scores[i]
    #     text_y = bar_height * 0.5  # Middle of the bar
    #     ax.text(i, text_y, f'#{rank}', ha='center', va='center', 
    #             fontsize=8, fontweight='bold', color='white',
    #             bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
    
    # Improve grid and spines
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save as high-quality PDF
    output_path = Path(output_dir) / f"consistency_scores_{task.replace('-', '_')}.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    
    print(f"Plot saved to: {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)


# local testing
if __name__ == "__main__":

    data_file = "/net/scratch2/listar2000/pm_ranking/slurm/latest_predictions_df_08_20.csv"
    # logical_chain_file = "/net/scratch2/listar2000/pm_ranking/slurm/prophet_problems_logical_chain_output.json"
    mutually_exclusive_file = "/net/scratch2/listar2000/pm_ranking/slurm/prophet_problems_mutually_exclusive_output.json"

    challenge_loader = ProphetArenaChallengeLoader(predictions_file=data_file, use_bid_for_odds=False)
    prophet_challenge = challenge_loader.load_challenge(add_market_baseline=False)
    problems = prophet_challenge.get_problems()

    raw_dict = json.load(open(mutually_exclusive_file))

    test_mutually_exclusive_consistency(problems, raw_dict)
    # test_logical_chain_consistency(problems, raw_dict)
    