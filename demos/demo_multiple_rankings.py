"""
This demo shows how to compute the Spearman and Kendall correlation between multiple rankings.
"""
import torch
from pm_rank.model.irt import IRTModel, SVIConfig
from pm_rank.model.scoring_rule import BrierScoringRule
from pm_rank.model.market_earning import MarketEarning
from pm_rank.model.utils import spearman_correlation, kendall_correlation
from pm_rank.data import GJOChallengeLoader
from typing import Dict

def demo_multiple_rankings():
    metadata_file = "src/pm_rank/data/raw/sports_challenge_metadata.json"
    predictions_file = "src/pm_rank/data/raw/all_predictions.json"
    gjo_loader = GJOChallengeLoader(metadata_file=metadata_file, predictions_file=predictions_file)

    gjo_challenge = gjo_loader.load_challenge(forecaster_filter=20, problem_filter=20)

    # can replace this with log scoring rule or spherical scoring rule, see `scoring_rule.py`.
    brier_scoring_rule = BrierScoringRule()

    brier_result = brier_scoring_rule.fit(gjo_challenge.forecast_problems, include_scores=False)
    brier_rankings: Dict[str, int] = brier_result if isinstance(brier_result, dict) else brier_result[1]

    device = "cuda" if torch.cuda.is_available() else "cpu" # cuda can be slower than cpu since we are running MCMC.
    print(f"Using device: {device}")
    irt_model = IRTModel(n_bins=6, use_empirical_quantiles=False)
    svi_config = SVIConfig(optimizer="Adam", num_steps=3000, learning_rate=0.005, device=device)
    irt_result = irt_model.fit(gjo_challenge.forecast_problems, method="SVI", config=svi_config)
    irt_rankings: Dict[str, int] = irt_result if isinstance(irt_result, dict) else irt_result[1]

    # compute the Spearman and Kendall correlation between the two rankings
    spearman_corr = spearman_correlation(brier_rankings, irt_rankings)
    kendall_corr = kendall_correlation(brier_rankings, irt_rankings)

    print(f"Spearman correlation between Brier and SVI IRT: {spearman_corr}")
    print(f"Kendall correlation between Brier and SVI IRT: {kendall_corr}")

if __name__ == "__main__":
    demo_multiple_rankings()