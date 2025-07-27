from pm_rank.data import GJOChallengeLoader
from pm_rank.model import *
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

GJO_METADATA_FILE = "src/pm_rank/data/raw/sports_challenge_metadata.json"
GJO_PREDICTIONS_FILE = "src/pm_rank/data/raw/all_predictions.json"


def test_all_models():
    gjo_loader = GJOChallengeLoader(
        metadata_file=GJO_METADATA_FILE,
        predictions_file=GJO_PREDICTIONS_FILE,
        challenge_title="Sports Challenge 2024"
    )
    challenge = gjo_loader.load_challenge(
        forecaster_filter=20, problem_filter=20)
    # we do this since GJO dataset has no built-in market odds
    challenge.fill_problem_with_fair_odds()

    # testing scoring rule (sufficient to use Brier score)
    brier_scoring_rule = BrierScoringRule()
    brier_result = brier_scoring_rule.fit(
        challenge.forecast_problems, include_scores=False)

    # testing irt model
    irt_model = IRTModel(n_bins=6, use_empirical_quantiles=False)
    svi_config = SVIConfig(optimizer="Adam", num_steps=5000,
                           learning_rate=0.005, device="cpu")
    irt_result = irt_model.fit(challenge.forecast_problems,
                               method="SVI", config=svi_config, include_scores=False)

    # testing market earning
    average_return = AverageReturn()
    average_return_result = average_return.fit(
        challenge.forecast_problems, include_scores=False)

    # testing Bradley-Terry model
    bt_model = GeneralizedBT(method="MM", num_iter=300)
    bt_result = bt_model.fit(challenge.forecast_problems, include_scores=False)

    # testing weighted brier score
    problem_discrimination_dict, _ = irt_model.get_problem_level_parameters()
    problem_discriminations = [problem_discrimination_dict[problem.problem_id]
                               for problem in challenge.forecast_problems]
    weighted_brier_result = brier_scoring_rule.fit(challenge.forecast_problems, include_scores=False,
                                                   problem_discriminations=problem_discriminations)

    all_results = [brier_result, irt_result,
                   average_return_result, weighted_brier_result, bt_result]
    all_results_names = ["brier_result", "irt_result",
                         "average_return_result", "weighted_brier_result", "bt_result"]

    # compute all pairwise correlations
    for i in range(len(all_results)):
        for j in range(i + 1, len(all_results)):
            print(
                f"Spearman correlation between {all_results_names[i]} and {all_results_names[j]}: {spearman_correlation(all_results[i], all_results[j])}")
            print(
                f"Kendall correlation between {all_results_names[i]} and {all_results_names[j]}: {kendall_correlation(all_results[i], all_results[j])}")
            print("-" * 100)


if __name__ == "__main__":
    test_all_models()
