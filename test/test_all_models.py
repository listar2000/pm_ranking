"""Integration test that fits every available model on the GJO sports challenge.

Requires the GJO raw data AND the ``full`` extra (for IRT via pyro-ppl). Both
are enforced via markers so this is auto-skipped by default.
"""
from pathlib import Path

import pytest

from pm_rank.data import GJOChallengeLoader
from pm_rank.model import (
    AverageReturn,
    BrierScoringRule,
    GeneralizedBT,
    kendall_correlation,
    spearman_correlation,
)

pytestmark = [pytest.mark.requires_gjo_data, pytest.mark.requires_pyro]

_RAW_DIR = Path(__file__).resolve().parent.parent / "src" / "pm_rank" / "data" / "raw"
GJO_METADATA_FILE = _RAW_DIR / "sports_challenge_metadata.json"
GJO_PREDICTIONS_FILE = _RAW_DIR / "all_predictions.json"


def test_all_models():
    from pm_rank.model import IRTModel, SVIConfig  # imported lazily to keep collection cheap

    gjo_loader = GJOChallengeLoader(
        metadata_file=str(GJO_METADATA_FILE),
        predictions_file=str(GJO_PREDICTIONS_FILE),
        challenge_title="Sports Challenge 2024",
    )
    challenge = gjo_loader.load_challenge(forecaster_filter=20, problem_filter=20)

    # fit(..., include_scores=False) returns a 1-element tuple (rankings,)
    # for Brier / IRT / AverageReturn / BT, so each call unpacks accordingly
    # before we feed the rank dicts into the correlation helpers.
    brier_scoring_rule = BrierScoringRule()
    (brier_ranks,) = brier_scoring_rule.fit(challenge.forecast_problems, include_scores=False)

    irt_model = IRTModel(n_bins=6, use_empirical_quantiles=False)
    svi_config = SVIConfig(optimizer="Adam", num_steps=5000, learning_rate=0.005, device="cpu")
    (irt_ranks,) = irt_model.fit(
        challenge.forecast_problems, method="SVI", config=svi_config, include_scores=False
    )

    average_return = AverageReturn()
    (average_return_ranks,) = average_return.fit(challenge.forecast_problems, include_scores=False)

    bt_model = GeneralizedBT(method="MM", num_iter=300)
    (bt_ranks,) = bt_model.fit(challenge.forecast_problems, include_scores=False)

    problem_discrimination_dict, _ = irt_model.get_problem_level_parameters()
    problem_discriminations = [
        problem_discrimination_dict[problem.problem_id] for problem in challenge.forecast_problems
    ]
    (weighted_brier_ranks,) = brier_scoring_rule.fit(
        challenge.forecast_problems,
        include_scores=False,
        problem_discriminations=problem_discriminations,
    )

    all_ranks = [brier_ranks, irt_ranks, average_return_ranks, weighted_brier_ranks, bt_ranks]

    # Every pairwise correlation should be a real number in [-1, 1].
    for i in range(len(all_ranks)):
        for j in range(i + 1, len(all_ranks)):
            s = spearman_correlation(all_ranks[i], all_ranks[j])
            k = kendall_correlation(all_ranks[i], all_ranks[j])
            assert -1.0 <= s <= 1.0
            assert -1.0 <= k <= 1.0
