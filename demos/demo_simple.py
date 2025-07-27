from pm_rank.model.scoring_rule import BrierScoringRule
from pm_rank.model.average_return import AverageReturn
from pm_rank.data import GJOChallengeLoader, ProphetArenaChallengeLoader


def demo_scoring_rule_and_gjo_challenge():

    metadata_file = "src/pm_rank/data/raw/sports_challenge_metadata.json"
    predictions_file = "src/pm_rank/data/raw/all_predictions.json"
    gjo_loader = GJOChallengeLoader(
        metadata_file=metadata_file, predictions_file=predictions_file)

    gjo_challenge = gjo_loader.load_challenge(
        forecaster_filter=20, problem_filter=20)

    # can replace this with log scoring rule or spherical scoring rule, see `scoring_rule.py`.
    brier_scoring_rule = BrierScoringRule()

    fitted_scores, rankings = brier_scoring_rule.fit(
        gjo_challenge.forecast_problems, include_scores=True)

    for forecaster, score in fitted_scores.items():  # type: ignore
        # type: ignore
        print(f"  {forecaster}: score={score}, rank={rankings[forecaster]}")


def demo_average_return_and_prophet_arena_challenge():
    arena_file = "src/pm_rank/data/raw/prophet_arena_full.csv"
    prophet_arena_loader = ProphetArenaChallengeLoader(
        predictions_file=arena_file)

    prophet_arena_challenge = prophet_arena_loader.load_challenge()

    average_return = AverageReturn(num_money_per_round=1, risk_aversion=0.0)
    results = average_return.fit_stream(
        prophet_arena_challenge.stream_problems(increment=50))

    for batch_id, (scores, rankings) in results.items():
        print(f"Batch {batch_id}:")
        for forecaster, score in scores.items():
            print(f"  {forecaster}: {score}")
        print(f"  Rankings: {rankings}")
        print()


if __name__ == "__main__":
    demo_scoring_rule_and_gjo_challenge()
    demo_average_return_and_prophet_arena_challenge()
