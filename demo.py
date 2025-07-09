from model.scoring_rule import BrierScoringRule
from model.market_earning import MarketEarning
from data import GJOChallengeLoader, ProphetArenaChallengeLoader

def demo_scoring_rule_and_gjo_challenge():

    metadata_file = "data/raw/sports_challenge_metadata.json"
    predictions_file = "data/raw/all_predictions.json"
    gjo_loader = GJOChallengeLoader(metadata_file=metadata_file, predictions_file=predictions_file)

    gjo_challenge = gjo_loader.load_challenge(forecaster_filter=20)

    # can replace this with log scoring rule or spherical scoring rule, see `scoring_rule.py`.
    brier_scoring_rule = BrierScoringRule()

    results = brier_scoring_rule.fit_stream(gjo_challenge.stream_problems(increment=20))

    for batch_id, (scores, rankings) in results.items():
        print(f"Batch {batch_id}:")
        for forecaster, score in scores.items():
            print(f"  {forecaster}: {score}")
        print(f"  Rankings: {rankings}")
        print()


def demo_market_earning_and_prophet_arena_challenge():
    arena_file = "data/raw/prophet_arena_full.csv"
    prophet_arena_loader = ProphetArenaChallengeLoader(predictions_file=arena_file)

    prophet_arena_challenge = prophet_arena_loader.load_challenge()

    market_earning = MarketEarning(num_money_per_round=1, risk_aversion=0.0)
    results  = market_earning.fit_stream(prophet_arena_challenge.stream_problems(increment=50))

    for batch_id, (scores, rankings) in results.items():
        print(f"Batch {batch_id}:")
        for forecaster, score in scores.items():
            print(f"  {forecaster}: {score}")
        print(f"  Rankings: {rankings}")
        print()

if __name__ == "__main__":
    # demo_scoring_rule_and_gjo_challenge()
    demo_market_earning_and_prophet_arena_challenge()