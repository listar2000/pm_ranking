import math
from pm_rank.model import BrierScoringRule, AverageReturn
from pm_rank.data.loaders import ProphetArenaChallengeLoader

prophet_loader = ProphetArenaChallengeLoader(
    predictions_file="src/pm_rank/data/raw/new_prophet_arena_data.csv"
)

prophet_challenge = prophet_loader.load_challenge()
prophet_problems = prophet_challenge.get_problems()

# get the most oldeset 10 problems
# prophet_problems = sorted(prophet_problems, key=lambda x: x.end_date, reverse=False)

brier_scoring_rule = BrierScoringRule()

# use the AverageReturn to rank
average_return = AverageReturn()
average_return_score, average_return_ranking = average_return.fit(prophet_problems, include_scores=True)

print("--------------------------------")

# print the ranking
print("AverageReturn Ranking")
for username, score in sorted(average_return_score.items(), key=lambda x: x[1], reverse=True):
    print(f"User {username}: score {score}, rank {average_return_ranking[username]}")

brier_score, brier_ranking = brier_scoring_rule.fit(prophet_problems, include_scores=True)

print("BrierScore Ranking")
for username, score in sorted(brier_score.items(), key=lambda x: x[1], reverse=True):
    print(f"User {username}: score {score}, rank {brier_ranking[username]}")



# # and use the 100 oldest problems
# prophet_problems = sorted(prophet_problems, key=lambda x: x.end_date, reverse=False)[:100]

# # 3. use the BrierScore to rank
# brier_score, brier_ranking = brier_scoring_rule.fit(prophet_problems, include_scores=True)

# # 4. print the ranking
# for username, score in brier_score.items():
#     print(f"User {username}: score {1 + score}, rank {brier_ranking[username]}")