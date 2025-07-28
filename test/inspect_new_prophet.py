import random
from pm_rank.model import BrierScoringRule, AverageReturn
from pm_rank.data.loaders import ProphetArenaChallengeLoader

prophet_loader = ProphetArenaChallengeLoader(
    predictions_file="src/pm_rank/data/raw/new_prophet_arena_data.csv",
    use_open_time=True
)

prophet_challenge = prophet_loader.load_challenge()
prophet_problems = prophet_challenge.get_problems()

# randomly give category to problems
for problem in prophet_problems:
    problem.category = random.choice(["A", "B", "C"])

brier_scoring_rule = BrierScoringRule()

# use the AverageReturn to rank
average_return = AverageReturn()
# average_return_results = average_return.fit_by_category(prophet_problems, include_scores=True)

# print("--------------------------------")

# # print the ranking
# print("AverageReturn Ranking")
# for category, results in average_return_results.items():
#     print(f"Category {category}:")
#     avg_scores, avg_rankings = results
#     for username, score in sorted(avg_scores.items(), key=lambda x: x[1], reverse=True):
#         print(f"{username}: score {score}, rank {avg_rankings[username]}")
#     print("--------------------------------")

# brier_results = brier_scoring_rule.fit_by_category(prophet_problems, include_scores=True)

# print("BrierScore Ranking")
# for category, results in brier_results.items():
#     print(f"Category {category}:")
#     brier_scores, brier_rankings = results
#     for username, score in sorted(brier_scores.items(), key=lambda x: x[1], reverse=True):
#         print(f"{username}: score {score}, rank {brier_rankings[username]}")
#     print("--------------------------------")


average_return_results_stream = average_return.fit_by_category(prophet_problems, include_scores=True, \
    stream_with_timestamp=True, stream_increment_by="day", min_bucket_size=5)

for category, results in average_return_results_stream.items():
    print("**" * 50)
    print(f"Category {category}:")
    for timestamp, result in results.items():
        print("--" * 10)
        print(f"Timestamp {timestamp}:")
        avg_scores, avg_rankings = result
        for username, score in sorted(avg_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"{username}: score {score}, rank {avg_rankings[username]}")

"""
{
    "overall": {
        "2025-01-01": {
            "avg_scores": {
                "user1": 0.5,
                "user2": 0.3
            },
            "avg_rankings": {
                "user1": 1,
                "user2": 2
            }
        }
    }
    "A": {
        "2025-01-01": {
            "avg_scores": {
                "user1": 0.5,
                "user2": 0.3
            },
            "avg_rankings": {
                "user1": 1,
                "user2": 2
            }
        },
        "2025-01-02": {
            "avg_scores": {
                "user1": 0.5,
                "user2": 0.3
            },
            "avg_rankings": {
                "user1": 1,
                "user2": 2
            }
        }
    },
    "B": {
        "2025-01-01": {
            "avg_scores": {
                "user1": 0.5,
                "user2": 0.3
            },
            "avg_rankings": {
                "user1": 1,
                "user2": 2
            }
        }
    },
    "C": {
        "2025-01-01": {
            "avg_scores": {
                "user1": 0.5,
                "user2": 0.3
            },
            "avg_rankings": {
                "user1": 1,
                "user2": 2
            }
        }
    }
}
"""


# # and use the 100 oldest problems
# prophet_problems = sorted(prophet_problems, key=lambda x: x.end_time, reverse=False)[:100]

# # 3. use the BrierScore to rank
# brier_score, brier_ranking = brier_scoring_rule.fit(prophet_problems, include_scores=True)

# # 4. print the ranking
# for username, score in brier_score.items():
#     print(f"User {username}: score {1 + score}, rank {brier_ranking[username]}")