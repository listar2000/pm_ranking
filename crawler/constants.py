# Good Judgement Open

GJO_BASE_URL = "https://goodjudgement.org"

GJO_CHALLENGE_URL = "https://www.gjopen.com/challenges/{challenge_id}?status=resolved&page={page}"

GJO_COMMENT_URL = "https://www.gjopen.com/comments?commentable_id={commentable_id}&commentable_type=Forecast%3A%3AQuestion&page={page}"


if __name__ == "__main__":
    # here we scrape some example pages and store them into the `html` folder
    import requests

    # example challenge
    # example_challenge_id, example_page = 97, 3

    # url = GJO_CHALLENGE_URL.format(challenge_id=example_challenge_id, page=example_page)
    # response = requests.get(url)
    # with open(f"crawler/html/challenge_{example_challenge_id}_{example_page}.html", "w") as f:
    #     f.write(response.text)

    # example predictions from users
    # comment_url = GJO_COMMENT_URL.format(commentable_id=3700, page=1)
    comment_url = "https://www.gjopen.com/comments?id=3700-which-college-football-team-will-win-the-big-12-conference-championship-in-the-2024-season&filters=&commentable_id=3700&commentable_type=Forecast%3A%3AQuestion&page=2"

    # print(comment_url)
    response = requests.get(comment_url)
    with open(f"crawler/html/comment_3700_2.html", "w") as f:
        f.write(response.text)