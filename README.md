## Human/LLM Prediction Market Scraping & Analysis

This is a repo that contains the code to scrape the prediction market data from *Good Judgement Open* [GJO](https://goodjudgement.org/), interface with our own LLM prediction market (TBD), and analyze (estimation & ranking) the forecasters/predictors.

### File Structure

- `crawler/`: contains the code to scrape the prediction market data from GJO
- `data/`: contains the scraped data
- `notebooks/`: contains the notebooks for the analysis

### Scraping Data

`crawler/scrape_gjo_problem_data.py`: 
> scrapes the problem data from GJO website using `requests` and `BeautifulSoup`. This step will gives us a metadata JSON file, e.g. `data/xxx_challenge_metadata.json`. Each entry in the metadata JSON file is a problem, with the following structure:

```json
{
    "problem_id": "3940",
    "title": "Who will win the NFL Most Valuable Player Award for the 2024 season?",
    "url": "https://www.gjopen.com/questions/3940-who-will-win-the-nfl-most-valuable-player-award-for-the-2024-season",
    "metadata": {
        "status": "Closed",
        "end_date": "2025-02-07T03:40:25Z",
        "num_forecasters": 61,
        "num_forecasts": 147
    },
    "options": [
        "Josh Allen",
        "Saquon Barkley",
        "Sam Darnold",
        "Jared Goff",
        "Lamar Jackson",
        "Someone else"
    ],
    "correct_answer": "Josh Allen"
}
```

`crawler/scrape_gjo_predictions_data.py`:
> scrapes the prediction data from GJO website using `Playwright`. This would require the additional installation of `Playwright` browser kernels via `playwright install`. This step will gives us a predictions JSON file, e.g. `data/all_predictions.json`. Each entry in the predictions JSON file is a prediction, with the following structure:

```json
{
    "problem_id": "3940",
    "username": "Jonah-Neuwirth",
    "timestamp": "2025-02-06T05:44:32Z",
    "prediction": [0.2,0.0,0.0,0.0,0.8,0.0]
}
```

> the username is an unique identifier for each forecaster. The prediction is a list of probabilities for the options specified in the problem metadata.

> 🔥 Warning: please respect the rate limit of GJO website when scraping any information. Our script does this by adding a random sleep time bewteen the requests.

