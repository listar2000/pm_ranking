## GJO Crawler

### Overview of the crawling pipeline

We will specify a "challenge" name or id, where a challenge is a collection of prediction problems (e.g. a challenge can be "economic condition in 2024" and a prediction problem can be "GDP growth rate in 2024"). We then need to extract all the problem ids from the challenge page. Each problem will have meta-data like all the possible options, and the correct answer (since we only care about problems already resolved, i.e. the correct answer is known). After this, we will also extract all the "user predictions" for each problem.

For a given problem, a user prediction will contain the username, the prediction will be placed as a dictionary of <option, probability> (need to sum to 1) given by the user. In JSON format, this will be like:

```json
{
    "username": "John Doe",
    "problem_id": "123",
    "prediction": {
        "option1": 0.5,
        "option2": 0.3,
        "option3": 0.2,
    },
    // some timestamp data that for our reference
    "timestamp": "2024-01-01 12:00:00" 
}
```
The JSON for the problem meta-data will be like:
```json
{
    "problem_id": "123",
    "options": ["option1", "option2", "option3"],
    "correct_answer": "option1"
}
```








