## CHANGELOG FOR PM-RANK VERSIONS

### v0.3.1 (Current)

- Anri fixed some bugs with the new average return algorithm.

### v0.3.0

- We now start to adopt a new algorithm for calculating the average return.

### v0.2.33

- Add more information when ranking the forecasters by Brier score or market return.

### v0.2.28

- Add the algorithm to calculate Sharpe ratio.
- Add streaming and categorization support.

### v0.2.27

- Add better handling logic for calculating each submission/prediction's time to the market close timestamp.

### v0.2.26

- **Nightly Version 🚀**: We add a new module `pm_rank.nightly` for scoring and ranking nightly predictions. This is to prepare for the upcoming API upgrade from the existing version and aims at improving the speed of loading data and making evaluations.

### v0.2.25

- **Minor Bugfix 🐛**: when the `odds` or `no_odds` info is lacking for certain markets, we now set the implied probabilities to be 1.0 (i.e. there is no surprising arbitrage opportunity) instead of 1e-3.

- The sharpe ratio marginal calculation (i.e. when `sharpe_mode="marginal"`) will still use default baseline of $`num_money_per_round` (constant).

### v0.2.24

- **Major Bugfix 🐛**: the `ForecastEvent` object now _independently stores the `odds` and `no_odds` fields_, which becomes consistent with the fact that different forecasts (to the same `ForecastProblem`) can have different `odds` and `no_odds` fields. As a result, the calculation of `AverageReturn` (and any related metrics) are now more accurate.

- Add a new `sharpe_mode` parameter to the `.fit` method of `AverageReturn` to support the calculation of the [Sharpe Ratio](https://en.wikipedia.org/wiki/Sharpe_ratio). Specifically, we can now specify the `sharpe_mode` parameter to be `"marginal"` or `"relative"` (default is `None`). The former is the marginal sharpe ratio, i.e. the sharpe ratio calculated on the forecasters' earnings only. The latter is the relative sharpe ratio, i.e. the sharpe ratio calculated on the forecasters' earnings minus the baseline earnings.

    Sample usage:
    ```python
    from pm_rank.model.average_return import AverageReturn
    average_return_config = AverageReturnConfig(xxx)
    average_return = AverageReturn(average_return_config)
    # NOTE: new parameter `sharpe_mode` is added
    ar_score, ar_ranking = average_return.fit(prophet_problems, sharpe_mode="relative", include_scores=True, include_per_problem_info=False, include_bootstrap_ci=False)
    ```
### v0.2.23

- Support for an important functionality: **measuring the calibration of probablistic predictors**. Specifically:
    - The `pm_rank.model.CalibrationMetric` class implements the usual model `.fit` method that takes in a list of `ForecastProblem` and returns a ranking of forecasters -- ranked by their expected calibration error (ECE, smaller is better).

    - Once the model is fitted, you can draw a reliability/calibration diagram to visualize the calibration errors via `model.plot(llm_name)`.

    - Demo usage:
        ```python
        from pm_rank.model.calibration import CalibrationMetric

        # assume you already have a list of `ForecastProblem` instances in `problems`
        calibration_metric = CalibrationMetric(num_bins=10, strategy="uniform", weight_event=False)
        calibration_scores, calibration_ranking = calibration_metric.fit(problems, include_scores=True)
        ```
        The resulting `calibration_scores` and `calibration_ranking` are dictionaries that map the forecaster name to their calibration score and ranking. You can print them out via `_format_ranking_table`, similar to other metric results.


### v0.2.22

- For `ProphetArenaChallengeLoader`, we do some additional checks to ensure the unnormalized probabilities will not error.

### v0.2.21

- `ForecastEvent`: we make the following specifications.
    - add a new field `weight`: the weight of the forecast. This is used to weight the forecast in scoring/ranking. Default to 1. This should be supplemented by user of `pm_rank`.
    - we explicitly specify that `forecast_id` is the unique identifier of `ForecastEvent`.
    - we add a subclass `ProphetArenaForecastEvent` for Prophet Arena's specific attributes. This helps the generic `ForecastEvent` to be as widely applicable as possible. Specifically, it has a new field `submission_id` to identify the submission batch.

- Per-problem info returned by `AverageReturn` and `ScoringRule` now includes the `submission_id` of the forecast as well.

- We implement from scratch the **nonparametric bootstrap confidence intervals** for `AverageReturn` and `ScoringRule`. Below we give a brief demo of how to obtain the CIs together with (previously available) point estimates.

    1. We specify a configuration `BootstrapCIConfig` for the CI:
        ```python
        from pm_rank.model.utils import BootstrapCIConfig
        bootstrap_ci_config = BootstrapCIConfig(num_bootstrap_samples=1000, bootstrap_ci_level=0.95, random_seed=42, symmetric=True)
        ```
        _or you can use the default configuration_
        ```python
        from pm_rank.model.utils import DEFAULT_BOOTSTRAP_CI_CONFIG
        ```
    
    2. When calling `.fit` method, we can specify the `include_bootstrap_ci` flag to be True and pass in the `bootstrap_ci_config` configuration.
        ```python
        # use the scoring rule for example; same for `AverageReturn`
        from pm_rank.model.scoring_rule import BrierScoringRule

        brier_scoring_rule = BrierScoringRule()
        brier_score, brier_ranking, brier_bootstrap_ci, brier_per_problem_info = \
            brier_scoring_rule.fit(problems, include_bootstrap_ci=True, \
            include_scores=True, include_per_problem_info=True, \
            bootstrap_ci_config=bootstrap_ci_config)
        ```

    3. The returned results will now include the bootstrap confidence intervals. You can still use the built-in `_format_ranking_table` function to print the results.
        ```python
        from pm_rank.utils import _format_ranking_table
        print(_format_ranking_table(brier_ranking, brier_score, brier_bootstrap_ci))
        ```


