## CHANGELOG FOR PM-RANK VERSIONS

### v0.2.22 (Current)

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


