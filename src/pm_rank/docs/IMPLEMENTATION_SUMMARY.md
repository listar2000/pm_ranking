# Implementation Summary: Calibration Metric and Bootstrap CI

This document summarizes the implementation of the calibration metric (ECE score) and bootstrap confidence intervals for the new pandas-based nightly API.

## 1. Calibration Metric (ECE Score)

### File: `algo.py` - `compute_calibration_ece()`

**Purpose**: Calculate the Expected Calibration Error (ECE) for each forecaster to measure how well-calibrated their probability predictions are.

**Key Features**:
- Combines two types of weights:
  1. **Prediction-level weight**: From the `weight` column (assigned by `weight_fn` in data loading)
  2. **Market-level weight**: Either uniform (1.0) or inverse of number of markets per prediction (controlled by `weight_event` parameter)
  
- Final weight for each market probability = `prediction_weight × market_weight`

**Parameters**:
- `forecasts`: DataFrame with columns (forecaster, event_ticker, round, prediction, outcome, weight)
- `num_bins`: Number of bins for discretization (default: 10)
- `strategy`: "uniform" or "quantile" binning (default: "uniform")
- `weight_event`: If True, weight each market by 1/num_markets; if False, all markets weighted equally (default: True)

**Returns**: DataFrame with columns (forecaster, ece_score), sorted by ECE (lower is better)

**Implementation Notes**:
- Reuses helper functions `_bin_stats` and `_calculate_ece` from the old API (`pm_rank.model.calibration`)
- Unlike Brier/Average Return, ECE is calculated holistically (not per-prediction) due to binning requirements
- Result is already aggregated per forecaster (no need for weighted averaging in ranking function)

## 2. Bootstrap Confidence Intervals

### File: `bootstrap.py` - `compute_bootstrap_ci()`

**Purpose**: Compute bootstrap confidence intervals for forecaster scores using weighted resampling of individual predictions.

**Key Features**:
- Simplified configuration via dictionary (no complex Pydantic models)
- Always uses symmetric confidence intervals around point estimates
- Supports both methods:
  1. **num_se**: Specify number of standard errors (e.g., 1.96 for 95% CI)
  2. **ci_level**: Use symmetric deviation method (finds percentile of deviations from point estimate)

**Bootstrap Process**:
1. Sample predictions with replacement using adjusted weights as sampling probabilities
2. For each resample, calculate weighted average score for each forecaster
3. Repeat `num_samples` times to build bootstrap distribution
4. Calculate standard error and confidence intervals from the distribution

**Parameters** (via `bootstrap_config` dict):
- `num_samples`: Number of bootstrap samples (default: 1000)
- `ci_level`: Confidence level for symmetric CI (default: 0.95)
- `num_se`: Number of standard errors for CI bounds (default: None, uses ci_level)
- `random_seed`: Random seed for reproducibility (default: 42)
- `show_progress`: Whether to show progress bar (default: True)

**Returns**: Tuple of (standard_errors, confidence_intervals)
- `standard_errors`: Dict mapping forecaster → SE
- `confidence_intervals`: Dict mapping forecaster → (lower, upper)

## 3. Updated Ranking Function

### File: `algo.py` - `rank_forecasters_by_score()`

**Enhanced Features**:
- Now supports optional `bootstrap_config` parameter
- When provided, adds three columns to output: `se`, `lower`, `upper`
- Bootstrap CI only supported for `brier_score` and `average_return` (not `ece_score`)

**Example Usage**:

```python
# Without Bootstrap CI
brier_scores = compute_brier_score(forecasts)
rankings = rank_forecasters_by_score(brier_scores, normalize_by_round=True)
# Output columns: forecaster, score

# With Bootstrap CI
bootstrap_config = {
    'num_samples': 1000,
    'num_se': 1.96,  # ±1.96 SE for 95% CI
    'random_seed': 42,
    'show_progress': True
}
rankings_with_ci = rank_forecasters_by_score(
    brier_scores, 
    normalize_by_round=True, 
    bootstrap_config=bootstrap_config
)
# Output columns: forecaster, score, se, lower, upper
```

## 4. Test Files

### `test_calibration.py`
- Tests ECE calculation with synthetic data
- Validates that well-calibrated forecasters get lower ECE scores
- Demonstrates both `weight_event=True` and `weight_event=False` modes

### `test_bootstrap.py`
- Tests bootstrap CI calculation for both Brier score and Average Return
- Demonstrates two CI methods: `num_se` and `ci_level`
- Validates that CIs reflect uncertainty in forecaster rankings

## 5. Design Decisions

### Calibration
1. **Two-level weighting**: Combines prediction-level and market-level weights to properly account for both submission frequency and event complexity
2. **Reuse existing helpers**: Leverages `_bin_stats` and `_calculate_ece` from old API to maintain consistency
3. **Simplified strategy**: Always "uniform" binning for market-level weights (no complex weighting schemes)

### Bootstrap
1. **Dictionary config**: Simpler than Pydantic models, more flexible for users
2. **Weighted resampling**: Properly accounts for prediction weights in bootstrap sampling
3. **Symmetric CIs**: Easier to interpret and implement than asymmetric methods
4. **Two CI methods**: Supports both normal-theory (num_se) and percentile-based (ci_level) approaches

### Integration
1. **Backward compatible**: Existing code works without changes; bootstrap is opt-in
2. **Consistent API**: Same `rank_forecasters_by_score` function for all metrics
3. **Clear output**: Additional columns (se, lower, upper) clearly indicate CI information

## 6. Usage Examples

See test files and `__main__` sections in `algo.py` and `data.py` for complete working examples.

### Quick Example: Calibration
```python
from pm_rank.nightly.data import NightlyForecasts, uniform_weighting
from pm_rank.nightly.algo import compute_calibration_ece, rank_forecasters_by_score

forecasts = NightlyForecasts.from_prophet_arena_csv(pred_csv, sub_csv, uniform_weighting())
ece_results = compute_calibration_ece(forecasts.data, num_bins=10, weight_event=True)
rankings = rank_forecasters_by_score(ece_results)
print(rankings)
```

### Quick Example: Bootstrap CI
```python
from pm_rank.nightly.algo import compute_brier_score, rank_forecasters_by_score

brier_scores = compute_brier_score(forecasts.data)
bootstrap_config = {'num_samples': 1000, 'num_se': 1.96, 'random_seed': 42}
rankings = rank_forecasters_by_score(brier_scores, bootstrap_config=bootstrap_config)
print(rankings)  # Shows: forecaster, score, se, lower, upper
```

