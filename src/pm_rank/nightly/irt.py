from typing import List, Literal, Tuple

import pandas as pd
import torch

from pm_rank.model.irt._dataset import IRTObs, _discretize_scoring_rules
from pm_rank.model.irt._pyro_models import MCMCConfig, IRTModel


def bipartite_core(results_df: pd.DataFrame, min_forecaster: int = 10, min_problem: int = 10, verbose: bool = False) -> Tuple[List[str], List[str]]:
    """
    Perform a bipartite core algorithm on the results dataframe to find the core forecasters and problems.
    We are given a dataframe containing "forecaster" and "problem" columns. We want to iteratively prune the set
    of forecasters and problems until each forecaster has predicted at least `min_problem` problems and each 
    problem has been predicted at least `min_forecaster` times.

    Return:
        A tuple of two lists, the first list the remaining forecasters, the second list the remaining problems.
    """
    df = results_df.copy()
    if "problem" not in df.columns:
        df['problem'] = df['event_ticker'] + ':' + df['round'].astype(str)

    # Keep only the relevant columns for the bipartite graph
    edges = df[['forecaster', 'problem']].drop_duplicates()

    if verbose:
        print(f"Initial number of edges: {len(edges)}")
        print(f"Initial number of forecasters: {len(edges['forecaster'].unique())}")
        print(f"Initial number of problems: {len(edges['problem'].unique())}")

    iter_count = 0
    while True:
        prev_len = len(edges)

        # Count problems per forecaster and filter
        forecaster_counts = edges.groupby('forecaster').size()
        valid_forecasters = forecaster_counts[forecaster_counts >= min_problem].index
        edges = edges[edges['forecaster'].isin(valid_forecasters)]

        # Count forecasters per problem and filter
        problem_counts = edges.groupby('problem').size()
        valid_problems = problem_counts[problem_counts >= min_forecaster].index
        edges = edges[edges['problem'].isin(valid_problems)]

        iter_count += 1
        if verbose:
            print(f"Iteration {iter_count}: Number of edges after filtering: {len(edges)}")
            print(f"Iteration {iter_count}: Number of forecasters after filtering: {len(edges['forecaster'].unique())}")
            print(f"Iteration {iter_count}: Number of problems after filtering: {len(edges['problem'].unique())}")

        # Check for convergence
        if len(edges) == prev_len:
            print(f"Bipartite core algorithm converged after {iter_count} iterations")
            break

    return edges['forecaster'].unique().tolist(), edges['problem'].unique().tolist()


def _prepare_irt_obs_from_df(
    results_df: pd.DataFrame,
    score_col: str = "brier_score",
    n_points: int = 10,
    use_empirical_quantiles: bool = True,
    device: Literal["cpu", "cuda"] = "cpu",
) -> IRTObs:
    # step 1: create a combined column "problem" that concatenates `event_ticker` and `round`
    if "problem" not in results_df.columns:
        results_df['problem'] = results_df['event_ticker'] + ':' + results_df['round'].astype(str)

    # step 2: factorize the forecaster and problem columns, respectively
    forecaster_ids, unique_forecasters = pd.factorize(results_df['forecaster'])
    problem_ids, unique_problems = pd.factorize(results_df['problem'])

    forecaster_id_to_idx = {forecaster: idx for idx, forecaster in enumerate(unique_forecasters)}
    problem_id_to_idx = {problem: idx for idx, problem in enumerate(unique_problems)}

    scores = results_df[score_col].values
    discretized_scores, anchor_points = _discretize_scoring_rules(scores, n_points=n_points, use_empirical_quantiles=use_empirical_quantiles)
    return IRTObs(
        forecaster_ids=torch.tensor(forecaster_ids, device=device, dtype=torch.long),
        problem_ids=torch.tensor(problem_ids, device=device, dtype=torch.long),
        forecaster_id_to_idx=forecaster_id_to_idx,
        problem_id_to_idx=problem_id_to_idx,
        scores=torch.tensor(scores, device=device, dtype=torch.float),
        discretized_scores=torch.tensor(discretized_scores, device=device, dtype=torch.long),
        anchor_points=torch.tensor(anchor_points, device=device, dtype=torch.float),
    )


if __name__ == "__main__":
    predictions_csv = "slurm/predictions_12_31_to_01_01.csv"  # Your predictions CSV file
    submissions_csv = "slurm/submissions_12_31_to_01_01.csv"  # Your submissions CSV file

    N_POINTS = 10
    USE_EMPIRICAL_QUANTILES = True
    DEVICE = "cpu"

    from pm_rank.nightly.algo import compute_brier_score
    from pm_rank.nightly.data import NightlyForecasts, uniform_weighting
    
    weight_fn = uniform_weighting()
    forecasts = NightlyForecasts.from_prophet_arena_csv(predictions_csv, submissions_csv, weight_fn)

    results_df = compute_brier_score(forecasts.data)
    # remove any row where brier score is NaN or does not lie within [0, 1]
    results_df = results_df[results_df['brier_score'].notna() & (results_df['brier_score'] >= 0) & (results_df['brier_score'] <= 1)]
    results_df["problem"] = results_df['event_ticker'] + ':' + results_df['round'].astype(str)

    core_forecasters, core_problems = bipartite_core(results_df, verbose=True, min_problem=100, min_forecaster=20)
    print(f"Core forecasters: {core_forecasters}")

    # filter the results dataframe to only include the core forecasters and problems
    results_df = results_df[results_df['forecaster'].isin(core_forecasters) & results_df['problem'].isin(core_problems)]
    print(f"Number of rows in the results dataframe after filtering by core bipartite: {len(results_df)}")

    irt_obs = _prepare_irt_obs_from_df(results_df, n_points=N_POINTS, use_empirical_quantiles=USE_EMPIRICAL_QUANTILES, device=DEVICE)

    mcmc_config = MCMCConfig(total_samples=1000, warmup_steps=100, num_workers=1, device=DEVICE, save_result=True)

    irt_model = IRTModel(n_bins=N_POINTS, use_empirical_quantiles=USE_EMPIRICAL_QUANTILES, verbose=True)
    irt_result = irt_model.fit(irt_obs, method="NUTS", config=mcmc_config)
    print(irt_result)