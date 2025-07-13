from functools import partial
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.mcmc import NUTS, MCMC
from pyro.optim import Adam # type: ignore
import pyro
import pyro.distributions as dist
from typing import Literal, List, Dict, Any, Tuple

from model.utils import forecaster_data_to_rankings
from model.irt._dataset import _prepare_pyro_obs, IRTObs
from data.base import ForecastProblem

OUTPUT_DIR = __file__.replace(__file__.split("/")[-1], "output") # the output directory

class IRTModel(object):
    def __init__(self, n_bins: int = 6, use_empirical_quantiles: bool = False, device: Literal["cpu", "cuda"] = "cpu", method: Literal["SVI", "NUTS"] = "SVI"):
        self.n_bins = n_bins
        self.use_empirical_quantiles = use_empirical_quantiles
        self.device = device
        self.method = method
        # initiate pyro observations with None
        self.irt_obs = None

    def fit(self, problems: List[ForecastProblem], include_scores: bool = True, save_result: bool = False, \
        num_samples: int = 1000, warmup_steps: int = 100) -> Tuple[Dict[str, Any], Dict[str, int]] | Dict[str, int]:
        """ fit the model to the problems """
        self.irt_obs = _prepare_pyro_obs(problems, self.n_bins, self.use_empirical_quantiles, self.device)  # type: ignore

        # TODO: leverage the `mcmc` object as well in the future. Currently, we only need the samples
        posterior_samples = self._fit_pyro_model(self.irt_obs.forecaster_ids, self.irt_obs.problem_ids, self.irt_obs.discretized_scores, self.irt_obs.anchor_points, \
            num_samples=num_samples, warmup_steps=warmup_steps)

        self.posterior_samples = posterior_samples

        if save_result:
            import time
            torch.save(posterior_samples, f"{OUTPUT_DIR}/posterior_samples_{time.strftime("%m%d_%H%M")}.pt")

        return self._score_and_rank_forecasters(self.posterior_samples, include_scores=include_scores)

    def _model(self, forecaster_ids: torch.Tensor, problem_ids: torch.Tensor, discretized_scores: torch.Tensor, anchor_points: torch.Tensor):
        """
        The model that defines the IRT model.
        """
        # Infer N forecasters, M problems, and K observations from data
        N = int(forecaster_ids.max()) + 1
        M = int(problem_ids.max()) + 1
        K = len(anchor_points)

        # Define the forecaster-level ability parameters - `theta`
        with pyro.plate("forecasters", N, device=self.device):
            mean_theta, std_theta = torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)
            theta = pyro.sample("theta", dist.Normal(mean_theta, std_theta))

        # Define the problem-level difficulty parameters - `a` for discrimination and `b` for difficulty
        with pyro.plate("problems", M, device=self.device):
            std_a = torch.tensor(1.0, device=self.device)
            a = pyro.sample("a", dist.HalfNormal(std_a))
            mean_b, std_b = torch.tensor(0.0, device=self.device), torch.tensor(5.0, device=self.device)
            b = pyro.sample("b", dist.Normal(mean_b, std_b))

        # Define the category-level parameter - `p`
        with pyro.plate("categories", K, device=self.device):
            mean_p, std_p = torch.tensor(0.0, device=self.device), torch.tensor(5.0, device=self.device)
            p = pyro.sample("p", dist.Normal(mean_p, std_p))

        # --- Likelihood ---
        num_obs = len(forecaster_ids)

        with pyro.plate("data", num_obs, device=self.device):
            # get the forecaster and problem ids
            theta_i = theta[forecaster_ids] # shape: (num_obs,)
            a_j = a[problem_ids] # shape: (num_obs,)
            b_j = b[problem_ids] # shape: (num_obs,)

            # We use broadcasting to achieve this efficiently.
            # Shapes:
            # theta_i.unsqueeze(1) -> [num_observations, 1]
            # a_j.unsqueeze(1)     -> [num_observations, 1]
            # b_j.unsqueeze(1)     -> [num_observations, 1]
            # anchor_points        -> [K]
            # p                    -> [K]
            logits = (a_j.unsqueeze(1) * (1. - anchor_points) * (theta_i.unsqueeze(1) - (b_j.unsqueeze(1) + p))) # shape: (num_obs, K)

            # Now, we can sample from the Categorical distribution.
            pyro.sample("obs", dist.Categorical(logits=logits), obs=discretized_scores)
            
    def _guide(self, forecaster_ids: torch.Tensor, problem_ids: torch.Tensor, discretized_scores: torch.Tensor, bin_edges: torch.Tensor):
        """
        The guide that defines the IRT model.
        """
        pass

    def _fit_pyro_model(self, forecaster_ids: torch.Tensor, problem_ids: torch.Tensor, discretized_scores: torch.Tensor, anchor_points: torch.Tensor, \
        num_samples: int = 1000, warmup_steps: int = 100):
        """
        The core function that leverages pyro and SVI/NUTS to fit the model.
        """
        pyro.clear_param_store() # make sure the param store is empty
        assert self.irt_obs is not None, "IRT observations must be prepared before fitting the model"

        if self.method == "SVI":
            raise NotImplementedError("SVI is not implemented yet")
        elif self.method == "NUTS":
            nuts_kernel = NUTS(self._model, adapt_step_size=True)
            mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
            
            mcmc.run(
                forecaster_ids=forecaster_ids,
                problem_ids=problem_ids,
                discretized_scores=discretized_scores,
                anchor_points=anchor_points,
            )

            posterior_samples = mcmc.get_samples()

            return posterior_samples

    def _score_and_rank_forecasters(self, posterior_samples, include_scores: bool = True):
        """
        Take the posterior samples and take the scores to be the posterior mean of the theta.
        """
        assert self.irt_obs is not None, "IRT observations must be prepared before scoring and ranking forecasters"
        theta_means = posterior_samples["theta"].mean(dim=0)
        forecaster_idx_to_id = self.irt_obs.forecaster_idx_to_id

        forecaster_data = {}

        for i in range(len(theta_means)):
            forecaster_id = forecaster_idx_to_id[i]
            forecaster_data[forecaster_id] = theta_means[i].item()

        return forecaster_data_to_rankings(forecaster_data, include_scores=include_scores, ascending=False, aggregate="mean")

    def _summary(self, traces, sites):
        """Aggregate marginals for MCMC samples
        
        Args:
            traces: Dictionary of posterior samples from MCMC
            sites: List of site names to summarize
        """
        site_stats = {}
        for site_name in sites:
            if site_name in traces:
                # Extract samples for this site
                samples = traces[site_name].detach().cpu().numpy()
                
                # Reshape if needed - samples should be (num_samples, num_parameters)
                if len(samples.shape) == 1:
                    samples = samples.reshape(-1, 1)
                
                # Create DataFrame for each parameter
                for i in range(samples.shape[1]):
                    param_name = f"{site_name}_{i}" if samples.shape[1] > 1 else site_name
                    marginal_site = pd.DataFrame(samples[:, i]).transpose()
                    describe = partial(pd.Series.describe, percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
                    site_stats[param_name] = marginal_site.apply(describe, axis=1)[
                        ["mean", "std", "5%", "25%", "50%", "75%", "95%"]
                    ]
        return site_stats
    

# some in-file tests
if __name__ == "__main__":
    from data.loaders import GJOChallengeLoader

    predictions_file = "data/raw/all_predictions.json"
    metadata_file = "data/raw/sports_challenge_metadata.json"

    # load the data
    challenge_loader = GJOChallengeLoader(predictions_file, metadata_file, challenge_title="GJO Challenge")
    challenge = challenge_loader.load_challenge(forecaster_filter=20, problem_filter=20)

    irt_model = IRTModel(n_bins=6, use_empirical_quantiles=False, device="cpu", method="NUTS")
    fitted_scores, rankings = irt_model.fit(challenge.forecast_problems, save_result=True, num_samples=1000, warmup_steps=100)

    # print the results
    for forecaster, score in fitted_scores.items(): # type: ignore
        print(f"  {forecaster}: score={score}, rank={rankings[forecaster]}") # type: ignore
    
