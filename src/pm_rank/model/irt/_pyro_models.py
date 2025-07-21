from functools import partial
import torch
import numpy as np

from pyro.infer import SVI, Trace_ELBO
from pyro.infer.mcmc import NUTS, MCMC, HMC
from pyro.optim import Adam, SGD # type: ignore
import pyro
import pyro.distributions as dist
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any, Tuple

from pm_rank.model.utils import forecaster_data_to_rankings
from pm_rank.model.irt._dataset import _prepare_pyro_obs
from pm_rank.data.base import ForecastProblem
from pm_rank.data.loaders import GJOChallengeLoader

OUTPUT_DIR = __file__.replace(__file__.split("/")[-1], "output") # the output directory

"""
Configurations for running MCMC and the SVI inference engines.
"""
class MCMCConfig(BaseModel):
    total_samples: int = Field(default=1000, description="The total number of samples to draw from the posterior distribution.")
    warmup_steps: int = Field(default=100, description="The number of warmup steps to run before sampling.")
    num_workers: int = Field(default=1, description="The number of workers to use for parallelization. Note that we use a customized multiprocessing approach \
        since the default implementation by Pyro can be very slow. This is why we don't use the name `num_chains`.")
    device: Literal["cpu", "cuda"] = Field(default="cpu", description="The device to use for the MCMC engine.")
    save_result: bool = Field(default=False, description="Whether to save the result to a file.")

class SVIConfig(BaseModel):
    optimizer: Literal["Adam", "SGD"] = Field(default="Adam", description="The optimizer to use for the SVI engine.")
    num_steps: int = Field(default=1000, description="The number of steps to run for the SVI engine.")
    learning_rate: float = Field(default=0.01, description="The learning rate to use for the SVI engine.")
    device: Literal["cpu", "cuda"] = Field(default="cpu", description="The device to use for the SVI engine.")

class IRTModel(object):
    def __init__(self, n_bins: int = 6, use_empirical_quantiles: bool = False):
        self.n_bins = n_bins
        self.use_empirical_quantiles = use_empirical_quantiles
        # initiate pyro observations with None
        self.irt_obs = None
        self.method = None

    def fit(self, problems: List[ForecastProblem], include_scores: bool = True, method: Literal["SVI", "NUTS"] = "SVI", \
        config: MCMCConfig | SVIConfig | None = None) -> Tuple[Dict[str, Any], Dict[str, int]] | Dict[str, int]:
        """ fit the model to the problems """

        assert method in ["SVI", "NUTS"], "Invalid method. Must be either 'SVI' or 'NUTS'."
        assert config is not None, "Configuration must be provided."

        self.method = method
        if self.method == "SVI":
            assert isinstance(config, SVIConfig), "SVI configuration must be provided."
        elif self.method == "NUTS":
            assert isinstance(config, MCMCConfig), "MCMC configuration must be provided."

        self.device = config.device
        self.irt_obs = _prepare_pyro_obs(problems, self.n_bins, self.use_empirical_quantiles, self.device)  # type: ignore

        if self.method == "NUTS":
            mcmc_config: MCMCConfig = config # type: ignore
            posterior_samples = self._fit_pyro_model_mcmc(self.irt_obs.forecaster_ids, self.irt_obs.problem_ids, self.irt_obs.discretized_scores, self.irt_obs.anchor_points, \
                num_samples=mcmc_config.total_samples, warmup_steps=mcmc_config.warmup_steps, num_chains=mcmc_config.num_workers)

            self.posterior_samples = posterior_samples

            if mcmc_config.save_result:
                import time
                torch.save(posterior_samples, f"{OUTPUT_DIR}/posterior_samples_{time.strftime("%m%d_%H%M")}.pt")

            return self._score_and_rank_mcmc(self.posterior_samples, include_scores=include_scores)
        else:
            svi_config: SVIConfig = config # type: ignore
            fitted_params = self._fit_pyro_model_svi(self.irt_obs.forecaster_ids, self.irt_obs.problem_ids, self.irt_obs.discretized_scores, self.irt_obs.anchor_points, \
                optimizer=svi_config.optimizer, num_steps=svi_config.num_steps, learning_rate=svi_config.learning_rate)

            self.fitted_params = fitted_params

            return self._score_and_rank_svi(self.fitted_params, include_scores=include_scores)

    def get_problem_level_parameters(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Returns the problem difficulties and discriminations. The method must be used after the model has been fit.
        """
        # make sure the model has been fit
        assert self.method is not None, "IRT model must be fit before getting problem attributes"

        if self.method == "SVI":
            raw_problem_difficulties = self.fitted_params["svi_mean_b"]
            raw_problem_discriminations = self.fitted_params["svi_mean_a"]

        elif self.method == "NUTS":
            raw_problem_difficulties = self.posterior_samples["b"].mean(dim=0)
            raw_problem_discriminations = self.posterior_samples["a"].mean(dim=0)

        problem_diff_dict, problem_discrim_dict = {}, {}
        for problem_id, idx in self.irt_obs.problem_id_to_idx.items():
            problem_diff_dict[problem_id] = raw_problem_difficulties[idx]
            problem_discrim_dict[problem_id] = raw_problem_discriminations[idx]
        return problem_diff_dict, problem_discrim_dict

    def _model(self, forecaster_ids: torch.Tensor, problem_ids: torch.Tensor, discretized_scores: torch.Tensor, anchor_points: torch.Tensor):
        """
        The model that defines the IRT model.
        """
        # Infer N forecasters, M problems, and K anchor points from data
        N = int(forecaster_ids.max()) + 1
        M = int(problem_ids.max()) + 1
        K = len(anchor_points)

        # Define the forecaster-level ability parameters - `theta`
        with pyro.plate("forecasters", N, device=self.device):
            mean_theta, std_theta = torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)
            theta = pyro.sample("theta", dist.Normal(mean_theta, std_theta))

        # Define the problem-level difficulty parameters - `a` for discrimination and `b` for difficulty
        with pyro.plate("problems", M, device=self.device):
            std_a = torch.tensor(5.0, device=self.device)
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
        The guide that defines the IRT model. Used for Stochastic Variational Inference (SVI).
        """
        # Infer N forecasters, M problems, and K anchor points from data
        N = int(forecaster_ids.max()) + 1
        M = int(problem_ids.max()) + 1
        K = len(bin_edges)
        # set up all the parameters (in a mean-field way)
        mean_theta_param = pyro.param("mean_theta", torch.zeros(N, device=self.device))
        std_theta_param = pyro.param("std_theta", torch.ones(N, device=self.device), constraint=dist.constraints.positive)

        std_a_param = pyro.param("std_a", torch.empty(M, device=self.device).fill_(5.0), constraint=dist.constraints.positive)

        mean_b_param = pyro.param("mean_b", torch.zeros(M, device=self.device))
        std_b_param = pyro.param("std_b", torch.empty(M, device=self.device).fill_(5.0), constraint=dist.constraints.positive)

        mean_p_param = pyro.param("mean_p", torch.zeros(K, device=self.device))
        std_p_param = pyro.param("std_p", torch.empty(K, device=self.device).fill_(5.0), constraint=dist.constraints.positive)

        with pyro.plate("forecasters", N, device=self.device):
            theta = pyro.sample("theta", dist.Normal(mean_theta_param, std_theta_param))

        with pyro.plate("problems", M, device=self.device):
            a = pyro.sample("a", dist.HalfNormal(std_a_param))
            b = pyro.sample("b", dist.Normal(mean_b_param, std_b_param))

        with pyro.plate("categories", K, device=self.device):
            p = pyro.sample("p", dist.Normal(mean_p_param, std_p_param))

    def _fit_pyro_model_mcmc(self, forecaster_ids: torch.Tensor, problem_ids: torch.Tensor, discretized_scores: torch.Tensor, anchor_points: torch.Tensor, \
        num_samples: int = 1000, warmup_steps: int = 100, num_chains: int = 1):
        """
        The core function that leverages pyro and NUTS to fit the model.
        """
        pyro.clear_param_store() # make sure the param store is empty
        assert self.irt_obs is not None, "IRT observations must be prepared before fitting the model"

        nuts_kernel = NUTS(self._model, adapt_step_size=True)
        mp_context = "spawn" if num_chains > 1 and self.device == "cuda" else None
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=num_chains, mp_context=mp_context)
        
        mcmc.run(
            forecaster_ids=forecaster_ids,
            problem_ids=problem_ids,
            discretized_scores=discretized_scores,
            anchor_points=anchor_points,
        )

        posterior_samples = mcmc.get_samples()

        return posterior_samples

    def _fit_pyro_model_svi(self, forecaster_ids: torch.Tensor, problem_ids: torch.Tensor, discretized_scores: torch.Tensor, anchor_points: torch.Tensor, \
        optimizer: Literal["Adam", "SGD"] = "Adam", num_steps: int = 1000, learning_rate: float = 0.01):
        """
        The core function that leverages pyro and SVI to fit the model.
        """
        from tqdm import tqdm

        pyro.clear_param_store() # make sure the param store is empty
        assert self.irt_obs is not None, "IRT observations must be prepared before fitting the model"

        assert optimizer in ["Adam", "SGD"], "Invalid optimizer. Must be either 'Adam' or 'SGD'."
        optim = Adam({"lr": learning_rate}) if optimizer == "Adam" else SGD({"lr": learning_rate})

        svi = SVI(self._model, self._guide, optim, loss=Trace_ELBO())

        pbar = tqdm(range(num_steps))
        for i in pbar:
            loss = svi.step(forecaster_ids, problem_ids, discretized_scores, anchor_points)
            if i % 20 == 0:
                pbar.set_description(f"SVI [Loss: {loss:5.1f}]")

        return {
            "svi_mean_thetas": pyro.param("mean_theta").detach().cpu().numpy(),
            "svi_mean_a": pyro.param("std_a").detach().cpu().numpy() * np.sqrt(2 / np.pi), # special case for the half-normal distribution
            "svi_mean_b": pyro.param("mean_b").detach().cpu().numpy(),
            "svi_mean_p": pyro.param("mean_p").detach().cpu().numpy(),
        }

    def _score_and_rank_helper(self, theta_means, include_scores: bool = True):
        """
        Shared logic to map theta means to forecaster IDs and compute rankings.
        """
        assert self.irt_obs is not None, "IRT observations must be prepared before scoring and ranking forecasters"
        forecaster_idx_to_id = self.irt_obs.forecaster_idx_to_id
        forecaster_data = {}
        for i in range(len(theta_means)):
            forecaster_id = forecaster_idx_to_id[i]
            # theta_means may be a numpy array or torch tensor; .item() works for both
            forecaster_data[forecaster_id] = theta_means[i].item() if hasattr(theta_means[i], 'item') else float(theta_means[i])
        return forecaster_data_to_rankings(forecaster_data, include_scores=include_scores, ascending=False, aggregate="mean")

    def _score_and_rank_mcmc(self, posterior_samples, include_scores: bool = True):
        """
        Take the posterior samples and take the scores to be the posterior mean of the theta.
        """
        theta_means = posterior_samples["theta"].mean(dim=0)
        return self._score_and_rank_helper(theta_means, include_scores=include_scores)

    def _score_and_rank_svi(self, fitted_params: Dict[str, Any], include_scores: bool = True):
        """
        Take the fitted parameters and take the scores to be the posterior mean of the theta.
        """
        theta_means = fitted_params["svi_mean_thetas"]
        return self._score_and_rank_helper(theta_means, include_scores=include_scores)

    def _summary(self, traces, sites):
        """Aggregate marginals for MCMC samples
        
        Args:
            traces: Dictionary of posterior samples from MCMC
            sites: List of site names to summarize
        """
        import pandas as pd

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
    # load the data
    predictions_file = "data/raw/all_predictions.json"
    metadata_file = "data/raw/sports_challenge_metadata.json"

    challenge_loader = GJOChallengeLoader(predictions_file, metadata_file, challenge_title="GJO Challenge")
    challenge = challenge_loader.load_challenge(forecaster_filter=20, problem_filter=20)

    irt_model = IRTModel(n_bins=6, use_empirical_quantiles=False)
    # mcmc_config = MCMCConfig(total_samples=200, warmup_steps=20, num_workers=1, device="cpu", save_result=True)
    svi_config = SVIConfig(optimizer="Adam", num_steps=5000, learning_rate=0.005, device="cpu")
    fitted_scores, rankings = irt_model.fit(challenge.forecast_problems, method="SVI", config=svi_config)

    # print the results
    for forecaster, score in fitted_scores.items(): # type: ignore
        print(f"  {forecaster}: score={score}, rank={rankings[forecaster]}") # type: ignore

    # get the problem level parameters
    problem_difficulties, problem_discriminations = irt_model.get_problem_level_parameters()
    print(f"Number of problems: {len(problem_difficulties)}")
    print(problem_difficulties[:10])
    print(problem_discriminations[:10])
    
