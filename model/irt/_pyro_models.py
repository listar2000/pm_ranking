from functools import partial
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.mcmc import NUTS, MCMC
from pyro.optim import Adam # type: ignore
import pyro
import pyro.distributions as dist
from typing import Literal, List, Dict, Any

from model.irt._dataset import _prepare_pyro_obs, IRTObs
from data.base import ForecastProblem

class IRTModel(object):
    def __init__(self, n_bins: int = 6, use_empirical_quantiles: bool = False, device: Literal["cpu", "cuda"] = "cpu", method: Literal["SVI", "NUTS"] = "SVI"):
        self.n_bins = n_bins
        self.use_empirical_quantiles = use_empirical_quantiles
        self.device = device
        self.method = method
        # initiate pyro observations with None
        self.irt_obs = None

    def fit(self, problems: List[ForecastProblem], include_scores: bool = True) -> Dict[str, Any]:
        """ fit the model to the problems """
        self.irt_obs = _prepare_pyro_obs(problems, self.n_bins, self.use_empirical_quantiles, self.device)  # type: ignore

        self._fit_pyro_model(self.irt_obs.forecaster_ids, self.irt_obs.problem_ids, self.irt_obs.discretized_scores, self.irt_obs.anchor_points)
        return {}

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


    def _fit_pyro_model(self, forecaster_ids: torch.Tensor, problem_ids: torch.Tensor, discretized_scores: torch.Tensor, anchor_points: torch.Tensor):
        """
        The core function that leverages pyro and SVI/NUTS to fit the model.
        """
        pyro.clear_param_store() # make sure the param store is empty
        assert self.irt_obs is not None, "IRT observations must be prepared before fitting the model"

        if self.method == "SVI":
            raise NotImplementedError("SVI is not implemented yet")
        elif self.method == "NUTS":
            nuts_kernel = NUTS(self._model, adapt_step_size=True)
            mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=50)
            
            mcmc.run(
                forecaster_ids=forecaster_ids,
                problem_ids=problem_ids,
                discretized_scores=discretized_scores,
                anchor_points=anchor_points,
            )

            posterior_samples = mcmc.get_samples()

            # get the summary of the posterior
            theta_summary = self._summary(posterior_samples, ["theta"]).items()
            a_summary = self._summary(posterior_samples, ["a"]).items()
            b_summary = self._summary(posterior_samples, ["b"]).items()
            p_summary = self._summary(posterior_samples, ["p"]).items()

            print(theta_summary)
            print(a_summary)
            print(b_summary)
            print(p_summary)
            
            # Plot the posterior samples
            self.plot_posterior_samples(posterior_samples)

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
    
    def plot_posterior_samples(self, posterior_samples: Dict[str, torch.Tensor]):
        """
        Plot histograms of posterior samples for theta, a, b, and p parameters.
        Creates 4 separate PNG files.
        
        Args:
            posterior_samples: Dictionary of posterior samples from MCMC
        """
        base_path = "/net/scratch2/listar2000/pm_ranking/model/irt/images"
        
        # Plot theta samples (first 10)
        if 'theta' in posterior_samples:
            plt.figure(figsize=(10, 6))
            theta_samples = posterior_samples['theta'].detach().cpu().numpy()
            for i in range(min(10, theta_samples.shape[1])):
                plt.hist(theta_samples[:, i], bins=30, alpha=0.7, label=f'θ_{i}')
            plt.title('Theta Parameters (First 10)', fontsize=14, fontweight='bold')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{base_path}/theta_samples.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Theta samples plot saved to: {base_path}/theta_samples.png")
        
        # Plot a samples (first 10)
        if 'a' in posterior_samples:
            plt.figure(figsize=(10, 6))
            a_samples = posterior_samples['a'].detach().cpu().numpy()
            for i in range(min(10, a_samples.shape[1])):
                plt.hist(a_samples[:, i], bins=30, alpha=0.7, label=f'a_{i}')
            plt.title('Discrimination Parameters (First 10)', fontsize=14, fontweight='bold')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{base_path}/a_samples.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Discrimination samples plot saved to: {base_path}/a_samples.png")
        
        # Plot b samples (first 10)
        if 'b' in posterior_samples:
            plt.figure(figsize=(10, 6))
            b_samples = posterior_samples['b'].detach().cpu().numpy()
            for i in range(min(10, b_samples.shape[1])):
                plt.hist(b_samples[:, i], bins=30, alpha=0.7, label=f'b_{i}')
            plt.title('Difficulty Parameters (First 10)', fontsize=14, fontweight='bold')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{base_path}/b_samples.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Difficulty samples plot saved to: {base_path}/b_samples.png")
        
        # Plot p samples (all)
        if 'p' in posterior_samples:
            plt.figure(figsize=(10, 6))
            p_samples = posterior_samples['p'].detach().cpu().numpy()
            for i in range(p_samples.shape[1]):
                plt.hist(p_samples[:, i], bins=30, alpha=0.7, label=f'p_{i}')
            plt.title('Category Parameters (All)', fontsize=14, fontweight='bold')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{base_path}/p_samples.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Category samples plot saved to: {base_path}/p_samples.png")
        

# some in-file tests
if __name__ == "__main__":
    from data.loaders import GJOChallengeLoader

    predictions_file = "data/raw/all_predictions.json"
    metadata_file = "data/raw/sports_challenge_metadata.json"

    # load the data
    challenge_loader = GJOChallengeLoader(predictions_file, metadata_file, challenge_title="GJO Challenge")
    challenge = challenge_loader.load_challenge(forecaster_filter=20, problem_filter=20)

    irt_model = IRTModel(n_bins=6, use_empirical_quantiles=False, device="cpu", method="NUTS")
    irt_model.fit(challenge.forecast_problems)
