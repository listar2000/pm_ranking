from functools import partial
import torch
import numpy as np
import logging
from tqdm.contrib.logging import logging_redirect_tqdm

from pyro.infer import SVI, Trace_ELBO
from pyro.infer.mcmc import NUTS, MCMC, HMC
from pyro.optim import Adam, SGD  # type: ignore
import pyro
import pyro.distributions as dist
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any, Tuple

from pm_rank.model.utils import forecaster_data_to_rankings, get_logger, log_ranking_table
from pm_rank.model.irt._dataset import _prepare_pyro_obs
from pm_rank.data.base import ForecastProblem
from pm_rank.data.loaders import GJOChallengeLoader

OUTPUT_DIR = __file__.replace(__file__.split(
    "/")[-1], "output")  # the output directory

"""
Configurations for running MCMC and the SVI inference engines.
"""


class MCMCConfig(BaseModel):
    """Configuration for MCMC (Markov Chain Monte Carlo) inference using NUTS sampler.

    This configuration class defines parameters for running Hamiltonian Monte Carlo
    sampling with the No-U-Turn Sampler (NUTS) algorithm, which provides exact
    posterior inference for the IRT model.

    :param total_samples: The total number of samples to draw from the posterior distribution (default: 1000).
    :param warmup_steps: The number of warmup steps to run before sampling (default: 100).
    :param num_workers: The number of workers to use for parallelization. Note that we use a customized
                        multiprocessing approach since the default implementation by Pyro can be very slow.
                        This is why we don't use the name `num_chains` (default: 1).
    :param device: The device to use for the MCMC engine ("cpu" or "cuda") (default: "cpu").
    :param save_result: Whether to save the result to a file (default: False).
    """
    total_samples: int = Field(
        default=1000, description="The total number of samples to draw from the posterior distribution.")
    warmup_steps: int = Field(
        default=100, description="The number of warmup steps to run before sampling.")
    num_workers: int = Field(default=1, description="The number of workers to use for parallelization. Note that we use a customized multiprocessing approach \
        since the default implementation by Pyro can be very slow. This is why we don't use the name `num_chains`.")
    device: Literal["cpu", "cuda"] = Field(
        default="cpu", description="The device to use for the MCMC engine.")
    save_result: bool = Field(
        default=False, description="Whether to save the result to a file.")


class SVIConfig(BaseModel):
    """Configuration for SVI (Stochastic Variational Inference) optimization.

    This configuration class defines parameters for running variational inference
    using stochastic gradient descent, which provides fast approximate posterior
    inference for the IRT model.

    :param optimizer: The optimizer to use for the SVI engine ("Adam" or "SGD") (default: "Adam").
    :param num_steps: The number of steps to run for the SVI engine (default: 1000).
    :param learning_rate: The learning rate to use for the SVI engine (default: 0.01).
    :param device: The device to use for the SVI engine ("cpu" or "cuda") (default: "cpu").
    """
    optimizer: Literal["Adam", "SGD"] = Field(
        default="Adam", description="The optimizer to use for the SVI engine.")
    num_steps: int = Field(
        default=1000, description="The number of steps to run for the SVI engine.")
    learning_rate: float = Field(
        default=0.01, description="The learning rate to use for the SVI engine.")
    device: Literal["cpu", "cuda"] = Field(
        default="cpu", description="The device to use for the SVI engine.")


class IRTModel(object):
    """Item Response Theory model for ranking forecasters using Pyro.

    This class implements an IRT model that estimates latent abilities of forecasters
    and difficulty/discrimination parameters of prediction problems. The model uses
    discretized scoring bins and supports both SVI and MCMC inference methods.

    The IRT model assumes that the probability of a forecaster achieving a certain
    score on a problem depends on their latent ability (θ), the problem's difficulty (b),
    the problem's discrimination (a), and category parameters (p) for the scoring bins.

    :param n_bins: Number of bins for discretizing continuous scores (default: 6).
    :param use_empirical_quantiles: Whether to use empirical quantiles for binning
                                   instead of uniform bins (default: False).
    :param verbose: Whether to enable verbose logging (default: False).
    """

    def __init__(self, n_bins: int = 6, use_empirical_quantiles: bool = False, verbose: bool = False):
        """Initialize the IRT model.

        :param n_bins: Number of bins for discretizing continuous scores (default: 6).
        :param use_empirical_quantiles: Whether to use empirical quantiles for binning
                                       instead of uniform bins (default: False).
        :param verbose: Whether to enable verbose logging (default: False).
        """
        self.n_bins = n_bins
        self.use_empirical_quantiles = use_empirical_quantiles
        # initiate pyro observations with None
        self.irt_obs = None
        self.method = None
        self.verbose = verbose
        self.logger = get_logger(f"pm_rank.model.{self.__class__.__name__}")
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)
        self.logger.info(f"Initialized {self.__class__.__name__} with hyperparam: \n" +
                         f"n_bins={n_bins}, use_empirical_quantiles={use_empirical_quantiles}")

    def fit(self, problems: List[ForecastProblem], include_scores: bool = True, method: Literal["SVI", "NUTS"] = "SVI",
            config: MCMCConfig | SVIConfig | None = None) -> Tuple[Dict[str, Any], Dict[str, int]] | Dict[str, int]:
        """Fit the IRT model to the given problems and return rankings.

        This method fits the IRT model using either SVI or MCMC inference, depending on
        the specified method. The model estimates latent abilities for each forecaster
        and difficulty/discrimination parameters for each problem.

        :param problems: List of ForecastProblem instances to fit the model to.
        :param include_scores: Whether to include scores in the results (default: True).
        :param method: Inference method to use ("SVI" for fast approximate inference
                       or "NUTS" for exact MCMC inference) (default: "SVI").
        :param config: Configuration object for the chosen inference method.
                       Must be MCMCConfig for "NUTS" or SVIConfig for "SVI".

        :returns: If include_scores=True, returns a tuple of (scores_dict, rankings_dict).
                  If include_scores=False, returns only rankings_dict.
                  scores_dict maps forecaster IDs to their estimated abilities.
                  rankings_dict maps forecaster IDs to their ranks (1-based).

        :raises: AssertionError if method is invalid or config is not provided.
        """

        assert method in [
            "SVI", "NUTS"], "Invalid method. Must be either 'SVI' or 'NUTS'."
        assert config is not None, "Configuration must be provided."

        self.method = method
        if self.method == "SVI":
            assert isinstance(
                config, SVIConfig), "SVI configuration must be provided."
        elif self.method == "NUTS":
            assert isinstance(
                config, MCMCConfig), "MCMC configuration must be provided."

        self.device = config.device
        self.irt_obs = _prepare_pyro_obs(
            problems, self.n_bins, self.use_empirical_quantiles, self.device)  # type: ignore

        if self.method == "NUTS":
            mcmc_config: MCMCConfig = config  # type: ignore
            posterior_samples = self._fit_pyro_model_mcmc(self.irt_obs.forecaster_ids, self.irt_obs.problem_ids, self.irt_obs.discretized_scores, self.irt_obs.anchor_points,
                                                          num_samples=mcmc_config.total_samples, warmup_steps=mcmc_config.warmup_steps, num_chains=mcmc_config.num_workers)

            self.posterior_samples = posterior_samples

            if mcmc_config.save_result:
                import time
                torch.save(
                    posterior_samples, f"{OUTPUT_DIR}/posterior_samples_{time.strftime('%m%d_%H%M')}.pt")

            result = self._score_and_rank_mcmc(
                self.posterior_samples, include_scores=include_scores)
        else:
            svi_config: SVIConfig = config  # type: ignore
            fitted_params = self._fit_pyro_model_svi(self.irt_obs.forecaster_ids, self.irt_obs.problem_ids, self.irt_obs.discretized_scores, self.irt_obs.anchor_points,
                                                     optimizer=svi_config.optimizer, num_steps=svi_config.num_steps, learning_rate=svi_config.learning_rate)

            self.fitted_params = fitted_params

            result = self._score_and_rank_svi(
                self.fitted_params, include_scores=include_scores)

        if self.verbose:
            log_ranking_table(self.logger, result)
        return result

    def get_problem_level_parameters(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Get problem difficulty and discrimination parameters.

        Returns the estimated difficulty and discrimination parameters for each problem
        after the model has been fitted. These parameters provide insights into how
        challenging each problem is and how well it distinguishes between forecasters
        of different abilities.

        :returns: A tuple of (difficulties_dict, discriminations_dict).
                  difficulties_dict maps problem IDs to their difficulty parameters (b).
                  discriminations_dict maps problem IDs to their discrimination parameters (a).

        :raises: AssertionError if the model has not been fitted yet.
        """
        # make sure the model has been fit
        assert self.method is not None, "IRT model must be fit before getting problem attributes"

        if self.method == "SVI":
            raw_problem_difficulties = self.fitted_params["svi_mean_b"]
            raw_problem_discriminations = self.fitted_params["svi_mean_a"]

        elif self.method == "NUTS":
            raw_problem_difficulties = self.posterior_samples["b"].mean(dim=0)
            raw_problem_discriminations = self.posterior_samples["a"].mean(
                dim=0)

        problem_diff_dict, problem_discrim_dict = {}, {}
        for problem_id, idx in self.irt_obs.problem_id_to_idx.items():
            problem_diff_dict[problem_id] = raw_problem_difficulties[idx]
            problem_discrim_dict[problem_id] = raw_problem_discriminations[idx]
        return problem_diff_dict, problem_discrim_dict

    def _model(self, forecaster_ids: torch.Tensor, problem_ids: torch.Tensor, discretized_scores: torch.Tensor, anchor_points: torch.Tensor):
        """Define the IRT model using Pyro.

        This method defines the probabilistic model for Item Response Theory using Pyro.
        The model includes:
        - Forecaster ability parameters (θ) ~ Normal(0, 1)
        - Problem discrimination parameters (a) ~ HalfNormal(5)
        - Problem difficulty parameters (b) ~ Normal(0, 5)
        - Category parameters (p) ~ Normal(0, 5)

        The likelihood function uses a categorical distribution with logits computed
        as: a_j * (1 - anchor_points) * (θ_i - (b_j + p))

        :param forecaster_ids: Tensor of forecaster indices for each observation.
        :param problem_ids: Tensor of problem indices for each observation.
        :param discretized_scores: Tensor of discretized scores for each observation.
        :param anchor_points: Tensor of anchor points defining the scoring bins.
        """
        # Infer N forecasters, M problems, and K anchor points from data
        N = int(forecaster_ids.max()) + 1
        M = int(problem_ids.max()) + 1
        K = len(anchor_points)

        # Define the forecaster-level ability parameters - `theta`
        with pyro.plate("forecasters", N, device=self.device):
            mean_theta, std_theta = torch.tensor(
                0.0, device=self.device), torch.tensor(1.0, device=self.device)
            theta = pyro.sample("theta", dist.Normal(mean_theta, std_theta))

        # Define the problem-level difficulty parameters - `a` for discrimination and `b` for difficulty
        with pyro.plate("problems", M, device=self.device):
            std_a = torch.tensor(5.0, device=self.device)
            a = pyro.sample("a", dist.HalfNormal(std_a))
            mean_b, std_b = torch.tensor(
                0.0, device=self.device), torch.tensor(5.0, device=self.device)
            b = pyro.sample("b", dist.Normal(mean_b, std_b))

        # Define the category-level parameter - `p`
        with pyro.plate("categories", K, device=self.device):
            mean_p, std_p = torch.tensor(
                0.0, device=self.device), torch.tensor(5.0, device=self.device)
            p = pyro.sample("p", dist.Normal(mean_p, std_p))

        # --- Likelihood ---
        num_obs = len(forecaster_ids)

        with pyro.plate("data", num_obs, device=self.device):
            # get the forecaster and problem ids
            theta_i = theta[forecaster_ids]  # shape: (num_obs,)
            a_j = a[problem_ids]  # shape: (num_obs,)
            b_j = b[problem_ids]  # shape: (num_obs,)

            # We use broadcasting to achieve this efficiently.
            # Shapes:
            # theta_i.unsqueeze(1) -> [num_observations, 1]
            # a_j.unsqueeze(1)     -> [num_observations, 1]
            # b_j.unsqueeze(1)     -> [num_observations, 1]
            # anchor_points        -> [K]
            # p                    -> [K]
            logits = (a_j.unsqueeze(1) * (1. - anchor_points) *
                      (theta_i.unsqueeze(1) - (b_j.unsqueeze(1) + p)))  # shape: (num_obs, K)

            # Now, we can sample from the Categorical distribution.
            pyro.sample("obs", dist.Categorical(
                logits=logits), obs=discretized_scores)

    def _guide(self, forecaster_ids: torch.Tensor, problem_ids: torch.Tensor, discretized_scores: torch.Tensor, bin_edges: torch.Tensor):
        """Define the variational guide for SVI inference.

        This method defines the variational approximation used in Stochastic Variational
        Inference (SVI). The guide approximates the posterior distribution of the model
        parameters using mean-field variational families:
        - θ ~ Normal(mean_theta, std_theta)
        - a ~ HalfNormal(std_a)
        - b ~ Normal(mean_b, std_b)
        - p ~ Normal(mean_p, std_p)

        :param forecaster_ids: Tensor of forecaster indices for each observation.
        :param problem_ids: Tensor of problem indices for each observation.
        :param discretized_scores: Tensor of discretized scores for each observation.
        :param bin_edges: Tensor of bin edges (unused parameter, kept for interface consistency).
        """
        # Infer N forecasters, M problems, and K anchor points from data
        N = int(forecaster_ids.max()) + 1
        M = int(problem_ids.max()) + 1
        K = len(bin_edges)
        # set up all the parameters (in a mean-field way)
        mean_theta_param = pyro.param(
            "mean_theta", torch.zeros(N, device=self.device))
        std_theta_param = pyro.param("std_theta", torch.ones(
            N, device=self.device), constraint=dist.constraints.positive)

        std_a_param = pyro.param("std_a", torch.empty(M, device=self.device).fill_(
            5.0), constraint=dist.constraints.positive)

        mean_b_param = pyro.param("mean_b", torch.zeros(M, device=self.device))
        std_b_param = pyro.param("std_b", torch.empty(M, device=self.device).fill_(
            5.0), constraint=dist.constraints.positive)

        mean_p_param = pyro.param("mean_p", torch.zeros(K, device=self.device))
        std_p_param = pyro.param("std_p", torch.empty(K, device=self.device).fill_(
            5.0), constraint=dist.constraints.positive)

        with pyro.plate("forecasters", N, device=self.device):
            theta = pyro.sample("theta", dist.Normal(
                mean_theta_param, std_theta_param))

        with pyro.plate("problems", M, device=self.device):
            a = pyro.sample("a", dist.HalfNormal(std_a_param))
            b = pyro.sample("b", dist.Normal(mean_b_param, std_b_param))

        with pyro.plate("categories", K, device=self.device):
            p = pyro.sample("p", dist.Normal(mean_p_param, std_p_param))

    def _fit_pyro_model_mcmc(self, forecaster_ids: torch.Tensor, problem_ids: torch.Tensor, discretized_scores: torch.Tensor, anchor_points: torch.Tensor,
                             num_samples: int = 1000, warmup_steps: int = 100, num_chains: int = 1):
        """Fit the IRT model using MCMC with NUTS sampler.

        This method performs exact posterior inference using Hamiltonian Monte Carlo
        with the No-U-Turn Sampler (NUTS). It draws samples from the posterior
        distribution of all model parameters.

        :param forecaster_ids: Tensor of forecaster indices for each observation.
        :param problem_ids: Tensor of problem indices for each observation.
        :param discretized_scores: Tensor of discretized scores for each observation.
        :param anchor_points: Tensor of anchor points defining the scoring bins.
        :param num_samples: Number of posterior samples to draw (default: 1000).
        :param warmup_steps: Number of warmup steps for the MCMC sampler (default: 100).
        :param num_chains: Number of parallel chains to run (default: 1).

        :returns: Dictionary containing posterior samples for all model parameters.
                  Keys include 'theta', 'a', 'b', 'p' with corresponding sample tensors.

        :raises: AssertionError if IRT observations have not been prepared.
        """
        pyro.clear_param_store()  # make sure the param store is empty
        assert self.irt_obs is not None, "IRT observations must be prepared before fitting the model"

        nuts_kernel = NUTS(self._model, adapt_step_size=True)
        mp_context = "spawn" if num_chains > 1 and self.device == "cuda" else None
        mcmc = MCMC(nuts_kernel, num_samples=num_samples,
                    warmup_steps=warmup_steps, num_chains=num_chains, mp_context=mp_context)

        if self.verbose:
            with logging_redirect_tqdm([self.logger]):
                mcmc.run(
                    forecaster_ids=forecaster_ids,
                    problem_ids=problem_ids,
                    discretized_scores=discretized_scores,
                    anchor_points=anchor_points,
                )
        else:
            mcmc.run(
                forecaster_ids=forecaster_ids,
                problem_ids=problem_ids,
                discretized_scores=discretized_scores,
                anchor_points=anchor_points,
                disable_progbar=True
            )

        posterior_samples = mcmc.get_samples()

        return posterior_samples

    def _fit_pyro_model_svi(self, forecaster_ids: torch.Tensor, problem_ids: torch.Tensor, discretized_scores: torch.Tensor, anchor_points: torch.Tensor,
                            optimizer: Literal["Adam", "SGD"] = "Adam", num_steps: int = 1000, learning_rate: float = 0.01):
        """Fit the IRT model using SVI (Stochastic Variational Inference).

        This method performs approximate posterior inference using variational methods.
        It optimizes the variational parameters to approximate the true posterior
        distribution of the model parameters.

        :param forecaster_ids: Tensor of forecaster indices for each observation.
        :param problem_ids: Tensor of problem indices for each observation.
        :param discretized_scores: Tensor of discretized scores for each observation.
        :param anchor_points: Tensor of anchor points defining the scoring bins.
        :param optimizer: Optimizer to use ("Adam" or "SGD") (default: "Adam").
        :param num_steps: Number of optimization steps (default: 1000).
        :param learning_rate: Learning rate for the optimizer (default: 0.01).

        :returns: Dictionary containing the fitted variational parameters.
                  Keys include 'svi_mean_thetas', 'svi_mean_a', 'svi_mean_b', 'svi_mean_p'.

        :raises: AssertionError if IRT observations have not been prepared or optimizer is invalid.
        """
        from tqdm import tqdm

        pyro.clear_param_store()  # make sure the param store is empty
        assert self.irt_obs is not None, "IRT observations must be prepared before fitting the model"

        assert optimizer in [
            "Adam", "SGD"], "Invalid optimizer. Must be either 'Adam' or 'SGD'."
        optim = Adam({"lr": learning_rate}) if optimizer == "Adam" else SGD(
            {"lr": learning_rate})

        svi = SVI(self._model, self._guide, optim, loss=Trace_ELBO())
        if self.verbose:
            with logging_redirect_tqdm([self.logger]):
                pbar = tqdm(range(num_steps))
                for i in pbar:
                    loss = svi.step(forecaster_ids, problem_ids,
                                    discretized_scores, anchor_points)
                    if i % 20 == 0:
                        pbar.set_description(f"SVI [Loss: {loss:5.1f}]")
        else:
            for i in range(num_steps):
                svi.step(forecaster_ids, problem_ids,
                         discretized_scores, anchor_points)

        return {
            "svi_mean_thetas": pyro.param("mean_theta").detach().cpu().numpy(),
            # special case for the half-normal distribution
            "svi_mean_a": pyro.param("std_a").detach().cpu().numpy() * np.sqrt(2 / np.pi),
            "svi_mean_b": pyro.param("mean_b").detach().cpu().numpy(),
            "svi_mean_p": pyro.param("mean_p").detach().cpu().numpy(),
        }

    def _score_and_rank_helper(self, theta_means, include_scores: bool = True):
        """Helper method to convert theta means to forecaster rankings.

        This method maps the estimated ability parameters (theta means) to forecaster IDs
        and computes rankings based on these abilities.

        :param theta_means: Array or tensor of theta means for each forecaster.
        :param include_scores: Whether to include scores in the results (default: True).

        :returns: If include_scores=True, returns a tuple of (scores_dict, rankings_dict).
                  If include_scores=False, returns only rankings_dict.
                  scores_dict maps forecaster IDs to their estimated abilities.
                  rankings_dict maps forecaster IDs to their ranks (1-based).

        :raises: AssertionError if IRT observations have not been prepared.
        """
        assert self.irt_obs is not None, "IRT observations must be prepared before scoring and ranking forecasters"
        forecaster_idx_to_id = self.irt_obs.forecaster_idx_to_id
        forecaster_data = {}
        for i in range(len(theta_means)):
            forecaster_id = forecaster_idx_to_id[i]
            # theta_means may be a numpy array or torch tensor; .item() works for both
            forecaster_data[forecaster_id] = theta_means[i].item() if hasattr(
                theta_means[i], 'item') else float(theta_means[i])
        return forecaster_data_to_rankings(forecaster_data, include_scores=include_scores, ascending=False, aggregate="mean")

    def _score_and_rank_mcmc(self, posterior_samples, include_scores: bool = True):
        """Convert MCMC posterior samples to forecaster rankings.

        This method takes posterior samples from MCMC inference and computes
        forecaster rankings based on the posterior mean of theta parameters.

        :param posterior_samples: Dictionary containing posterior samples from MCMC.
        :param include_scores: Whether to include scores in the results (default: True).

        :returns: If include_scores=True, returns a tuple of (scores_dict, rankings_dict).
                  If include_scores=False, returns only rankings_dict.
                  scores_dict maps forecaster IDs to their estimated abilities.
                  rankings_dict maps forecaster IDs to their ranks (1-based).
        """
        theta_means = posterior_samples["theta"].mean(dim=0)
        return self._score_and_rank_helper(theta_means, include_scores=include_scores)

    def _score_and_rank_svi(self, fitted_params: Dict[str, Any], include_scores: bool = True):
        """Convert SVI fitted parameters to forecaster rankings.

        This method takes fitted parameters from SVI inference and computes
        forecaster rankings based on the variational means of theta parameters.

        :param fitted_params: Dictionary containing fitted variational parameters from SVI.
        :param include_scores: Whether to include scores in the results (default: True).

        :returns: If include_scores=True, returns a tuple of (scores_dict, rankings_dict).
                  If include_scores=False, returns only rankings_dict.
                  scores_dict maps forecaster IDs to their estimated abilities.
                  rankings_dict maps forecaster IDs to their ranks (1-based).
        """
        theta_means = fitted_params["svi_mean_thetas"]
        return self._score_and_rank_helper(theta_means, include_scores=include_scores)

    def _summary(self, traces, sites):
        """Aggregate marginals for MCMC samples.

        This method computes summary statistics for MCMC posterior samples,
        including mean, standard deviation, and percentiles for each parameter.

        :param traces: Dictionary of posterior samples from MCMC.
        :param sites: List of site names to summarize.

        :returns: Dictionary containing summary statistics for each parameter.
                  Each parameter gets statistics including mean, std, 5%, 25%, 50%, 75%, 95%.
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
                    describe = partial(pd.Series.describe, percentiles=[
                                       0.05, 0.25, 0.5, 0.75, 0.95])
                    site_stats[param_name] = marginal_site.apply(describe, axis=1)[
                        ["mean", "std", "5%", "25%", "50%", "75%", "95%"]
                    ]
        return site_stats
