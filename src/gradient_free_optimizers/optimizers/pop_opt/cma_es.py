# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CMA-ES (Covariance Matrix Adaptation Evolution Strategy) Optimizer."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from .base_population_optimizer import BasePopulationOptimizer

if TYPE_CHECKING:
    pass


class CMAESOptimizer(BasePopulationOptimizer):
    name = "CMA-ES"
    _name_ = "cma_es"
    __name__ = "CMAESOptimizer"

    optimizer_type = "population"
    computationally_expensive = False

    def __init__(
        self,
        search_space: dict[str, Any],
        initialize: dict[str, int] | None = None,
        constraints: list[Callable[[dict[str, Any]], bool]] | None = None,
        random_state: int | None = None,
        rand_rest_p: float = 0,
        nth_process: int | None = None,
        population: int | None = None,
        mu: int | None = None,
        sigma: float = 0.3,
        ipop_restart: bool = False,
    ) -> None:
        n = len(search_space)

        if population is None:
            population = 4 + int(3 * np.log(n)) if n > 0 else 4

        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            population=population,
        )

        self._lambda = population
        self._mu = mu if mu is not None else max(1, self._lambda // 2)
        self._initial_sigma = sigma
        self.ipop_restart = ipop_restart

        # CMA-ES does not use sub-optimizers but needs enough init
        # positions to match population size (consistent with other pop opts)
        self.optimizers = [self]
        diff_init = self._lambda - self.init.n_inits
        if diff_init > 0:
            self.init.add_n_random_init_pos(diff_init)

        self._rng = np.random.default_rng(self.random_seed)
        self._n = n

        # Normalization arrays (maps positions to [0, 1]^n)
        self._setup_normalization()

        # Strategy parameters (weights, learning rates)
        self._compute_strategy_params()

        # CMA-ES state (mean initialized in _on_finish_initialization)
        self._mean = np.full(n, 0.5)
        self._cma_sigma = sigma
        self._C = np.eye(n)
        self._p_sigma = np.zeros(n)
        self._p_c = np.zeros(n)
        self._B = np.eye(n)
        self._D = np.ones(n)
        self._invsqrtC = np.eye(n)

        # Generation state
        self._generation_samples = []
        self._generation_scores = []
        self._sample_idx = 0
        self._generation_count = 0

        # Template method coordination
        self._iteration_setup_done = False
        self._current_denorm_pos = None

        # IPOP restart state
        self._initial_lambda = self._lambda
        self._best_score_at_restart = -np.inf
        self._gens_without_improvement = 0

    def _setup_normalization(self):
        """Set up arrays for normalizing positions to [0, 1]^n.

        This makes CMA-ES dimension-agnostic: the covariance matrix
        starts as identity, treating all dimensions equally regardless
        of their original scale.
        """
        n = self._n
        self._dim_scales = np.ones(n)
        self._dim_offsets = np.zeros(n)

        for i, (name, dim_def) in enumerate(self.search_space.items()):
            if isinstance(dim_def, tuple) and len(dim_def) == 2:
                self._dim_offsets[i] = dim_def[0]
                self._dim_scales[i] = dim_def[1] - dim_def[0]
            elif isinstance(dim_def, list):
                self._dim_offsets[i] = 0
                self._dim_scales[i] = max(len(dim_def) - 1, 1)
            elif isinstance(dim_def, np.ndarray):
                self._dim_offsets[i] = 0
                self._dim_scales[i] = max(len(dim_def) - 1, 1)

        self._dim_scales[self._dim_scales == 0] = 1.0

    def _normalize(self, pos):
        """Map position from original space to [0, 1]^n."""
        return (pos - self._dim_offsets) / self._dim_scales

    def _denormalize(self, norm_pos):
        """Map position from [0, 1]^n back to original space."""
        return norm_pos * self._dim_scales + self._dim_offsets

    def _compute_strategy_params(self):
        """Compute CMA-ES strategy parameters from n, lambda, mu.

        These follow Hansen & Ostermeier (2001) with the standard
        default parameter settings from the CMA-ES tutorial.
        """
        n = self._n
        mu = self._mu

        if n == 0 or mu == 0:
            return

        # Recombination weights (log-proportional, normalized)
        raw_w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self._weights = raw_w / raw_w.sum()
        self._mu_eff = 1.0 / np.sum(self._weights**2)

        mu_eff = self._mu_eff

        # Step-size control (CSA)
        self._c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        self._d_sigma = (
            1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + self._c_sigma
        )

        # Covariance matrix adaptation
        self._c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        self._c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        self._c_mu_cov = min(
            1 - self._c_1,
            2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff),
        )

        # Expected length of N(0, I) vector
        self._chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

    def _eigendecomposition(self):
        """Decompose C = B * D^2 * B^T for sampling and invsqrtC.

        Also enforces symmetry and positive-definiteness, and resets
        C if the condition number becomes too large.
        """
        self._C = np.triu(self._C) + np.triu(self._C, 1).T

        D_sq, B = np.linalg.eigh(self._C)
        D_sq = np.maximum(D_sq, 1e-20)

        # Reset if condition number is dangerously large
        if D_sq.max() / D_sq.min() > 1e14:
            self._C = np.eye(self._n)
            D_sq = np.ones(self._n)
            B = np.eye(self._n)

        self._D = np.sqrt(D_sq)
        self._B = B
        self._invsqrtC = B @ np.diag(1.0 / self._D) @ B.T

    def _sample_generation(self):
        """Sample lambda candidates from N(mean, sigma^2 * C)."""
        self._generation_samples = []
        self._generation_scores = []
        self._sample_idx = 0

        for _ in range(self._lambda):
            z = self._rng.standard_normal(self._n)
            sample = self._mean + self._cma_sigma * (self._B @ (self._D * z))
            self._generation_samples.append(sample)

    def _on_finish_initialization(self):
        """Initialize CMA-ES mean from best init position, then sample."""
        if self._pos_best is not None:
            self._mean = self._normalize(self._pos_best.copy())
        else:
            self._mean = np.full(self._n, 0.5)

        self._cma_sigma = self._initial_sigma
        self._C = np.eye(self._n)
        self._p_sigma = np.zeros(self._n)
        self._p_c = np.zeros(self._n)
        self._eigendecomposition()
        self._sample_generation()
