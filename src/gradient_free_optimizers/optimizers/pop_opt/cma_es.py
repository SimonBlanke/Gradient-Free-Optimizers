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

