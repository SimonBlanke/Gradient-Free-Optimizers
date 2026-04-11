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
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

    CMA-ES adapts a multivariate normal distribution to the fitness
    landscape by learning the full covariance structure. It samples
    lambda candidate solutions per generation, evaluates them, selects
    the mu best, and updates the distribution parameters (mean, covariance
    matrix, step size) using evolution paths.

    All dimensions are internally normalized to [0, 1] so the initial
    covariance matrix (identity) treats all dimensions equally. For
    discrete and categorical dimensions, sampled float values are rounded
    to the nearest valid index by the framework's clipping layer.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to search dimension definitions.
    initialize : dict, optional
        Strategy for generating initial positions.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random restart.
    nth_process : int, optional
        Process index for parallel optimization.
    population : int or None, default=None
        Number of samples per generation (lambda in CMA-ES notation).
        If None, uses the standard heuristic: 4 + floor(3 * ln(n)).
    mu : int or None, default=None
        Number of parents selected from each generation.
        If None, uses population // 2.
    sigma : float, default=0.3
        Initial step size as a fraction of the normalized search space.
        Controls how far from the mean new samples are drawn initially.
    ipop_restart : bool, default=False
        Enable IPOP (Increasing Population) restart strategy.
        When stagnation is detected, doubles the population size and
        restarts from a random position while keeping the best solution.
    """

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
        conditions: list | None = None,
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
            conditions=conditions,
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

    def _setup_iteration(self):
        """Denormalize the next pre-sampled position for batch extraction.

        Called lazily by the first _iterate_*_batch() method.
        """
        if self._iteration_setup_done:
            return

        sample = self._generation_samples[self._sample_idx]
        self._current_denorm_pos = self._denormalize(sample)
        self._iteration_setup_done = True

    def _iterate_continuous_batch(self) -> np.ndarray:
        """Return continuous portion of the current CMA-ES sample."""
        self._setup_iteration()
        return self._current_denorm_pos[self._continuous_mask]

    def _iterate_categorical_batch(self) -> np.ndarray:
        """Return categorical portion of the current CMA-ES sample."""
        self._setup_iteration()
        return self._current_denorm_pos[self._categorical_mask]

    def _iterate_discrete_batch(self) -> np.ndarray:
        """Return discrete portion of the current CMA-ES sample."""
        self._setup_iteration()
        return self._current_denorm_pos[self._discrete_mask]

    def _on_evaluate(self, score_new: float) -> None:
        """Collect score and trigger CMA update when generation is complete.

        After lambda evaluations, performs the full CMA-ES parameter
        update (mean, evolution paths, covariance matrix, step size)
        and samples the next generation.

        Parameters
        ----------
        score_new : float
            Score of the most recently evaluated position.
        """
        self._generation_scores.append(score_new)
        self._sample_idx += 1

        self._update_best(self._pos_new, score_new)
        self._update_current(self._pos_new, score_new)

        self._iteration_setup_done = False
        self._current_denorm_pos = None

        if self._sample_idx >= self._lambda:
            self._update_cma()
            self._generation_count += 1

            if self.ipop_restart:
                self._check_restart()

            self._sample_generation()

    def _update_cma(self):
        """Full CMA-ES parameter update after one generation.

        Steps:
        1. Rank samples by score (descending, since we maximize)
        2. Compute weighted mean from mu best samples
        3. Update cumulation paths (p_sigma, p_c)
        4. Update covariance matrix C (rank-one + rank-mu)
        5. Update step size sigma via CSA
        6. Re-decompose C for next generation's sampling
        """
        n = self._n
        old_mean = self._mean.copy()

        # 1. Rank by score (descending)
        indices = np.argsort(self._generation_scores)[::-1]

        # 2. Weighted mean of mu best samples
        self._mean = np.zeros(n)
        for i in range(self._mu):
            self._mean += self._weights[i] * self._generation_samples[indices[i]]

        # Weighted step in normalized space
        y_w = (self._mean - old_mean) / self._cma_sigma

        # 3a. Evolution path for step-size control (CSA)
        self._p_sigma = (1 - self._c_sigma) * self._p_sigma + np.sqrt(
            self._c_sigma * (2 - self._c_sigma) * self._mu_eff
        ) * (self._invsqrtC @ y_w)

        # h_sigma: stall indicator for p_c update
        p_sigma_norm = np.linalg.norm(self._p_sigma)
        gen_factor = 1 - (1 - self._c_sigma) ** (2 * (self._generation_count + 1))
        threshold = (1.4 + 2 / (n + 1)) * self._chi_n * np.sqrt(gen_factor)
        h_sigma = 1.0 if p_sigma_norm < threshold else 0.0

        # 3b. Evolution path for rank-one update
        self._p_c = (1 - self._c_c) * self._p_c + h_sigma * np.sqrt(
            self._c_c * (2 - self._c_c) * self._mu_eff
        ) * y_w

        # 4. Covariance matrix update
        delta_h = (1 - h_sigma) * self._c_c * (2 - self._c_c)

        rank_one = np.outer(self._p_c, self._p_c)

        rank_mu = np.zeros((n, n))
        for i in range(self._mu):
            y_i = (self._generation_samples[indices[i]] - old_mean) / self._cma_sigma
            rank_mu += self._weights[i] * np.outer(y_i, y_i)

        self._C = (
            (1 + self._c_1 * delta_h - self._c_1 - self._c_mu_cov) * self._C
            + self._c_1 * rank_one
            + self._c_mu_cov * rank_mu
        )

        # 5. Step-size update
        self._cma_sigma *= np.exp(
            (self._c_sigma / self._d_sigma) * (p_sigma_norm / self._chi_n - 1)
        )
        self._cma_sigma = np.clip(self._cma_sigma, 1e-20, 10.0)

        # 6. Eigendecomposition for next generation
        self._eigendecomposition()

    def _check_restart(self):
        """Check for stagnation and trigger IPOP restart if needed.

        Doubles the population size on restart while resetting the
        distribution to a random mean with identity covariance.
        """
        if self._score_best > self._best_score_at_restart:
            self._gens_without_improvement = 0
            self._best_score_at_restart = self._score_best
        else:
            self._gens_without_improvement += 1

        stag_threshold = 10 + int(30 * self._n / max(self._lambda, 1))

        if self._gens_without_improvement >= stag_threshold:
            self._lambda = min(self._lambda * 2, 2048)
            self._mu = max(1, self._lambda // 2)
            self._compute_strategy_params()

            self._mean = self._rng.uniform(0, 1, self._n)
            self._cma_sigma = self._initial_sigma
            self._C = np.eye(self._n)
            self._p_sigma = np.zeros(self._n)
            self._p_c = np.zeros(self._n)
            self._eigendecomposition()

            self._gens_without_improvement = 0
            self._generation_count = 0

    def _iterate_batch(self, n):
        """Generate n positions from the current CMA-ES generation.

        Only returns actual CMA-ES samples for positions that fit within
        the current generation. Overflow positions are filled with random
        samples to avoid contaminating the next generation's CMA update
        with scores from unrelated positions.
        """
        remaining_in_generation = self._lambda - self._sample_idx
        self._batch_n_from_generation = min(n, remaining_in_generation)

        positions = []
        for i in range(self._batch_n_from_generation):
            sample = self._generation_samples[self._sample_idx + i]
            pos = self._denormalize(sample)
            positions.append(self._clip_position(pos))

        for _ in range(n - self._batch_n_from_generation):
            positions.append(self._clip_position(self.init.move_random_typed()))

        return positions

    def _evaluate_batch(self, positions, scores):
        """Process batch results, keeping CMA-ES samples separate from overflow.

        CMA-ES updates (mean, covariance, step size) depend on scores
        matching their corresponding generation samples exactly. Overflow
        positions (random fill beyond the generation boundary) must not
        enter _on_evaluate, as that would insert foreign scores into
        _generation_scores and corrupt the next CMA update.
        """
        n_cma = self._batch_n_from_generation

        for pos, score in zip(positions[:n_cma], scores[:n_cma]):
            self._pos_new = pos
            self._evaluate(score)

        for pos, score in zip(positions[n_cma:], scores[n_cma:]):
            self._pos_new = pos
            self._track_score(score)
            self._update_best(pos, score)
