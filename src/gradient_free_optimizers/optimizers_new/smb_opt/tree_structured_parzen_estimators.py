# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Tree-structured Parzen Estimators (TPE).

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np

from .smbo import SMBO

if TYPE_CHECKING:
    import pandas as pd

# Use sklearn's KDE if available, otherwise native implementation
try:
    from sklearn.neighbors import KernelDensity

    SKLEARN_AVAILABLE = True
except ImportError:
    from gradient_free_optimizers._estimators import (
        KernelDensityEstimator as KernelDensity,
    )

    SKLEARN_AVAILABLE = False


class TreeStructuredParzenEstimators(SMBO):
    """Tree-structured Parzen Estimator (TPE) optimization algorithm.

    Dimension Support:
        - Continuous: YES (KDE-based modeling)
        - Categorical: YES (with index encoding)
        - Discrete: YES (treated as continuous for KDE)

    TPE models the conditional probability P(x|y) instead of P(y|x) used by
    other SMBO methods. It maintains two kernel density estimators: one for
    the best-performing samples (l(x)) and one for the rest (g(x)). The
    acquisition function is the ratio l(x)/g(x).

    The key insight is that maximizing Expected Improvement (EI) is equivalent
    to maximizing l(x)/g(x) when we model p(x|y<y*) with l(x) and p(x|y>=y*)
    with g(x), where y* is the quantile threshold.

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
        Probability of random iteration.
    nth_process : int, optional
        Process index for parallel optimization.
    warm_start_smbo : pd.DataFrame, optional
        Previous results to initialize the KDEs.
    max_sample_size : int, default=10000000
        Maximum positions to consider.
    sampling : dict or False, default=None
        Sampling strategy for large search spaces.
    replacement : bool, default=True
        Allow re-evaluation of positions.
    gamma_tpe : float, default=0.2
        Quantile threshold for splitting samples into "good" and "bad".
        Top gamma_tpe fraction are used for l(x), rest for g(x).
    """

    name = "Tree Structured Parzen Estimators"
    _name_ = "tree_structured_parzen_estimators"
    __name__ = "TreeStructuredParzenEstimators"

    optimizer_type = "sequential"
    computationally_expensive = True

    def __init__(
        self,
        search_space: dict[str, Any],
        initialize: dict[str, int] | None = None,
        constraints: list[Callable[[dict[str, Any]], bool]] | None = None,
        random_state: int | None = None,
        rand_rest_p: float = 0,
        nth_process: int | None = None,
        warm_start_smbo: pd.DataFrame | None = None,
        max_sample_size: int = 10000000,
        sampling: dict[str, int] | Literal[False] | None = None,
        replacement: bool = True,
        gamma_tpe: float = 0.2,
    ) -> None:
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            warm_start_smbo=warm_start_smbo,
            max_sample_size=max_sample_size,
            sampling=sampling,
            replacement=replacement,
        )

        self.gamma_tpe = gamma_tpe

        # Initialize KDE - sklearn and native have different interfaces
        if SKLEARN_AVAILABLE:
            kde_params: dict[str, Any] = {"kernel": "gaussian", "bandwidth": 1.0}
        else:
            kde_params = {"bandwidth": 1.0}

        self.kd_best = KernelDensity(**kde_params)
        self.kd_worst = KernelDensity(**kde_params)

    def _get_samples(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Split samples into best and worst groups based on gamma_tpe.

        Returns
        -------
        best_samples : list
            Top gamma_tpe fraction of samples by score.
        worst_samples : list
            Remaining (1 - gamma_tpe) fraction of samples.
        """
        n_samples = len(self.X_sample)
        n_best = max(round(n_samples * self.gamma_tpe), 1)

        # Ensure we have at least 1 sample in each group
        n_worst = max(int(n_samples - n_best), 1)
        # Adjust n_best if n_samples is too small for both groups
        if n_best + n_worst > n_samples and n_samples >= 2:
            n_best = max(n_samples - 1, 1)
            n_worst = max(n_samples - n_best, 1)

        Y_sample = np.array(self.Y_sample)
        sorted_indices = Y_sample.argsort()
        index_best = sorted_indices[-n_best:]
        index_worst = sorted_indices[:n_worst]

        best_samples = [self.X_sample[i] for i in index_best]
        worst_samples = [self.X_sample[i] for i in index_worst]

        return best_samples, worst_samples

    def _expected_improvement(self) -> np.ndarray:
        """Compute acquisition as ratio of good/bad density estimates.

        The acquisition function for TPE is derived from Expected Improvement
        but expressed as l(x)/g(x), the ratio of densities for good and bad
        observations.

        Returns
        -------
        np.ndarray
            Acquisition values for each candidate position.
        """
        self.pos_comb = self._sampling(self.all_pos_comb)

        logprob_best = self.kd_best.score_samples(self.pos_comb)
        logprob_worst = self.kd_worst.score_samples(self.pos_comb)

        prob_best = np.exp(np.array(logprob_best))
        prob_worst = np.exp(np.array(logprob_worst))

        # Safe division: only divide where prob_worst != 0
        worst_over_best = np.zeros_like(prob_worst)
        nonzero_worst = prob_worst != 0
        nonzero_best = prob_best != 0

        # Where both are nonzero, compute ratio
        both_nonzero = nonzero_worst & nonzero_best
        worst_over_best[both_nonzero] = (
            prob_worst[both_nonzero] / prob_best[both_nonzero]
        )

        # Where worst != 0 but best == 0, set to inf
        worst_only = nonzero_worst & ~nonzero_best
        worst_over_best[worst_only] = float("inf")

        # Compute expected improvement inverse and invert
        exp_imp_inv = self.gamma_tpe + worst_over_best * (1 - self.gamma_tpe)
        exp_imp = 1 / exp_imp_inv

        return exp_imp

    def _training(self) -> None:
        """Fit KDE models on best and worst sample groups."""
        best_samples, worst_samples = self._get_samples()

        self.kd_best.fit(best_samples)
        self.kd_worst.fit(worst_samples)
