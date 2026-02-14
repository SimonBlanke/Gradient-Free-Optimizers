# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Ensemble Optimizer.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
Combines multiple surrogate models for robust predictions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from ..smb_opt import SMBO
from ..smb_opt._normalize import normalize
from ..smb_opt.acquisition_function import ExpectedImprovement
from ..smb_opt.surrogate_models import EnsembleRegressor

if TYPE_CHECKING:
    import pandas as pd


class EnsembleOptimizer(SMBO):
    """Ensemble-based sequential model-based optimization.

    Dimension Support:
        - Continuous: YES (surrogate model based)
        - Categorical: YES (with index encoding)
        - Discrete: YES (surrogate model based)

    Combines multiple surrogate models (e.g., Gradient Boosting, Gaussian Process)
    into an ensemble for more robust predictions. This experimental optimizer
    averages predictions from multiple models to reduce variance and improve
    reliability.

    The algorithm:
    1. Train ensemble of surrogate models on observed data
    2. Compute expected improvement using ensemble predictions
    3. Select position with highest expected improvement
    4. Repeat until convergence

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
    estimators : list, optional
        List of scikit-learn estimator instances to ensemble.
        Default uses GradientBoostingRegressor and GaussianProcessRegressor.
    xi : float, default=0.01
        Exploration-exploitation parameter for Expected Improvement.
    warm_start_smbo : pd.DataFrame, optional
        Previous optimization results to initialize the surrogate model.
    max_sample_size : int, default=10000000
        Maximum number of positions to consider for sampling.
    sampling : dict, False, or None, default=None
        Sampling strategy for large search spaces.
    replacement : bool, default=True
        Whether to allow re-evaluation of the same position.

    Notes
    -----
    This is an experimental optimizer. For production use, consider
    BayesianOptimizer or ForestOptimizer instead.
    """

    name = "Ensemble Optimizer"
    _name_ = "ensemble_optimizer"
    __name__ = "EnsembleOptimizer"

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
        estimators: list | None = None,
        xi: float = 0.01,
        warm_start_smbo: pd.DataFrame | None = None,
        max_sample_size: int = 10000000,
        sampling: dict[str, int] | Literal[False] | None = None,
        replacement: bool = True,
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

        # Default ensemble of estimators
        if estimators is None:
            estimators = [
                GradientBoostingRegressor(n_estimators=5),
                GaussianProcessRegressor(),
            ]

        self.estimators = estimators
        self.regr = EnsembleRegressor(estimators)
        self.xi = xi

    def _on_finish_initialization(self) -> None:
        """Generate candidate positions for ensemble model.

        Called by CoreOptimizer.finish_initialization() after all init
        positions have been evaluated. Generates the candidate position
        grid for the acquisition function.

        Note: DO NOT set search_state here - CoreOptimizer handles that.
        """
        self.all_pos_comb = self._all_possible_pos()

    def _training(self) -> None:
        """Train the ensemble of surrogate models."""
        X_sample = np.array(self.X_sample)
        Y_sample = np.array(self.Y_sample)

        if len(Y_sample) == 0:
            return

        # Normalize Y values for better model fitting
        Y_sample = normalize(Y_sample).reshape(-1, 1)
        self.regr.fit(X_sample, Y_sample)

    def _expected_improvement(self) -> np.ndarray:
        """Compute expected improvement for candidate positions."""
        self.pos_comb = self._sampling(self.all_pos_comb)

        acqu_func = ExpectedImprovement(self.regr, self.pos_comb, self.xi)
        return acqu_func.calculate(self.X_sample, self.Y_sample)

    def _on_evaluate(self, score_new: float) -> None:
        """Update best and current positions.

        Parameters
        ----------
        score_new : float
            Score for the evaluated position.
        """
        self._update_best(self._pos_new, score_new)
        self._update_current(self._pos_new, score_new)
