# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Bayesian Optimization with Gaussian Process.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np

from .smbo import SMBO
from .surrogate_models import GPR
from .acquisition_function import ExpectedImprovement

if TYPE_CHECKING:
    import pandas as pd


def normalize(arr):
    """Normalize array to [0, 1] range."""
    arr = np.array(arr)
    arr_min = arr.min()
    arr_max = arr.max()
    range_ = arr_max - arr_min

    if range_ == 0:
        return np.random.uniform(0, 1, size=arr.shape)
    else:
        return (arr - arr_min) / range_


class BayesianOptimizer(SMBO):
    """Bayesian Optimization with Gaussian Process surrogate.

    Dimension Support:
        - Continuous: YES (native GP support)
        - Categorical: YES (with index encoding)
        - Discrete: YES (treated as continuous, then rounded)

    Uses a Gaussian Process as surrogate model to approximate the objective
    function and Expected Improvement as acquisition function. The GP provides
    both mean predictions and uncertainty estimates, enabling principled
    exploration-exploitation trade-offs.

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
        Previous results to initialize the GP.
    max_sample_size : int, default=10000000
        Maximum positions to consider.
    sampling : dict or False, default=None
        Sampling strategy for large search spaces.
    replacement : bool, default=True
        Allow re-evaluation of positions.
    gpr : object, default=None
        Gaussian Process regressor instance. If None, uses default GPR.
    xi : float, default=0.03
        Exploration-exploitation parameter for Expected Improvement.
        Higher values favor exploration.
    """

    name = "Bayesian Optimization"
    _name_ = "bayesian_optimization"
    __name__ = "BayesianOptimizer"

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
        gpr=None,
        xi: float = 0.03,
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

        # Instantiate GPR - supports both class and instance
        if gpr is None:
            self.gpr = GPR()
        elif isinstance(gpr, type):
            # User passed a class, instantiate it
            self.gpr = gpr()
        else:
            # User passed an instance
            self.gpr = gpr

        self.regr = self.gpr
        self.xi = xi

    def _expected_improvement(self) -> np.ndarray:
        """Compute Expected Improvement for all candidate positions."""
        self.pos_comb = self._sampling(self.all_pos_comb)

        acqu_func = ExpectedImprovement(self.regr, self.pos_comb, self.xi)
        return acqu_func.calculate(self.X_sample, self.Y_sample)

    def _training(self) -> None:
        """Fit the Gaussian Process on training data."""
        X_sample = np.array(self.X_sample)
        Y_sample = np.array(self.Y_sample)

        Y_sample = normalize(Y_sample).reshape(-1, 1)
        self.regr.fit(X_sample, Y_sample)
