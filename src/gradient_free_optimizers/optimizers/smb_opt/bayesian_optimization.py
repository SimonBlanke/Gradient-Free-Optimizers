# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from scipy.stats import norm

from .smbo import SMBO
from .surrogate_models import (
    GPR_linear,
    GPR,
)
from .acquisition_function import ExpectedImprovement
from ._normalize import normalize


gaussian_process = {"gp_nonlinear": GPR(), "gp_linear": GPR_linear()}


class BayesianOptimizer(SMBO):
    """Bayesian optimization using Gaussian Process regression.

    Uses a Gaussian Process as surrogate model to approximate the objective
    function and Expected Improvement as acquisition function. The GP provides
    both mean predictions and uncertainty estimates, enabling principled
    exploration-exploitation trade-offs.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to arrays of possible values.
    initialize : dict, default={"grid": 4, "random": 2, "vertices": 4}
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
    sampling : dict or False, default={"random": 1000000}
        Sampling strategy for large search spaces.
    replacement : bool, default=True
        Allow re-evaluation of positions.
    gpr : object, default=GPR()
        Gaussian Process regressor instance (gp_nonlinear or gp_linear).
    xi : float, default=0.03
        Exploration-exploitation parameter for Expected Improvement.
        Higher values favor exploration.

    See Also
    --------
    ForestOptimizer : Uses tree ensemble instead of GP.
    TreeStructuredParzenEstimators : Non-parametric density estimation approach.
    """

    name = "Bayesian Optimization"
    _name_ = "bayesian_optimization"
    __name__ = "BayesianOptimizer"

    optimizer_type = "sequential"
    computationally_expensive = True

    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        warm_start_smbo=None,
        max_sample_size=10000000,
        sampling={"random": 1000000},
        replacement=True,
        gpr=gaussian_process["gp_nonlinear"],
        xi=0.03,
    ):
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

        self.gpr = gpr
        self.regr = gpr
        self.xi = xi

    def finish_initialization(self):
        self.all_pos_comb = self._all_possible_pos()
        return super().finish_initialization()

    def _expected_improvement(self):
        """Compute Expected Improvement for all candidate positions."""
        self.pos_comb = self._sampling(self.all_pos_comb)

        acqu_func = ExpectedImprovement(self.regr, self.pos_comb, self.xi)
        return acqu_func.calculate(self.X_sample, self.Y_sample)

    def _training(self):
        """Fit the Gaussian Process on collected samples."""
        X_sample = np.array(self.X_sample)
        Y_sample = np.array(self.Y_sample)

        Y_sample = normalize(Y_sample).reshape(-1, 1)
        self.regr.fit(X_sample, Y_sample)
