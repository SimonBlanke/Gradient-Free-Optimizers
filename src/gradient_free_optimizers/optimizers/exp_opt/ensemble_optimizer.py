# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from gradient_free_optimizers._init_utils import (
    get_default_initialize,
    get_default_sampling,
)

from ..smb_opt._normalize import normalize
from ..smb_opt.acquisition_function import ExpectedImprovement
from ..smb_opt.smbo import SMBO
from ..smb_opt.surrogate_models import EnsembleRegressor


class EnsembleOptimizer(SMBO):
    """Ensemble-based sequential model-based optimization.

    Combines multiple surrogate models (e.g., Gradient Boosting, Gaussian Process)
    into an ensemble for more robust predictions. This experimental optimizer
    averages predictions from multiple models to reduce variance.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to arrays of possible values.
    initialize : dict, default=None
        Strategy for generating initial positions.
        If None, uses {"grid": 4, "random": 2, "vertices": 4}.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random restart.
    nth_process : int, optional
        Process index for parallel optimization.
    epsilon : float, default=0.03
        Step size for local search fallback.
    distribution : str, default="normal"
        Distribution for step sizes.
    n_neighbours : int, default=3
        Number of neighbors for local search.
    estimators : list
        List of scikit-learn estimator instances to ensemble.
    xi : float, default=0.01
        Exploration-exploitation parameter for Expected Improvement.
    warm_start_smbo : pd.DataFrame, optional
        Previous results to initialize the models.
    max_sample_size : int, default=10000000
        Maximum positions to consider.
    sampling : dict or False, default=None
        Sampling strategy for large search spaces.
        If None, uses {"random": 1000000}.
    replacement : bool, default=True
        Allow re-evaluation of positions.
    warnings : int, default=100000000
        Threshold for memory warnings.

    Notes
    -----
    This is an experimental optimizer. For production use, consider
    BayesianOptimizer or ForestOptimizer instead.
    """

    name = "Ensemble Optimizer"

    def __init__(
        self,
        search_space,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        epsilon=0.03,
        distribution="normal",
        n_neighbours=3,
        estimators=None,
        xi=0.01,
        warm_start_smbo=None,
        max_sample_size=10000000,
        sampling=None,
        replacement=True,
        warnings=100000000,
        **kwargs,
    ):
        if initialize is None:
            initialize = get_default_initialize()
        if sampling is None:
            sampling = get_default_sampling()
        if estimators is None:
            estimators = [
                GradientBoostingRegressor(n_estimators=5),
                GaussianProcessRegressor(),
            ]

        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            epsilon=epsilon,
            distribution=distribution,
            n_neighbours=n_neighbours,  #
            warm_start_smbo=warm_start_smbo,
            max_sample_size=max_sample_size,
            sampling=sampling,
            replacement=replacement,
        )
        self.estimators = estimators
        self.regr = EnsembleRegressor(estimators)
        self.xi = xi
        self.warm_start_smbo = warm_start_smbo
        self.max_sample_size = max_sample_size
        self.sampling = sampling
        self.warnings = warnings

        self.init_warm_start_smbo()

    def finish_initialization(self):
        self.all_pos_comb = self._all_possible_pos()
        return super().finish_initialization()

    def _expected_improvement(self):
        self.pos_comb = self._sampling(self.all_pos_comb)

        acqu_func = ExpectedImprovement(self.regr, self.pos_comb, self.xi)
        return acqu_func.calculate(self.X_sample, self.Y_sample)

    def _training(self):
        X_sample = np.array(self.X_sample)
        Y_sample = np.array(self.Y_sample)

        if len(Y_sample) == 0:
            return self.move_random()

        Y_sample = normalize(Y_sample).reshape(-1, 1)
        self.regr.fit(X_sample, Y_sample)
