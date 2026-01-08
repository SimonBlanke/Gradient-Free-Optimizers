# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from ..smb_opt.smbo import SMBO
from ..smb_opt.surrogate_models import EnsembleRegressor
from ..smb_opt.acquisition_function import ExpectedImprovement
from ..smb_opt._normalize import normalize

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor


class EnsembleOptimizer(SMBO):
    """Ensemble-based sequential model-based optimization.

    Combines multiple surrogate models (e.g., Gradient Boosting, Gaussian Process)
    into an ensemble for more robust predictions. This experimental optimizer
    averages predictions from multiple models to reduce variance.

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
    sampling : dict or False, default={"random": 1000000}
        Sampling strategy for large search spaces.
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
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        epsilon=0.03,
        distribution="normal",
        n_neighbours=3,
        estimators=[
            GradientBoostingRegressor(n_estimators=5),
            # DecisionTreeRegressor(),
            # MLPRegressor(),
            GaussianProcessRegressor(),
        ],
        xi=0.01,
        warm_start_smbo=None,
        max_sample_size=10000000,
        sampling={"random": 1000000},
        replacement=True,
        warnings=100000000,
        **kwargs,
    ):
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
