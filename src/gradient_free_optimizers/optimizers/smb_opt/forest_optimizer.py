# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from scipy.stats import norm

from .smbo import SMBO
from .surrogate_models import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from .acquisition_function import ExpectedImprovement
from ._normalize import normalize


tree_regressor_dict = {
    "random_forest": RandomForestRegressor,
    "extra_tree": ExtraTreesRegressor,
    "gradient_boost": GradientBoostingRegressor,
}


class ForestOptimizer(SMBO):
    """Sequential model-based optimization using tree ensemble surrogates.

    Uses a tree-based ensemble (Random Forest, Extra Trees, or Gradient Boosting)
    as surrogate model. Tree ensembles can capture non-linear relationships and
    provide uncertainty estimates via prediction variance across trees.

    Based on the forest-optimizer in the scikit-optimize package.

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
        Previous results to initialize the model.
    max_sample_size : int, default=10000000
        Maximum positions to consider.
    sampling : dict or False, default={"random": 1000000}
        Sampling strategy for large search spaces.
    replacement : bool, default=True
        Allow re-evaluation of positions.
    tree_regressor : str, default="extra_tree"
        Type of tree ensemble: "random_forest", "extra_tree", or "gradient_boost".
    tree_para : dict, default={"n_estimators": 100}
        Parameters passed to the tree regressor.
    xi : float, default=0.03
        Exploration-exploitation parameter for Expected Improvement.

    See Also
    --------
    BayesianOptimizer : Uses Gaussian Process instead of trees.
    TreeStructuredParzenEstimators : Density estimation approach.
    """

    name = "Forest Optimization"
    _name_ = "forest_optimization"
    __name__ = "ForestOptimizer"

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
        tree_regressor="extra_tree",
        tree_para={"n_estimators": 100},
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

        self.tree_regressor = tree_regressor
        self.tree_para = tree_para
        self.regr = tree_regressor_dict[tree_regressor](**self.tree_para)
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
        """Fit the tree ensemble on collected samples."""
        X_sample = np.array(self.X_sample)
        Y_sample = np.array(self.Y_sample)

        if len(Y_sample) == 0:
            return self.move_random()

        Y_sample = normalize(Y_sample).reshape(-1, 1)
        self.regr.fit(X_sample, Y_sample)
