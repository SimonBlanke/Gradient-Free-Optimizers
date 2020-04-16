# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .bayesian_optimization import BayesianOptimizer


class DecisionTreeOptimizer(BayesianOptimizer):
    """Based on the forest-optimizer in the scikit-optimize package"""

    def __init__(self, n_iter, opt_para):
        super().__init__(n_iter, opt_para)
        self.n_positioners = 1
        self.regr = self._opt_args_.tree_regressor
