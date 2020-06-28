# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .bayesian_optimization import BayesianOptimizer


class DecisionTreeOptimizer(BayesianOptimizer):
    """Based on the forest-optimizer in the scikit-optimize package"""

    def __init__(self, init_positions, space_dim, opt_para):
        super().__init__(init_positions, space_dim, opt_para)
        self.regr = self._opt_args_.tree_regressor
