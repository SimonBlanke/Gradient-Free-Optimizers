# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Bayesian Optimization with Gaussian Process.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
"""

from .smbo import SMBO


class BayesianOptimizer(SMBO):
    """Bayesian Optimization with Gaussian Process surrogate.

    Dimension Support:
        - Continuous: YES (native GP support)
        - Categorical: YES (with one-hot encoding)
        - Discrete: YES (treated as continuous, then rounded)

    Uses a Gaussian Process to model the objective function and
    Expected Improvement as the acquisition function.
    """

    name = "Bayesian Optimizer"
    _name_ = "bayesian_optimizer"
    __name__ = "BayesianOptimizer"

    optimizer_type = "sequential"
    computationally_expensive = True

    def __init__(
        self,
        search_space,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        max_sample_size=10000000,
        xi=0.01,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            max_sample_size=max_sample_size,
        )
        self.xi = xi  # Exploration-exploitation trade-off

    def evaluate(self, score_new):
        """Evaluate and update GP model."""
        # TODO: Implement GP update
        raise NotImplementedError("evaluate() not yet implemented")
