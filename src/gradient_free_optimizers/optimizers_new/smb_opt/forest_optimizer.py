# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Forest (Random Forest) Optimizer.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
"""

from .smbo import SMBO


class ForestOptimizer(SMBO):
    """Forest Optimizer using Random Forest surrogate.

    Dimension Support:
        - Continuous: YES (RF handles continuous naturally)
        - Categorical: YES (RF handles categorical naturally)
        - Discrete: YES (RF handles discrete naturally)

    Uses a Random Forest to model the objective function.
    Generally faster than GP for high-dimensional problems.
    """

    name = "Forest Optimizer"
    _name_ = "forest_optimizer"
    __name__ = "ForestOptimizer"

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
        n_estimators=100,
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
        self.n_estimators = n_estimators

    def evaluate(self, score_new):
        """Evaluate and update RF model."""
        # TODO: Implement RF update
        raise NotImplementedError("evaluate() not yet implemented")
