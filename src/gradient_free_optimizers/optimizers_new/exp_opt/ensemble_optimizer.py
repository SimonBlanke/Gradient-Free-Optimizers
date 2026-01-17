# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Ensemble Optimizer.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
(depends on component optimizers)
"""

import numpy as np

from ..core_optimizer import CoreOptimizer


class EnsembleOptimizer(CoreOptimizer):
    """Ensemble of multiple optimizers.

    Dimension Support:
        - Continuous: Depends on component optimizers
        - Categorical: Depends on component optimizers
        - Discrete: Depends on component optimizers

    Runs multiple optimizers and combines their results.
    """

    name = "Ensemble Optimizer"
    _name_ = "ensemble_optimizer"
    __name__ = "EnsembleOptimizer"

    optimizer_type = "ensemble"
    computationally_expensive = True

    def __init__(
        self,
        search_space,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        optimizers=None,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )
        self.optimizers = optimizers or []

    def iterate(self):
        """Run iteration on component optimizers."""
        # TODO: Implement ensemble iteration
        raise NotImplementedError("iterate() not yet implemented")

    def _iterate_continuous_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Ensemble delegates to component optimizers."""
        raise NotImplementedError("Use iterate() for ensemble optimization")

    def _iterate_categorical_batch(
        self,
        current: np.ndarray,
        n_categories: np.ndarray,
    ) -> np.ndarray:
        """Ensemble delegates to component optimizers."""
        raise NotImplementedError("Use iterate() for ensemble optimization")

    def _iterate_discrete_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Ensemble delegates to component optimizers."""
        raise NotImplementedError("Use iterate() for ensemble optimization")

    def evaluate(self, score_new):
        """Evaluate across ensemble."""
        # TODO: Implement ensemble evaluation
        raise NotImplementedError("evaluate() not yet implemented")
