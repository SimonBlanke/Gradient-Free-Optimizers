# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Tree-structured Parzen Estimators (TPE).

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
"""

from .smbo import SMBO


class TreeStructuredParzenEstimators(SMBO):
    """TPE optimizer as used in Hyperopt.

    Dimension Support:
        - Continuous: YES (KDE-based modeling)
        - Categorical: YES (categorical distribution)
        - Discrete: YES (discrete distribution)

    Models p(x|y) and p(y) separately for good and bad observations,
    then maximizes the ratio p(x|y<y*) / p(x|y>=y*).
    """

    name = "TPE"
    _name_ = "tpe"
    __name__ = "TreeStructuredParzenEstimators"

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
        gamma=0.25,
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
        self.gamma = gamma  # Quantile for splitting good/bad

    def evaluate(self, score_new):
        """Evaluate and update TPE model."""
        # TODO: Implement TPE update
        raise NotImplementedError("evaluate() not yet implemented")
