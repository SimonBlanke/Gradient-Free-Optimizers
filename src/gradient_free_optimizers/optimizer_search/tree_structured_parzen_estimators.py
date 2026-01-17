# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Tree-structured Parzen Estimator (TPE) using kernel density estimation."""

from typing import Literal

from .._init_utils import get_default_initialize, get_default_sampling
from ..optimizers_new import (
    TreeStructuredParzenEstimators as _TreeStructuredParzenEstimators,
)
from ..search import Search


class TreeStructuredParzenEstimators(_TreeStructuredParzenEstimators, Search):
    """
    Sequential model-based optimizer using kernel density estimation.

    Tree-structured Parzen Estimator (TPE) is an efficient sequential model-based
    optimization algorithm that models the probability of good and bad parameter
    configurations separately. Unlike Bayesian Optimization which models P(y|x),
    TPE models P(x|y) by maintaining two density estimators: one for parameters
    that led to good results (l) and one for poor results (g).

    The algorithm selects the next point by maximizing the ratio l(x)/g(x), which
    is equivalent to optimizing Expected Improvement but is computationally more
    efficient. TPE uses kernel density estimation (Parzen estimators) to model
    these distributions, with a tree structure for handling conditional parameters.

    The algorithm is well-suited for:

    - Hyperparameter optimization of machine learning models
    - High-dimensional optimization problems (scales better than GP-based methods)
    - Problems with conditional or hierarchical parameter spaces
    - Situations requiring fast surrogate model updates

    The `gamma_tpe` parameter controls the quantile threshold for splitting
    observations into good and bad groups. A value of 0.2 means the top 20%
    of observations are considered "good".

    Parameters
    ----------
    search_space : dict[str, list]
        The search space to explore. A dictionary with parameter
        names as keys and a numpy array as values.
    initialize : dict[str, int]
        The method to generate initial positions. A dictionary with
        the following key literals and the corresponding value type:
        {"grid": int, "vertices": int, "random": int, "warm_start": list[dict]}
    constraints : list[callable]
        A list of constraints, where each constraint is a callable.
        The callable returns `True` or `False` dependend on the input parameters.
    random_state : None, int
        If None, create a new random state. If int, create a new random state
        seeded with the value.
    rand_rest_p : float
        The probability of a random iteration during the search process.
    warm_start_smbo : object, optional
        Previous SMBO state for warm starting optimization.
    max_sample_size : int
        Maximum number of candidate points for acquisition optimization.
        Default is 10000000.
    sampling : dict
        Configuration for candidate sampling. Default is {"random": 1000000}.
    replacement : bool
        Whether to sample candidates with replacement. Default is True.
    gamma_tpe : float
        Quantile threshold for splitting observations into good/bad groups.
        Value between 0 and 1, where 0.2 means top 20% are "good".
        Lower values are more selective. Default is 0.2.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import TreeStructuredParzenEstimators

    >>> def ml_hyperparameter_objective(para):
    ...     # Simulating ML model performance
    ...     learning_rate = para["learning_rate"]
    ...     n_estimators = para["n_estimators"]
    ...     return -(0.9 - abs(learning_rate - 0.1) - abs(n_estimators - 100) / 1000)

    >>> search_space = {
    ...     "learning_rate": np.logspace(-4, 0, 100),
    ...     "n_estimators": np.arange(10, 500, 10),
    ... }

    >>> opt = TreeStructuredParzenEstimators(search_space, gamma_tpe=0.25)
    >>> opt.search(ml_hyperparameter_objective, n_iter=100)
    """

    def __init__(
        self,
        search_space: dict[str, list],
        initialize: dict[
            Literal["grid", "vertices", "random", "warm_start"],
            int | list[dict],
        ] = None,
        constraints: list[callable] = None,
        random_state: int = None,
        rand_rest_p: float = 0,
        nth_process: int = None,
        warm_start_smbo=None,
        max_sample_size: int = 10000000,
        sampling: dict[Literal["random"], int] = None,
        replacement: bool = True,
        gamma_tpe=0.2,
    ):
        if initialize is None:
            initialize = get_default_initialize()
        if constraints is None:
            constraints = []
        if sampling is None:
            sampling = get_default_sampling()

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
            gamma_tpe=gamma_tpe,
        )
