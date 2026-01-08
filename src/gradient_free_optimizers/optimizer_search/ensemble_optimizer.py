# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Ensemble optimizer combining multiple surrogate model types."""

from typing import Literal

from ..optimizers import EnsembleOptimizer as _EnsembleOptimizer
from ..search import Search


class EnsembleOptimizer(_EnsembleOptimizer, Search):
    """
    Sequential model-based optimizer combining multiple surrogate model types.

    Ensemble Optimizer combines predictions from multiple surrogate model types
    to make more robust optimization decisions. By aggregating different models
    (e.g., Gaussian Processes, tree ensembles, kernel density estimators), the
    ensemble can leverage the strengths of each approach while mitigating their
    individual weaknesses.

    This approach is particularly useful when it is unclear which surrogate model
    type is best suited for a given problem. The ensemble provides more stable
    uncertainty estimates and can adapt to different regions of the search space
    where different model types may perform better.

    The algorithm is well-suited for:

    - Problems where the best surrogate model type is unknown
    - Robust optimization requiring reliable uncertainty estimates
    - Complex objective functions with varying characteristics
    - Situations where model selection overhead is acceptable

    The ensemble combines SMBO techniques to provide a general-purpose
    optimizer that performs well across a wide range of problem types.

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
        The probability of a random iteration during the the search process.
    warm_start_smbo : object, optional
        Previous SMBO state for warm starting optimization.
    max_sample_size : int
        Maximum number of candidate points for acquisition optimization.
        Default is 10000000.
    sampling : dict
        Configuration for candidate sampling. Default is {"random": 1000000}.
    replacement : bool
        Whether to sample candidates with replacement. Default is True.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import EnsembleOptimizer

    >>> def complex_objective(para):
    ...     x, y = para["x"], para["y"]
    ...     return -(x**2 + y**2 + 0.5 * np.sin(10 * x) * np.cos(10 * y))

    >>> search_space = {
    ...     "x": np.linspace(-2, 2, 100),
    ...     "y": np.linspace(-2, 2, 100),
    ... }

    >>> opt = EnsembleOptimizer(search_space)
    >>> opt.search(complex_objective, n_iter=100)
    """

    def __init__(
        self,
        search_space: dict[str, list],
        initialize: dict[
            Literal["grid", "vertices", "random", "warm_start"],
            int | list[dict],
        ] = {"grid": 4, "random": 2, "vertices": 4},
        constraints: list[callable] = [],
        random_state: int = None,
        rand_rest_p: float = 0,
        nth_process: int = None,
        warm_start_smbo=None,
        max_sample_size: int = 10000000,
        sampling: dict[Literal["random"], int] = {"random": 1000000},
        replacement: bool = True,
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
