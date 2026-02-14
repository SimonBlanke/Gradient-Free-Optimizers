# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Random search that samples uniformly from the search space."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers import RandomSearchOptimizer as _RandomSearchOptimizer
from ..search import Search


class RandomSearchOptimizer(_RandomSearchOptimizer, Search):
    r"""
    Simple optimizer that samples random positions from the search space.

    Random Search is the simplest optimization strategy that samples positions
    uniformly at random from the search space. Despite its simplicity, random
    search is surprisingly effective for many problems, especially in high
    dimensions where it can outperform grid search due to better coverage of
    important dimensions.

    The algorithm has no memory of previous evaluations and makes no assumptions
    about the objective function. Each iteration independently samples a random
    point, making it trivially parallelizable and resistant to getting stuck
    in local optima.

    The algorithm is well-suited for:

    - High-dimensional search spaces (avoids curse of dimensionality)
    - Baseline comparison for other optimization methods
    - Initial exploration before applying more sophisticated methods
    - Problems with many irrelevant dimensions
    - Embarrassingly parallel optimization scenarios

    Random search is often recommended as a first approach for hyperparameter
    optimization due to its simplicity and effectiveness. It provides uniform
    coverage of the search space without requiring any tuning parameters.

    Parameters
    ----------
    search_space : dict[str, list]
        The search space to explore, defined as a dictionary mapping parameter
        names to arrays of possible values.

        Each key is a parameter name (string), and each value is a numpy array
        or list of discrete values that the parameter can take. The optimizer
        will only evaluate positions that are on this discrete grid.

        Example: A 2D search space with 100 points per dimension::

            search_space = {
                "x": np.linspace(-10, 10, 100),
                "y": np.linspace(-10, 10, 100),
            }

        The resolution of each dimension (number of points in the array)
        directly affects optimization quality and speed. More points give
        finer resolution but increase the search space size exponentially.

    initialize : dict[str, int], default={"vertices": 4, "random": 2}
        Strategy for generating initial positions before the main optimization
        loop begins. Initialization samples are evaluated first, and the best
        one becomes the starting point for the optimizer.

        Supported keys:

        - ``"grid"``: ``int`` -- Number of positions on a regular grid.
        - ``"vertices"``: ``int`` -- Number of corner/edge positions of the
          search space.
        - ``"random"``: ``int`` -- Number of uniformly random positions.
        - ``"warm_start"``: ``list[dict]`` -- Specific positions to evaluate,
          each as a dict mapping parameter names to values.

        Multiple strategies can be combined::

            initialize = {"vertices": 4, "random": 10}
            initialize = {"warm_start": [{"x": 0.5, "y": 1.0}], "random": 5}

        More initialization samples improve the starting point but consume
        iterations from ``n_iter``. For expensive objectives, a few targeted
        warm-start points are often more efficient than many random samples.

    constraints : list[callable], default=[]
        A list of constraint functions that restrict the search space. Each
        constraint is a callable that receives a parameter dictionary and
        returns ``True`` if the position is valid, ``False`` if it should
        be rejected.

        Rejected positions are discarded and regenerated: the optimizer
        resamples a new candidate position (up to 100 retries per step).
        During initialization, positions that violate constraints are
        filtered out entirely.

        Example: Constrain the search to a circular region::

            def circular_constraint(para):
                return para["x"]**2 + para["y"]**2 <= 25

            constraints = [circular_constraint]

        Multiple constraints are combined with AND logic (all must return
        ``True``).

    random_state : int or None, default=None
        Seed for the random number generator to ensure reproducible results.

        - ``None``: Use a new random state each run (non-deterministic).
        - ``int``: Seed the random number generator for reproducibility.

        Setting a fixed seed is recommended for debugging and benchmarking.
        Different seeds may lead to different optimization trajectories,
        especially for stochastic optimizers.

    Notes
    -----
    At each iteration, the algorithm independently samples a uniformly
    random position from the search space:

    .. math::

        x_t \\sim \\text{Uniform}(\\text{search\\_space})

    Random search has no memory of previous evaluations. Each sample is
    independent, making it trivially parallelizable and immune to local
    optima traps. Research has shown that random search is more efficient
    than grid search in high dimensions because it provides better coverage
    of each individual dimension.

    Time complexity per iteration is O(d), where d is the number of
    dimensions.

    For visual explanations and comparisons, see
    the :ref:`Random Search user guide <random_search>`.

    See Also
    --------
    GridSearchOptimizer : Systematic exhaustive search on a regular grid.
    RandomAnnealingOptimizer : Starts with large random steps and narrows over time.
    BayesianOptimizer : Sample-efficient alternative that builds a surrogate model.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import RandomSearchOptimizer

    >>> def objective(para):
    ...     return -(para["x"] ** 2 + para["y"] ** 2)

    >>> search_space = {
    ...     "x": np.linspace(-10, 10, 100),
    ...     "y": np.linspace(-10, 10, 100),
    ... }

    >>> opt = RandomSearchOptimizer(search_space)
    >>> opt.search(objective, n_iter=1000)
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
        nth_process: int = None,
    ):
        if initialize is None:
            initialize = get_default_initialize()
        if constraints is None:
            constraints = []

        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            nth_process=nth_process,
        )
