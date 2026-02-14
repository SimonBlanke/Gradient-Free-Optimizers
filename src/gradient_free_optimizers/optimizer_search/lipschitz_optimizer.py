# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Lipschitz optimization using continuity bounds for deterministic search."""

from typing import Literal

from .._init_utils import get_default_initialize, get_default_sampling
from ..optimizers import LipschitzOptimizer as _LipschitzOptimizer
from ..search import Search


class LipschitzOptimizer(_LipschitzOptimizer, Search):
    r"""
    Global optimizer using Lipschitz continuity bounds for deterministic search.

    Lipschitz Optimization exploits the Lipschitz continuity property of objective
    functions to provide deterministic bounds on the optimum. If a function is
    Lipschitz continuous with constant L, the function value at any point cannot
    differ from nearby points by more than L times the distance. This property
    allows the algorithm to rule out regions that provably cannot contain the
    global optimum.

    The algorithm estimates the Lipschitz constant from observed data and uses
    it to compute upper bounds on the objective function across the search space.
    Points with the highest potential (according to these bounds) are selected
    for evaluation, gradually tightening the bounds until convergence.

    The algorithm is well-suited for:

    - Lipschitz continuous objective functions
    - Problems requiring global optimality guarantees
    - Low-dimensional optimization (bounds become loose in high dimensions)
    - Deterministic optimization without relying on random sampling

    This method provides theoretical guarantees for finding the global optimum
    but requires the objective function to satisfy Lipschitz continuity. The
    estimated Lipschitz constant adapts as more observations are collected.

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

    rand_rest_p : float, default=0
        Probability of performing a random restart instead of the normal
        algorithm step. At each iteration, a uniform random number is drawn;
        if it falls below ``rand_rest_p``, the optimizer jumps to a random
        position instead of following its strategy.

        - ``0.0``: No random restarts (pure algorithm behavior).
        - ``0.01-0.05``: Light diversification, helps escape shallow local
          optima.
        - ``0.1-0.3``: Aggressive restarts, useful for highly multi-modal
          landscapes.
        - ``1.0``: Equivalent to random search.

        This is especially useful for local search optimizers (Hill Climbing,
        Simulated Annealing) that can get trapped. For population-based
        optimizers, the effect is less pronounced since they already maintain
        diversity through multiple agents.

    warm_start_smbo : object or None, default=None
        Previous SMBO state for warm-starting the surrogate model. Allows
        continuing optimization from a previous run by reusing the fitted
        model state, avoiding the cost of refitting from scratch.

        Pass ``None`` to start fresh without warm-starting.

    max_sample_size : int, default=10000000
        Maximum number of candidate points to consider when optimizing the
        acquisition function. The surrogate model predicts scores for these
        candidates, and the best one according to the acquisition function
        is selected for evaluation.

        Larger values improve acquisition optimization quality but increase
        memory usage and computation time. For most problems, the default
        is more than sufficient. Reduce this if memory is a concern.

    sampling : dict, default={"random": 1000000}
        Configuration for how candidate points are generated for acquisition
        function optimization. The key specifies the sampling strategy and
        the value specifies the number of samples.

        Currently supported: ``{"random": N}`` for uniform random sampling
        of N candidate points from the search space.

        Example::

            sampling = {"random": 500000}  # Fewer candidates, faster

    replacement : bool, default=True
        Whether to sample candidate points with replacement when generating
        candidates for acquisition function optimization. When ``True``,
        the same point can appear multiple times. When ``False``, each
        candidate is unique, ensuring diversity but potentially slower for
        very large sample sizes.

    Notes
    -----
    The algorithm exploits the Lipschitz continuity property:

    .. math::

        |f(x) - f(y)| \\leq L \\cdot \\|x - y\\|

    where L is the Lipschitz constant. This provides upper bounds on the
    function value at unobserved points:

    .. math::

        f(x) \\leq f(x_i) + \\hat{L} \\cdot \\|x - x_i\\| \\quad \\forall i

    The algorithm estimates :math:`\\hat{L}` from observed data and selects
    the point with the highest upper bound for the next evaluation. This
    gradually tightens the bounds and provably converges to the global
    optimum for Lipschitz continuous functions.

    For visual explanations and tuning guides, see
    the :ref:`Lipschitz Optimizer user guide <lipschitz>`.

    See Also
    --------
    DirectAlgorithm : Deterministic global search without Lipschitz constant.
    BayesianOptimizer : Surrogate-model-based approach using Gaussian Processes.
    RandomSearchOptimizer : Simple alternative without Lipschitz assumptions.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import LipschitzOptimizer

    >>> def smooth_function(para):
    ...     x, y = para["x"], para["y"]
    ...     return -(x**2 + y**2)  # Lipschitz continuous

    >>> search_space = {
    ...     "x": np.linspace(-5, 5, 100),
    ...     "y": np.linspace(-5, 5, 100),
    ... }

    >>> opt = LipschitzOptimizer(search_space)
    >>> opt.search(smooth_function, n_iter=100)
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
        )
