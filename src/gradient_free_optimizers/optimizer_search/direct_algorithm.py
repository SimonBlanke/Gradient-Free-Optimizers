# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""DIRECT algorithm using adaptive hyperrectangle subdivision."""

from typing import Literal

import pandas as pd

from .._init_utils import get_default_initialize
from ..optimizers import DirectAlgorithm as _DirectAlgorithm
from ..search import Search


class DirectAlgorithm(_DirectAlgorithm, Search):
    r"""
    Deterministic global optimizer using adaptive hyperrectangle subdivision.

    DIRECT (DIviding RECTangles) is a deterministic global optimization algorithm
    that systematically divides the search space into smaller hyperrectangles
    and samples their centers. The algorithm identifies "potentially optimal"
    rectangles based on a trade-off between the function value at the center
    and the size of the rectangle, balancing local refinement and global
    exploration without requiring derivatives or Lipschitz constants.

    Note: Unlike surrogate-model-based optimizers (Bayesian, Forest, TPE),
    DIRECT does not train a model. It uses deterministic subspace division
    with Lipschitz bounds for selection.

    At each iteration, DIRECT identifies hyperrectangles that could contain the
    global optimum (based on comparing function values and rectangle sizes),
    then divides these rectangles along their longest dimension. This creates
    a tree structure that adaptively refines the search in promising regions
    while maintaining coverage of the entire space.

    The algorithm is well-suited for:

    - Global optimization requiring deterministic guarantees
    - Lipschitz continuous functions (but doesn't require knowing the constant)
    - Low to moderate dimensional problems (typically < 10 dimensions)
    - Problems where both local and global search are important

    DIRECT provides a balance between exploration (large rectangles) and
    exploitation (rectangles with good function values) through its selection
    criterion, making it robust without requiring parameter tuning.

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

    warm_start : pd.DataFrame or None, default=None
        Previous optimization results to warm-start the algorithm. The
        DataFrame should contain columns matching the search space parameter
        names plus a "score" column. This allows continuing a previous
        optimization run.

    resolution : int, default=100
        Number of grid points for continuous dimensions specified as tuples
        (e.g., ``(0.0, 10.0)``). These are automatically discretized into
        this many evenly-spaced points. Has no effect on dimensions already
        specified as arrays.

        - ``50``: Coarse resolution, faster but less precise.
        - ``100``: Standard resolution (default).
        - ``500-1000``: Fine resolution for high-precision optimization.

    Notes
    -----
    DIRECT (DIviding RECTangles) partitions the search space into
    hyperrectangles and identifies "potentially optimal" rectangles
    that balance function value and rectangle size:

    A rectangle is potentially optimal if there exists a Lipschitz
    constant :math:`\\hat{L} > 0` such that it could contain the global
    minimum. This criterion naturally balances:

    - **Exploitation**: Rectangles with good function values (small
      :math:`f` at center).
    - **Exploration**: Large rectangles (high :math:`\\|d\\|` diameter).

    Potentially optimal rectangles are divided along their longest
    dimension, creating a tree structure that adaptively refines the
    most promising regions.

    Unlike Lipschitz optimization, DIRECT does not require knowing or
    estimating the Lipschitz constant explicitly.

    For visual explanations and tuning guides, see
    the :ref:`DIRECT Algorithm user guide <direct>`.

    See Also
    --------
    LipschitzOptimizer : Global optimization using Lipschitz constants.
    GridSearchOptimizer : Exhaustive fixed-grid search without adaptive subdivision.
    PatternSearch : Adaptive pattern-based search at a single resolution level.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import DirectAlgorithm

    >>> def multimodal(para):
    ...     x, y = para["x"], para["y"]
    ...     return -(np.sin(x) * np.sin(y) + 0.1 * (x**2 + y**2))

    >>> search_space = {
    ...     "x": np.linspace(-3, 3, 100),
    ...     "y": np.linspace(-3, 3, 100),
    ... }

    >>> opt = DirectAlgorithm(search_space)
    >>> opt.search(multimodal, n_iter=200)
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
        warm_start: pd.DataFrame = None,
        resolution: int = 100,
        # Legacy SMBO parameters - no-op, kept for backwards compatibility
        warm_start_smbo: pd.DataFrame = None,
        max_sample_size: int = 10000000,
        sampling: dict[str, int] = None,
        replacement: bool = True,
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
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            warm_start=warm_start,
            resolution=resolution,
            warm_start_smbo=warm_start_smbo,
            max_sample_size=max_sample_size,
            sampling=sampling,
            replacement=replacement,
        )
