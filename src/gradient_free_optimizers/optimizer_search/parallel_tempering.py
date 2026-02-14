# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Parallel tempering using multiple annealers with temperature swapping."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers import (
    ParallelTemperingOptimizer as _ParallelTemperingOptimizer,
)
from ..search import Search


class ParallelTemperingOptimizer(_ParallelTemperingOptimizer, Search):
    r"""
    Ensemble of simulated annealers at different temperatures with periodic swapping.

    Parallel Tempering (also known as Replica Exchange Monte Carlo) runs multiple
    simulated annealing processes simultaneously at different temperatures. The
    key innovation is periodic swapping of temperatures between the parallel
    searches based on the Metropolis criterion. This allows solutions found at
    high temperatures (broad exploration) to be refined at low temperatures
    (focused exploitation), and vice versa.

    The algorithm maintains a population of searchers, each operating at a
    different temperature. Periodically, adjacent temperature levels attempt to
    swap their current positions. This mechanism helps overcome energy barriers
    that would trap a single simulated annealing run, making the algorithm
    particularly effective for rugged optimization landscapes.

    The algorithm is well-suited for:

    - Highly multimodal optimization problems
    - Problems with deep local optima that trap single-searcher methods
    - Scenarios where computational resources allow parallel evaluations
    - Sampling from complex probability distributions

    The `population` parameter controls the number of parallel searchers, while
    `n_iter_swap` determines how frequently temperature swaps are attempted.

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
    population : int, default=5
        Number of parallel simulated annealers, each running at a
        different temperature level. Temperatures are distributed
        geometrically between a minimum and maximum.

        - ``3-5``: Few temperature levels, limited coverage of the
          energy landscape.
        - ``5-10``: Good coverage for most problems (default region).
        - ``15-30``: Fine temperature resolution, better sampling of the
          energy landscape but higher computational cost.

        Each additional searcher adds one function evaluation per
        iteration. More temperature levels improve the probability of
        escaping deep local optima through the swap mechanism.
    n_iter_swap : int, default=5
        Number of iterations between temperature swap attempts. At each
        swap step, adjacent temperature levels may exchange their current
        positions based on the Metropolis criterion.

        - ``1-3``: Very frequent swaps, high communication between
          temperature levels. Can be noisy.
        - ``5-10``: Moderate swap frequency (default region). Allows each
          searcher to make progress between swaps.
        - ``20-50``: Infrequent swaps, each searcher explores more
          independently between exchanges.

        Lower values increase information flow between temperature levels
        but add overhead. The swap acceptance probability depends on the
        score difference and temperature gap between adjacent levels.

    Notes
    -----
    Parallel Tempering maintains ``population`` simulated annealers at
    temperatures :math:`T_1 < T_2 < \\ldots < T_n`. Every
    ``n_iter_swap`` iterations, adjacent pairs attempt to swap positions
    using the Metropolis criterion:

    .. math::

        P(\\text{swap}) = \\min\\left(1, \\exp\\left(
        (\\beta_i - \\beta_j)(E_i - E_j)\\right)\\right)

    where :math:`\\beta_i = 1/T_i` is the inverse temperature and
    :math:`E_i` is the score at temperature level i.

    This mechanism allows high-temperature explorers to pass promising
    positions to low-temperature searchers for refinement, overcoming
    energy barriers that would trap a single annealer.

    For visual explanations and tuning guides, see
    the :ref:`Parallel Tempering user guide <parallel_tempering>`.

    See Also
    --------
    SimulatedAnnealingOptimizer : Single-searcher annealing used as the base algorithm.
    ParticleSwarmOptimizer : Population-based search with velocities.
    EvolutionStrategyOptimizer : Population-based search using mutation and selection.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import ParallelTemperingOptimizer

    >>> def griewank(para):
    ...     x, y = para["x"], para["y"]
    ...     sum_sq = (x**2 + y**2) / 4000
    ...     prod_cos = np.cos(x / np.sqrt(1)) * np.cos(y / np.sqrt(2))
    ...     return -(sum_sq - prod_cos + 1)

    >>> search_space = {
    ...     "x": np.linspace(-600, 600, 1000),
    ...     "y": np.linspace(-600, 600, 1000),
    ... }

    >>> opt = ParallelTemperingOptimizer(search_space, population=10, n_iter_swap=10)
    >>> opt.search(griewank, n_iter=2000)
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
        population: int = 5,
        n_iter_swap: int = 5,
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
            population=population,
            n_iter_swap=n_iter_swap,
        )
