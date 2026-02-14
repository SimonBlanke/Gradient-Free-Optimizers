# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Spiral optimization using spiral movement patterns toward the best solution."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers import SpiralOptimization as _SpiralOptimization
from ..search import Search


class SpiralOptimization(_SpiralOptimization, Search):
    r"""
    Population-based optimizer using spiral movement patterns toward the best solution.

    Spiral Optimization Algorithm (SOA) is a metaheuristic inspired by spiral
    phenomena in nature, such as spiral galaxies and hurricanes. The algorithm
    maintains a population of search agents that move in spiral trajectories
    toward the current best solution. This spiral movement provides a natural
    balance between exploration (wider spiral paths) and exploitation (tighter
    convergence).

    At each iteration, particles rotate around and move toward the best-known
    position following a logarithmic spiral pattern. The decay rate controls
    how quickly the spiral tightens, determining the transition from global
    exploration to local refinement.

    The algorithm is well-suited for:

    - Continuous optimization problems
    - Multimodal functions with multiple local optima
    - Problems requiring smooth convergence behavior
    - Situations where controlled exploration-exploitation balance is needed

    The `decay_rate` is the key parameter: values below 1 cause the spiral
    to contract (convergent behavior), while values above 1 cause expansion
    (divergent exploration).

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
    population : int, default=10
        Number of search agents in the spiral population. Each agent follows
        a spiral trajectory toward the current best position.

        - ``5-10``: Small populations, fast per generation but risk of
          premature convergence.
        - ``15-30``: Good diversity-convergence balance for most problems.
        - ``50-100``: Thorough exploration, better for high-dimensional or
          highly multimodal problems.

        Each individual requires one function evaluation per generation, so
        total cost scales linearly with population size. As a rule of thumb,
        use larger populations for higher-dimensional or more multimodal
        problems.
    decay_rate : float, default=0.99
        Controls the spiral trajectory behavior. Determines how quickly
        the spiral radius contracts toward the best-known position.

        - ``0.8-0.9``: Fast contraction, rapid convergence but limited
          exploration.
        - ``0.95-0.99``: Moderate contraction, good balance (default
          region).
        - ``1.0``: No contraction, agents orbit at constant radius (pure
          exploration).
        - ``>1.0``: Expanding spirals, divergent exploration. Use with
          caution.

        The effective search radius at iteration t is proportional to
        ``decay_rate^t``. After 100 iterations with ``decay_rate=0.99``,
        the radius is ~37% of its initial value.

    Notes
    -----
    Each agent moves along a logarithmic spiral toward the current best
    position. The position update combines rotation and contraction:

    .. math::

        x_{t+1} = R(\\theta) \\cdot r \\cdot (x_t - x_{\\text{best}}) + x_{\\text{best}}

    where :math:`R(\\theta)` is a rotation matrix, :math:`r` is the
    ``decay_rate``, and :math:`x_{\\text{best}}` is the global best
    position. The rotation angle is randomized per dimension to avoid
    synchronization.

    The spiral movement naturally provides a balance between exploration
    (early iterations with wide orbits) and exploitation (later iterations
    with tight orbits around the best).

    For visual explanations and tuning guides, see
    the :ref:`Spiral Optimization user guide <spiral>`.

    See Also
    --------
    ParticleSwarmOptimizer : Population-based search using velocity-based movement.
    ParallelTemperingOptimizer : Multiple searchers with temperature-based exploration.
    DifferentialEvolutionOptimizer : Evolution using vector differences.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import SpiralOptimization

    >>> def sphere(para):
    ...     return -(para["x"] ** 2 + para["y"] ** 2)

    >>> search_space = {
    ...     "x": np.linspace(-10, 10, 100),
    ...     "y": np.linspace(-10, 10, 100),
    ... }

    >>> opt = SpiralOptimization(search_space, population=15, decay_rate=0.95)
    >>> opt.search(sphere, n_iter=300)
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
        population: int = 10,
        decay_rate: float = 0.99,
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
            decay_rate=decay_rate,
        )
