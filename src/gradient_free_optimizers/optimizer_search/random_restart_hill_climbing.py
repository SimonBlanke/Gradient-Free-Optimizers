# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Random restart hill climbing with periodic restarts from random positions."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers import (
    RandomRestartHillClimbingOptimizer as _RandomRestartHillClimbingOptimizer,
)
from ..search import Search


class RandomRestartHillClimbingOptimizer(_RandomRestartHillClimbingOptimizer, Search):
    """
    Hill climbing variant that periodically restarts from random positions.

    Random Restart Hill Climbing addresses the local optima problem by periodically
    resetting the search to a new random position after a fixed number of iterations.
    This simple yet effective strategy allows the algorithm to explore multiple
    regions of the search space, increasing the probability of finding the global
    optimum. The best solution found across all restarts is retained.

    The algorithm is well-suited for:

    - Multimodal optimization problems with many local optima
    - Problems where the location of the global optimum is unknown
    - Scenarios where multiple independent searches are beneficial
    - Situations requiring a simple, parallelizable approach

    The `n_iter_restart` parameter controls the frequency of restarts. Shorter
    intervals lead to more exploration but less exploitation of each local region,
    while longer intervals allow more thorough local search before restarting.
    The optimal value depends on the problem's landscape and the expected basin
    of attraction size.

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

    epsilon : float, default=0.03
        Step size for generating neighbor positions, expressed as a fraction
        of each dimension's range. Controls how far the optimizer looks from
        the current position when sampling neighbors.

        - ``0.01-0.02``: Fine-grained local search, slow convergence.
        - ``0.03-0.05``: Moderate step size, good default range.
        - ``0.1-0.3``: Large steps, broader exploration.
        - ``0.5-1.0``: Very large steps, nearly global jumps.

        Example: For a dimension with ``np.linspace(0, 100, 1000)``
        (range = 100):

        - ``epsilon=0.03`` leads to neighbors within ~3 units
        - ``epsilon=0.1`` leads to neighbors within ~10 units

        Smaller values are better for fine-tuning near a known good solution.
        Larger values help escape local optima but may overshoot narrow peaks.

    distribution : {"normal", "laplace", "gumbel", "logistic"}, default="normal"
        Probability distribution used to sample neighbor offsets. Each
        distribution produces different exploration patterns:

        - ``"normal"``: Gaussian distribution. Most neighbors are close to
          the current position with rare far jumps. Best general-purpose
          choice.
        - ``"laplace"``: Sharper peak than normal with heavier tails. Good
          for landscapes where occasional large jumps help.
        - ``"gumbel"``: Asymmetric, skewed distribution. Can bias
          exploration in one direction.
        - ``"logistic"``: Similar to normal but with slightly heavier tails.
          A middle ground between normal and Laplace.

        The distribution interacts with ``epsilon``: heavy-tailed
        distributions (Laplace) effectively increase the chance of large
        steps beyond what ``epsilon`` alone suggests.

    n_neighbours : int, default=3
        Number of neighbor positions to sample and evaluate per iteration.
        The optimizer moves to the best among these neighbors (if it
        improves on the current position).

        - ``1``: Minimal sampling, fast iterations but may miss good
          directions. Good for very cheap objective functions.
        - ``3-5``: Moderate sampling, good balance of quality and speed.
        - ``10-20``: Thorough neighborhood evaluation, better directional
          choices but more function evaluations per step.

        Higher values act like a local beam search, giving the optimizer
        more information about the local landscape at each iteration.

    n_iter_restart : int, default=10
        Number of iterations between random restarts. After this many
        iterations of hill climbing, the optimizer jumps to a new random
        position and begins climbing again. The best solution across all
        restarts is retained.

        - ``5-10``: Frequent restarts, many short climbs. Good for
          landscapes with many shallow local optima.
        - ``20-50``: Moderate restart frequency. Good balance for most
          problems.
        - ``100+``: Infrequent restarts, thorough local search per
          restart. Good for landscapes with wide basins of attraction.

        The optimal value depends on how many iterations the hill climber
        typically needs to reach a local optimum. With a total budget of
        ``n_iter`` iterations, you get approximately
        ``n_iter / n_iter_restart`` independent restarts.

    Notes
    -----
    The algorithm alternates between hill climbing phases and random
    restarts:

    1. Perform standard hill climbing for ``n_iter_restart`` iterations.
    2. Jump to a uniformly random position in the search space.
    3. Repeat until the iteration budget is exhausted.

    The best solution found across all restart cycles is retained. With
    k restarts on a landscape with m local optima, the probability of
    finding the global optimum is approximately :math:`1 - (1 - 1/m)^k`,
    assuming roughly equal basin sizes.

    For visual explanations and tuning guides, see
    the :ref:`Random Restart Hill Climbing user guide <random_restart>`.

    See Also
    --------
    HillClimbingOptimizer : Base hill climbing without restarts.
    RepulsingHillClimbingOptimizer : Escapes local optima via step size increase.
    RandomAnnealingOptimizer : Temperature-based step size reduction.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import RandomRestartHillClimbingOptimizer

    >>> def schwefel(para):
    ...     x, y = para["x"], para["y"]
    ...     return -(418.9829 * 2 - x * np.sin(np.sqrt(abs(x)))
    ...              - y * np.sin(np.sqrt(abs(y))))

    >>> search_space = {
    ...     "x": np.linspace(-500, 500, 1000),
    ...     "y": np.linspace(-500, 500, 1000),
    ... }

    >>> opt = RandomRestartHillClimbingOptimizer(search_space, n_iter_restart=20)
    >>> opt.search(schwefel, n_iter=500)
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
        epsilon: float = 0.03,
        distribution: Literal["normal", "laplace", "gumbel", "logistic"] = "normal",
        n_neighbours: int = 3,
        n_iter_restart: int = 10,
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
            epsilon=epsilon,
            distribution=distribution,
            n_neighbours=n_neighbours,
            n_iter_restart=n_iter_restart,
        )
