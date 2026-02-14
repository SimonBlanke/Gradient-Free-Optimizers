# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Random annealing using temperature to control the search radius."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers import RandomAnnealingOptimizer as _RandomAnnealingOptimizer
from ..search import Search


class RandomAnnealingOptimizer(_RandomAnnealingOptimizer, Search):
    r"""
    Annealing optimizer that uses temperature to control the search radius.

    Random Annealing is a variant of simulated annealing that uses the temperature
    parameter differently. Instead of controlling the acceptance probability of
    worse solutions, the temperature directly affects the step size (epsilon) of
    the search. At high temperatures, the optimizer takes large random steps
    across the search space. As the temperature decreases, the steps become
    smaller, allowing for finer local search around promising regions.

    This approach provides a natural transition from global exploration to local
    exploitation without the need for explicit acceptance probability calculations.
    The algorithm always moves to the best neighbor found, but the neighborhood
    size shrinks over time according to the annealing schedule.

    The algorithm is well-suited for:

    - Problems requiring extensive initial exploration
    - Optimization landscapes with large basins of attraction
    - Scenarios where controlling step size is more intuitive than acceptance
      probability
    - Problems where the scale of the search space varies

    The `start_temp` parameter controls the initial search radius multiplier,
    while `annealing_rate` determines how quickly this radius shrinks.

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

    annealing_rate : float, default=0.98
        Multiplicative cooling factor applied to the temperature each
        iteration. The temperature at iteration t is
        ``start_temp * annealing_rate^t``, and the effective step size is
        ``epsilon * temperature``.

        - ``0.9-0.95``: Fast cooling, transitions quickly from exploration
          to exploitation.
        - ``0.97-0.99``: Moderate cooling, good for most problems.
        - ``0.999``: Very slow cooling, extensive exploration phase. Pair
          with high ``n_iter``.

    start_temp : float, default=10
        Initial temperature that multiplies the base step size ``epsilon``.
        The effective step size at the start is ``epsilon * start_temp``.

        - ``1.0-5.0``: Moderate initial exploration radius.
        - ``10.0``: Good default for broad initial exploration.
        - ``50.0-100.0``: Very broad initial exploration, useful for large
          search spaces.

        Example: With ``epsilon=0.03`` and ``start_temp=10``, the initial
        effective step size is 0.3 (30% of dimension range), which shrinks
        to ``epsilon=0.03`` as temperature approaches 1.

    Notes
    -----
    Unlike standard Simulated Annealing which uses temperature for
    acceptance probability, Random Annealing uses temperature to control
    the search radius:

    .. math::

        \\epsilon_{\\text{eff}}(t) = \\epsilon \\cdot T_0 \\cdot r^t

    where :math:`T_0` = ``start_temp`` and :math:`r` = ``annealing_rate``.

    The algorithm always moves to the best neighbor found (greedy), but
    the neighborhood shrinks over time. Early iterations explore broadly
    across the search space; later iterations focus on fine-tuning in a
    small region around the current best.

    For visual explanations and tuning guides, see
    the :ref:`Random Annealing user guide <random_annealing>`.

    See Also
    --------
    SimulatedAnnealingOptimizer : Temperature-based acceptance probability.
    RandomRestartHillClimbingOptimizer : Escapes local optima through periodic restarts.
    HillClimbingOptimizer : Fixed step size without annealing schedule.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import RandomAnnealingOptimizer

    >>> def sphere(para):
    ...     return -(para["x"] ** 2 + para["y"] ** 2 + para["z"] ** 2)

    >>> search_space = {
    ...     "x": np.linspace(-100, 100, 1000),
    ...     "y": np.linspace(-100, 100, 1000),
    ...     "z": np.linspace(-100, 100, 1000),
    ... }

    >>> opt = RandomAnnealingOptimizer(search_space, start_temp=20, annealing_rate=0.99)
    >>> opt.search(sphere, n_iter=500)
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
        annealing_rate=0.98,
        start_temp=10,
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
            annealing_rate=annealing_rate,
            start_temp=start_temp,
        )
