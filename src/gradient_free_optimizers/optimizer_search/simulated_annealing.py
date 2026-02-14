# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Simulated annealing inspired by the metallurgical annealing process."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers import (
    SimulatedAnnealingOptimizer as _SimulatedAnnealingOptimizer,
)
from ..search import Search


class SimulatedAnnealingOptimizer(_SimulatedAnnealingOptimizer, Search):
    r"""
    Probabilistic optimizer inspired by the annealing process in metallurgy.

    Simulated Annealing is a classic metaheuristic that mimics the physical process
    of heating and slowly cooling a material to reduce defects. The algorithm starts
    with a high "temperature" that allows accepting worse solutions with high
    probability, enabling broad exploration. As the temperature decreases according
    to the annealing schedule, the acceptance probability for worse solutions
    decreases, and the algorithm gradually focuses on exploitation.

    The acceptance probability follows the Metropolis criterion: worse solutions
    are accepted with probability exp(-delta/T), where delta is the score
    difference and T is the current temperature. This allows the algorithm to
    escape local optima early in the search while converging to good solutions
    later.

    The algorithm is well-suited for:

    - Combinatorial optimization problems
    - Multimodal functions with many local optima
    - Problems where a good balance of exploration and exploitation is needed
    - Situations where solution quality matters more than speed

    The `annealing_rate` controls how fast the temperature decreases. Values close
    to 1.0 cool slowly (more exploration), while smaller values cool faster
    (quicker convergence but risk of premature convergence).

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
    annealing_rate : float, default=0.97
        Multiplicative cooling factor applied to the temperature at each
        iteration. The temperature at iteration t is
        ``start_temp * annealing_rate^t``.

        - ``0.8-0.9``: Fast cooling, quick convergence but limited
          exploration. Good for unimodal problems.
        - ``0.95-0.99``: Slow cooling, thorough exploration before
          convergence. Better for multimodal landscapes.
        - ``0.999``: Very slow cooling, extensive exploration. Use with
          high ``n_iter``.

        Example: With ``start_temp=1`` and ``annealing_rate=0.97``, the
        temperature after 100 iterations is approximately 0.048, and
        after 200 iterations approximately 0.002.
    start_temp : float, default=1
        Initial temperature controlling the acceptance probability of worse
        solutions at the start of the search. Higher temperatures allow
        more exploration initially.

        - ``0.1-0.5``: Low initial temperature, conservative exploration.
        - ``1.0``: Moderate starting temperature (default).
        - ``5.0-20.0``: High temperature, strong initial exploration phase.

        The temperature interacts with the score scale of your objective
        function. If scores span a large range, higher ``start_temp`` may
        be needed to enable meaningful exploration.

    Notes
    -----
    The acceptance decision follows the Metropolis criterion:

    .. math::

        P(\\text{accept}) = \\begin{cases}
        1 & \\text{if } \\Delta f > 0 \\\\
        \\exp(\\Delta f / T) & \\text{if } \\Delta f \\leq 0
        \\end{cases}

    where :math:`\\Delta f = f(x_{\\text{new}}) - f(x_{\\text{current}})`
    and :math:`T = T_0 \\cdot r^t` is the temperature at iteration t, with
    :math:`T_0` = ``start_temp`` and :math:`r` = ``annealing_rate``.

    At high temperatures, almost all moves are accepted (exploration).
    As temperature decreases, only improving moves are accepted
    (exploitation). This provides a natural transition from global to
    local search.

    For visual explanations and tuning guides, see
    the :ref:`Simulated Annealing user guide <simulated_annealing>`.

    See Also
    --------
    StochasticHillClimbingOptimizer : Constant-probability acceptance.
    ParallelTemperingOptimizer : Multiple annealers at different temperatures.
    RandomAnnealingOptimizer : Temperature-based step size control.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import SimulatedAnnealingOptimizer

    >>> def rosenbrock(para):
    ...     x, y = para["x"], para["y"]
    ...     return -((1 - x) ** 2 + 100 * (y - x ** 2) ** 2)

    >>> search_space = {
    ...     "x": np.linspace(-2, 2, 100),
    ...     "y": np.linspace(-1, 3, 100),
    ... }

    >>> opt = SimulatedAnnealingOptimizer(
    ...     search_space, annealing_rate=0.98, start_temp=10
    ... )
    >>> opt.search(rosenbrock, n_iter=1000)
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
        annealing_rate: float = 0.97,
        start_temp: float = 1,
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
