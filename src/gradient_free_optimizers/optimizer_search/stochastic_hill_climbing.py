# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Stochastic hill climbing that probabilistically accepts worse solutions."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers import (
    StochasticHillClimbingOptimizer as _StochasticHillClimbingOptimizer,
)
from ..search import Search


class StochasticHillClimbingOptimizer(_StochasticHillClimbingOptimizer, Search):
    r"""Hill climbing variant that accepts worse solutions to escape local optima.

    Stochastic Hill Climbing extends the basic hill climbing algorithm by introducing
    a probability of accepting solutions that are worse than the current one. This
    stochastic acceptance mechanism helps the optimizer escape local optima and
    explore a broader region of the search space. Unlike standard hill climbing,
    which always moves to better positions, this variant can temporarily accept
    inferior solutions, enabling it to "climb down" from local peaks.

    The algorithm is well-suited for:

    - Multimodal optimization problems with multiple local optima
    - Problems where standard hill climbing gets stuck frequently
    - Situations requiring a balance between local refinement and exploration
    - Optimization landscapes with many plateaus or ridges

    The `p_accept` parameter controls the probability of accepting worse solutions.
    Higher values increase exploration but may slow convergence, while lower values
    make the algorithm behave more like standard hill climbing. A value of 0.0
    reduces this to deterministic hill climbing.

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
    p_accept : float, default=0.5
        Probability of accepting a solution that is worse than the current
        one. This flat acceptance probability applies uniformly regardless
        of how much worse the candidate is.

        - ``0.0``: Never accept worse solutions (equivalent to standard
          Hill Climbing).
        - ``0.1-0.3``: Mild exploration, occasionally escapes shallow
          local optima.
        - ``0.5``: Balanced exploration and exploitation (default).
        - ``0.7-1.0``: Strong exploration, frequently accepts worse
          solutions, slow convergence.

        Unlike Simulated Annealing where acceptance depends on score
        difference and temperature, here the probability is constant
        throughout the search.

    Notes
    -----
    At each step, the best neighbor is found (same as Hill Climbing).
    If the best neighbor is better, it is always accepted. If worse, it
    is accepted with probability ``p_accept``:

    .. math::

        P(\\text{accept}) = \\begin{cases}
        1 & \\text{if } f(x_{\\text{neighbor}}) > f(x_{\\text{current}}) \\\\
        p_{\\text{accept}} & \\text{otherwise}
        \\end{cases}

    This is simpler than Simulated Annealing's Metropolis criterion but
    lacks the adaptive cooling behavior. The constant acceptance rate
    means exploration intensity does not change over time.

    For visual explanations and tuning guides, see
    the :ref:`Stochastic Hill Climbing user guide <stochastic_hill_climbing>`.

    See Also
    --------
    HillClimbingOptimizer : Deterministic variant that only accepts improvements.
    SimulatedAnnealingOptimizer : Score-dependent acceptance with temperature cooling.
    RepulsingHillClimbingOptimizer : Escapes local optima by increasing step size.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import StochasticHillClimbingOptimizer

    >>> def rastrigin(para):
    ...     x, y = para["x"], para["y"]
    ...     return -(20 + x**2 + y**2 - 10 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))

    >>> search_space = {
    ...     "x": np.linspace(-5.12, 5.12, 100),
    ...     "y": np.linspace(-5.12, 5.12, 100),
    ... }

    >>> opt = StochasticHillClimbingOptimizer(search_space, p_accept=0.3)
    >>> opt.search(rastrigin, n_iter=200)
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
        p_accept: float = 0.5,
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
            p_accept=p_accept,
        )
