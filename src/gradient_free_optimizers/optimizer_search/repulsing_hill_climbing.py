# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Repulsing hill climbing that increases step size to escape local optima."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers import (
    RepulsingHillClimbingOptimizer as _RepulsingHillClimbingOptimizer,
)
from ..search import Search


class RepulsingHillClimbingOptimizer(_RepulsingHillClimbingOptimizer, Search):
    r"""
    Hill climbing variant that increases step size when stuck to escape local optima.

    Repulsing Hill Climbing is an adaptive variant of hill climbing that dynamically
    increases the search radius (epsilon) when no improvement is found. This
    "repulsion" mechanism helps the optimizer escape local optima by taking
    progressively larger steps when stuck, effectively being "pushed away" from
    the current region. Once a better solution is found, the step size resets
    to its original value for fine-grained local search.

    The algorithm is well-suited for:

    - Optimization landscapes with isolated local optima
    - Problems where the distance between optima is unknown
    - Scenarios requiring automatic adaptation of search radius
    - Balancing local exploitation and global exploration without manual tuning

    The `repulsion_factor` parameter controls how aggressively the step size
    increases when stuck. A factor of 5 means epsilon is multiplied by 5 each
    time no improvement is found. Higher values lead to faster escape from
    local optima but may overshoot good regions.

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
    repulsion_factor : float, default=5
        Multiplicative factor applied to ``epsilon`` each time no improvement
        is found. The step size grows geometrically while stuck, enabling
        escape from local optima basins of increasing size.

        - ``1.0-2.0``: Gentle increase, slow escape from local optima.
        - ``3.0-5.0``: Moderate increase, good default range.
        - ``10.0+``: Aggressive increase, quickly jumps to distant regions
          but may overshoot good areas.

        After an improvement is found, epsilon resets to its original value.
        The effective step size after k stuck iterations is
        ``epsilon * repulsion_factor^k``.

    Notes
    -----
    The algorithm extends standard Hill Climbing with an adaptive step
    size mechanism:

    .. math::

        \\epsilon_{\\text{eff}} = \\epsilon \\cdot r^{k}

    where r is the ``repulsion_factor`` and k is the number of consecutive
    iterations without improvement. When an improvement is found, k resets
    to 0.

    This creates an automatic exploration-exploitation cycle: the optimizer
    performs fine-grained local search when improving, then broadens its
    search when stuck, and returns to local search after finding a new
    promising region.

    For visual explanations and tuning guides, see
    the :ref:`Repulsing Hill Climbing user guide <repulsing_hill_climbing>`.

    See Also
    --------
    HillClimbingOptimizer : Base hill climbing without adaptive step size.
    RandomRestartHillClimbingOptimizer : Periodic random restarts.
    SimulatedAnnealingOptimizer : Escapes local optima through probabilistic acceptance.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import RepulsingHillClimbingOptimizer

    >>> def ackley(para):
    ...     x, y = para["x"], para["y"]
    ...     return -(-20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    ...              - np.exp(0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))
    ...              + np.e + 20)

    >>> search_space = {
    ...     "x": np.linspace(-5, 5, 100),
    ...     "y": np.linspace(-5, 5, 100),
    ... }

    >>> opt = RepulsingHillClimbingOptimizer(search_space, repulsion_factor=3)
    >>> opt.search(ackley, n_iter=200)
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
        repulsion_factor: float = 5,
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
            repulsion_factor=repulsion_factor,
        )
