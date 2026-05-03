# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Automatic optimizer with adaptive algorithm selection."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers import AutoOptimizer as _AutoOptimizer
from ..optimizers.pop_opt._selection_strategy import SelectionStrategy
from ..search import Search


class AutoOptimizer(_AutoOptimizer, Search):
    r"""
    Automatic optimizer that selects among a portfolio of algorithms at runtime.

    AutoOptimizer maintains a heterogeneous population of optimization algorithms
    and adaptively allocates iteration budget to the most effective one. The
    default strategy measures wall-clock time per iteration and uses a UCB1 bandit
    to balance exploitation (giving more iterations to efficient optimizers) with
    exploration (trying underused optimizers periodically).

    The key design principle is time-awareness. An optimizer that produces small
    improvements very quickly (like HillClimbing on cheap objectives) scores higher
    than one that produces larger improvements but takes much longer per step (like
    BayesianOptimizer with its surrogate model overhead). This makes AutoOptimizer
    suitable as a drop-in default for packages that need gradient-free optimization
    without requiring the user to choose an algorithm.

    The default portfolio contains three complementary algorithms:

    - **RandomSearch**: unbiased exploration, zero overhead
    - **HillClimbing**: fast local refinement from the best known position
    - **RepulsingHillClimbing**: local search that escapes previously visited regions

    Users can override the portfolio and strategy for more specialized use cases,
    including surrogate-based optimizers for expensive objective functions.

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
    portfolio : list of optimizer classes, optional
        Optimizer types to include in the portfolio. Each class is
        instantiated once with the same search_space and constraints.
        Defaults to ``[RandomSearchOptimizer, HillClimbingOptimizer,
        RepulsingHillClimbingOptimizer]``.

        For expensive objective functions (evaluation time > 1 second),
        consider adding surrogate-based optimizers::

            from gradient_free_optimizers.optimizers import (
                ForestOptimizer, BayesianOptimizer,
            )
            portfolio = [
                RandomSearchOptimizer,
                HillClimbingOptimizer,
                ForestOptimizer,
            ]

        The strategy's time-weighted selection will automatically prefer
        faster algorithms when the objective is cheap and allow slower
        but more sample-efficient algorithms when the objective is
        expensive.
    strategy : SelectionStrategy, optional
        Strategy controlling which optimizer gets each iteration.
        Defaults to ``DefaultStrategy(min_rounds=3, exploration=0.5)``,
        which uses a time-weighted UCB1 bandit after an initial
        round-robin warmup phase.

    Notes
    -----
    The default strategy operates in two phases:

    **Warmup phase**: Each sub-optimizer receives ``min_rounds``
    evaluations in round-robin order to establish baseline performance
    data.

    **Adaptive phase**: The strategy computes a UCB1 score for each
    sub-optimizer:

    .. math::

        \text{UCB}_i = \frac{\sum \Delta_i}{T_i}
        + c \sqrt{\frac{\ln N}{n_i}}

    where :math:`\Delta_i` is the total improvement achieved by optimizer
    *i* (sum of positive score deltas when it finds a new personal best),
    :math:`T_i` is the total wall-clock time consumed by optimizer *i*,
    :math:`N` is the total number of evaluations across all optimizers,
    :math:`n_i` is the number of evaluations for optimizer *i*, and
    :math:`c` is the exploration coefficient.

    The ratio :math:`\sum \Delta_i / T_i` is the time-weighted efficiency:
    improvement per second. This naturally penalizes computationally
    expensive optimizers when the objective function is cheap, and allows
    them to compete when the objective function is expensive.

    See Also
    --------
    HillClimbingOptimizer : Fast local search, used in the default portfolio.
    RandomSearchOptimizer : Unbiased exploration, used in the default portfolio.
    RepulsingHillClimbingOptimizer : Local search with repulsion from visited regions.
    ParallelTemperingOptimizer : Population-based search with temperature swapping.

    Examples
    --------
    Basic usage with default portfolio and strategy:

    >>> import numpy as np
    >>> from gradient_free_optimizers import AutoOptimizer

    >>> def sphere(para):
    ...     return -(para["x"] ** 2 + para["y"] ** 2)

    >>> search_space = {
    ...     "x": np.linspace(-10, 10, 100),
    ...     "y": np.linspace(-10, 10, 100),
    ... }

    >>> opt = AutoOptimizer(search_space)
    >>> opt.search(sphere, n_iter=200)

    Using the ask/tell interface for manual control:

    >>> opt = AutoOptimizer(search_space)
    >>> opt.setup_search(sphere, n_iter=200)
    >>> for _ in range(200):
    ...     params = opt.ask()
    ...     score = sphere(params)
    ...     opt.tell(params, score)
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
        portfolio: list = None,
        strategy: SelectionStrategy = None,
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
            portfolio=portfolio,
            strategy=strategy,
        )
