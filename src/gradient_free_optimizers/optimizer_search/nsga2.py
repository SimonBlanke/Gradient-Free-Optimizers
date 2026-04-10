"""NSGA-II multi-objective optimizer (public API wrapper)."""

from .._init_utils import get_default_initialize
from ..optimizers.pop_opt.nsga2 import _NSGA2Optimizer
from ..search import Search


class NSGA2Optimizer(_NSGA2Optimizer, Search):
    """Multi-objective optimizer using Non-dominated Sorting Genetic Algorithm II.

    NSGA-II evolves a population of candidate solutions toward the
    Pareto front of a multi-objective problem. Selection favors
    solutions with better non-dominated rank (front number), and
    within the same front, solutions in less crowded regions are
    preferred (crowding distance).

    Offspring are created by uniform crossover and self-adaptive
    mutation inherited from the Individual class. After each
    generation (``population`` evaluations), parents and offspring
    are combined and reduced back to ``population`` size via
    non-dominated sorting.

    Designed for use with ``n_objectives > 1`` in the ``search()``
    call. Works with single-objective too but offers no advantage
    over simpler algorithms in that case.

    Parameters
    ----------
    search_space : dict[str, np.ndarray | tuple | list]
        Parameter names mapped to their value definitions.
    initialize : dict or None, default=None
        Initialization strategy. When None, uses a mix of grid,
        random, and vertex positions.
    constraints : list[callable] or None, default=None
        Constraint functions that receive a parameter dict and return
        True if the point is feasible.
    random_state : int or None, default=None
        Random seed for reproducibility.
    rand_rest_p : float, default=0
        Probability of random restart per iteration.
    nth_process : int or None, default=None
        Process index for parallel setups.
    population : int, default=20
        Number of individuals in the population.
    crossover_rate : float, default=0.9
        Probability that offspring is produced via crossover rather
        than cloning a single parent.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import NSGA2Optimizer
    >>>
    >>> def bi_objective(params):
    ...     x = params["x"]
    ...     return [-(x ** 2), -((x - 3) ** 2)]
    >>>
    >>> search_space = {"x": np.linspace(-5, 5, 100)}
    >>> opt = NSGA2Optimizer(search_space, population=20)
    >>> opt.search(bi_objective, n_iter=200, n_objectives=2,
    ...            verbosity=False)
    >>> print(opt.pareto_front[["x", "objective_0", "objective_1"]])
    """

    def __init__(
        self,
        search_space,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        population=20,
        crossover_rate=0.9,
        crossover_eta=20.0,
        mutation_eta=20.0,
    ):
        if initialize is None:
            initialize = get_default_initialize()

        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            population=population,
            crossover_rate=crossover_rate,
            crossover_eta=crossover_eta,
            mutation_eta=mutation_eta,
        )
