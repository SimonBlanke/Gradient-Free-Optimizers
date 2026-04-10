"""MOEA/D multi-objective optimizer (public API wrapper)."""

from .._init_utils import get_default_initialize
from ..optimizers.pop_opt.moead import _MOEADOptimizer
from ..search import Search


class MOEADOptimizer(_MOEADOptimizer, Search):
    """Multi-objective optimizer using decomposition into scalar subproblems.

    MOEA/D assigns each individual in the population a weight vector
    that defines a scalar subproblem via Tchebycheff scalarization.
    Individuals only interact with their weight-vector neighbors,
    creating a structured information flow through the population.
    When an offspring improves a neighbor's subproblem, it replaces
    that neighbor immediately, allowing good solutions to propagate
    within a single generation.

    Compared to NSGA-II, MOEA/D tends to produce more uniformly
    distributed Pareto fronts, especially for convex front shapes.
    For concave fronts, the Tchebycheff scalarization still works
    (unlike weighted-sum approaches).

    Designed for use with ``n_objectives > 1`` in the ``search()``
    call. The ``population`` parameter determines the number of
    subproblems (and thus weight vectors). Larger populations give
    finer coverage of the Pareto front.

    Parameters
    ----------
    search_space : dict[str, np.ndarray | tuple | list]
        Parameter names mapped to their value definitions.
    initialize : dict or None, default=None
        Initialization strategy. When None, uses a mix of grid,
        random, and vertex positions.
    constraints : list[callable] or None, default=None
        Constraint functions returning True for feasible points.
    random_state : int or None, default=None
        Random seed for reproducibility.
    rand_rest_p : float, default=0
        Probability of random restart per iteration.
    nth_process : int or None, default=None
        Process index for parallel setups.
    population : int, default=20
        Number of subproblems (weight vectors). The actual number
        may be adjusted slightly to fit the simplex-lattice design
        for 3+ objectives.
    n_neighbors : int or None, default=None
        Neighborhood size for each weight vector. When None,
        defaults to ``max(3, population // 5)``. Smaller values
        increase exploitation, larger values increase exploration.
    crossover_rate : float, default=0.9
        Probability of crossover vs. cloning a parent.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import MOEADOptimizer
    >>>
    >>> def bi_objective(params):
    ...     x = params["x"]
    ...     return [-(x ** 2), -((x - 3) ** 2)]
    >>>
    >>> search_space = {"x": np.linspace(-5, 5, 100)}
    >>> opt = MOEADOptimizer(search_space, population=20)
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
        n_neighbors=None,
        crossover_rate=0.9,
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
            n_neighbors=n_neighbors,
            crossover_rate=crossover_rate,
        )
