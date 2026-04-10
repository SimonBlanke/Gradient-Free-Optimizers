"""SMS-EMOA multi-objective optimizer (public API wrapper)."""

from .._init_utils import get_default_initialize
from ..optimizers.pop_opt.sms_emoa import _SMSEMOAOptimizer
from ..search import Search


class SMSEMOAOptimizer(_SMSEMOAOptimizer, Search):
    """Multi-objective optimizer using hypervolume-based selection.

    SMS-EMOA is a steady-state algorithm: each iteration generates one
    offspring, adds it to the population, and removes the individual
    whose loss causes the smallest decrease in the dominated hypervolume.
    This drives the population toward a Pareto front that maximizes
    the covered hypervolume.

    Compared to NSGA-II (crowding distance) and MOEA/D (decomposition),
    SMS-EMOA has the strongest theoretical convergence guarantee because
    the hypervolume indicator is the only unary quality indicator that
    is strictly monotone with respect to Pareto dominance.

    The steady-state design means the population is updated after every
    single evaluation, giving the algorithm the most up-to-date
    information for each subsequent decision.

    Parameters
    ----------
    search_space : dict[str, np.ndarray | tuple | list]
        Parameter names mapped to their value definitions.
    initialize : dict or None, default=None
        Initialization strategy.
    constraints : list[callable] or None, default=None
        Constraint functions returning True for feasible points.
    random_state : int or None, default=None
        Random seed for reproducibility.
    rand_rest_p : float, default=0
        Probability of random restart per iteration.
    nth_process : int or None, default=None
        Process index for parallel setups.
    population : int, default=20
        Population size. Each iteration adds one offspring and removes
        one individual, keeping the population constant.
    crossover_rate : float, default=0.9
        Probability of crossover vs. cloning a parent.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import SMSEMOAOptimizer
    >>>
    >>> def bi_objective(params):
    ...     x = params["x"]
    ...     return [-(x ** 2), -((x - 3) ** 2)]
    >>>
    >>> search_space = {"x": np.linspace(-5, 5, 100)}
    >>> opt = SMSEMOAOptimizer(search_space, population=20)
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
        )
