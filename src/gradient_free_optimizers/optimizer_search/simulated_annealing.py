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
    """
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
        The search space to explore. A dictionary with parameter
        names as keys and a numpy array as values.
    initialize : dict[str, int]
        The method to generate initial positions. A dictionary with
        the following key literals and the corresponding value type:
        {"grid": int, "vertices": int, "random": int, "warm_start": list[dict]}
    constraints : list[callable]
        A list of constraints, where each constraint is a callable.
        The callable returns `True` or `False` dependend on the input parameters.
    random_state : None, int
        If None, create a new random state. If int, create a new random state
        seeded with the value.
    rand_rest_p : float
        The probability of a random iteration during the the search process.
    epsilon : float
        The step-size for generating neighbor solutions. Controls how far
        from the current position neighbors are sampled.
    distribution : str
        The type of distribution to sample neighbors from. Options are
        "normal", "laplace", "gumbel", or "logistic".
    n_neighbours : int
        The number of neighbours to sample and evaluate before selecting
        the best candidate for the acceptance decision.
    annealing_rate : float
        The multiplicative factor applied to temperature each iteration.
        Values should be between 0 and 1. Higher values (e.g., 0.99) result
        in slower cooling. Default is 0.97.
    start_temp : float
        The initial temperature. Higher values allow more exploration at the
        start. Default is 1.

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
