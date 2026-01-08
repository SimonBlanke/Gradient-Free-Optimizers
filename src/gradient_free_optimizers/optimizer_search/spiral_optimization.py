# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import Literal

from ..optimizers import SpiralOptimization as _SpiralOptimization
from ..search import Search


class SpiralOptimization(_SpiralOptimization, Search):
    """
    Population-based optimizer using spiral movement patterns toward the best solution.

    Spiral Optimization Algorithm (SOA) is a metaheuristic inspired by spiral
    phenomena in nature, such as spiral galaxies and hurricanes. The algorithm
    maintains a population of search agents that move in spiral trajectories
    toward the current best solution. This spiral movement provides a natural
    balance between exploration (wider spiral paths) and exploitation (tighter
    convergence).

    At each iteration, particles rotate around and move toward the best-known
    position following a logarithmic spiral pattern. The decay rate controls
    how quickly the spiral tightens, determining the transition from global
    exploration to local refinement.

    The algorithm is well-suited for:

    - Continuous optimization problems
    - Multimodal functions with multiple local optima
    - Problems requiring smooth convergence behavior
    - Situations where controlled exploration-exploitation balance is needed

    The `decay_rate` is the key parameter: values below 1 cause the spiral
    to contract (convergent behavior), while values above 1 cause expansion
    (divergent exploration).

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
    population : int
        The number of search agents in the population. More agents provide
        better coverage but require more function evaluations. Default is 10.
    decay_rate : float
        Controls the spiral trajectory behavior. Values below 1.0 cause
        convergent spiraling toward the best position (recommended for
        exploitation). Values above 1.0 cause divergent spiraling (more
        exploration). Default is 0.99.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import SpiralOptimization

    >>> def sphere(para):
    ...     return -(para["x"] ** 2 + para["y"] ** 2)

    >>> search_space = {
    ...     "x": np.linspace(-10, 10, 100),
    ...     "y": np.linspace(-10, 10, 100),
    ... }

    >>> opt = SpiralOptimization(search_space, population=15, decay_rate=0.95)
    >>> opt.search(sphere, n_iter=300)
    """

    def __init__(
        self,
        search_space: dict[str, list],
        initialize: dict[
            Literal["grid", "vertices", "random", "warm_start"],
            int | list[dict],
        ] = {"grid": 4, "random": 2, "vertices": 4},
        constraints: list[callable] = [],
        random_state: int = None,
        rand_rest_p: float = 0,
        nth_process: int = None,
        population: int = 10,
        decay_rate: float = 0.99,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            population=population,
            decay_rate=decay_rate,
        )
