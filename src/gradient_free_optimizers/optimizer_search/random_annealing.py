# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Random annealing using temperature to control the search radius."""

from typing import Literal

from ..optimizers import RandomAnnealingOptimizer as _RandomAnnealingOptimizer
from ..search import Search


class RandomAnnealingOptimizer(_RandomAnnealingOptimizer, Search):
    """
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
        The probability of a random iteration during the search process.
    epsilon : float
        The base step-size for the search. This is multiplied by the current
        temperature to determine the actual step size.
    distribution : str
        The type of distribution to sample neighbors from. Options are
        "normal", "laplace", "gumbel", or "logistic".
    n_neighbours : int
        The number of neighbours to sample and evaluate before moving to the best
        of those neighbours.
    annealing_rate : float
        The rate at which temperature decreases each iteration. Values close
        to 1.0 result in slower cooling. Default is 0.98.
    start_temp : float
        The initial temperature that multiplies the step size. Higher values
        result in larger initial exploration. Default is 10.

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
        ] = {"grid": 4, "random": 2, "vertices": 4},
        constraints: list[callable] = [],
        random_state: int = None,
        rand_rest_p: float = 0,
        nth_process: int = None,
        epsilon: float = 0.03,
        distribution: Literal["normal", "laplace", "gumbel", "logistic"] = "normal",
        n_neighbours: int = 3,
        annealing_rate=0.98,
        start_temp=10,
    ):
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
