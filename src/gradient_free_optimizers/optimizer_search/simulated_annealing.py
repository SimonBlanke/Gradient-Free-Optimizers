# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import List, Dict, Literal, Union

from ..search import Search
from ..optimizers import (
    SimulatedAnnealingOptimizer as _SimulatedAnnealingOptimizer,
)


class SimulatedAnnealingOptimizer(_SimulatedAnnealingOptimizer, Search):
    """
    A class implementing **simulated annealing** for the public API.
    Inheriting from the `Search`-class to get the `search`-method and from
    the `SimulatedAnnealingOptimizer`-backend to get the underlying algorithm.

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
        The step-size for the climbing.
    distribution : str
        The type of distribution to sample from.
    n_neighbours : int
        The number of neighbours to sample and evaluate before moving to the best
        of those neighbours.
    annealing_rate : float
        The rate at which the temperature is annealed.
    start_temp : float
        The initial temperature.
    """

    def __init__(
        self,
        search_space: Dict[str, list],
        initialize: Dict[
            Literal["grid", "vertices", "random", "warm_start"],
            Union[int, list[dict]],
        ] = {"grid": 4, "random": 2, "vertices": 4},
        constraints: List[callable] = [],
        random_state: int = None,
        rand_rest_p: float = 0,
        nth_process: int = None,
        epsilon: float = 0.03,
        distribution: Literal[
            "normal", "laplace", "gumbel", "logistic"
        ] = "normal",
        n_neighbours: int = 3,
        annealing_rate: float = 0.97,
        start_temp: float = 1,
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
