# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import List, Dict, Literal, Union

from ..search import Search
from ..optimizers import (
    EvolutionStrategyOptimizer as _EvolutionStrategyOptimizer,
)


class EvolutionStrategyOptimizer(_EvolutionStrategyOptimizer, Search):
    """
    A class implementing the **evolution strategy** for the public API.
    Inheriting from the `Search`-class to get the `search`-method and from
    the `EvolutionStrategyOptimizer`-backend to get the underlying algorithm.

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
        The number of individuals in the population.
    offspring : int
        The number of offspring to generate in each generation.
    replace_parents : bool
        If True, the parents are replaced with the offspring in the next
        generation. If False, the parents are kept in the next generation and the
        offspring are added to the population.
    mutation_rate : float
        The mutation rate for the mutation operator.
    crossover_rate : float
        The crossover rate for the crossover operator.
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
        population=10,
        offspring=20,
        replace_parents=False,
        mutation_rate=0.7,
        crossover_rate=0.3,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            population=population,
            offspring=offspring,
            replace_parents=replace_parents,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
        )
