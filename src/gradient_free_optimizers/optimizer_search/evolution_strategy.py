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
    Evolutionary optimizer focused on self-adaptive mutation for continuous domains.

    Evolution Strategy (ES) is an evolutionary algorithm originally designed
    for continuous parameter optimization. Unlike genetic algorithms that
    emphasize crossover, ES primarily relies on mutation as the main variation
    operator. The algorithm generates offspring by adding random perturbations
    to parent solutions, then selects the best individuals for the next
    generation.

    Two main selection schemes exist: (mu, lambda) where only offspring compete
    for selection (replace_parents=True), and (mu + lambda) where parents and
    offspring compete together (replace_parents=False). The comma strategy
    provides stronger selection pressure and better escapes from local optima,
    while the plus strategy preserves good solutions.

    The algorithm is well-suited for:

    - Continuous optimization problems
    - Real-valued parameter tuning
    - Problems where fine-grained mutation control is beneficial
    - Situations requiring self-adaptive step sizes

    The `mutation_rate` controls the probability of perturbing each parameter,
    while `crossover_rate` determines how often recombination is applied.

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
        Number of parent individuals (mu). Default is 10.
    offspring : int
        Number of offspring to generate each generation (lambda). Should be
        larger than population for effective selection. Default is 20.
    replace_parents : bool
        Selection scheme. If True, uses (mu, lambda) strategy where only
        offspring can become parents. If False, uses (mu + lambda) where
        parents compete with offspring. Default is False.
    mutation_rate : float
        Probability of mutating each parameter in an offspring. Higher values
        increase exploration. Default is 0.7.
    crossover_rate : float
        Probability of recombining parent solutions. ES traditionally
        emphasizes mutation, so this is often lower. Default is 0.3.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import EvolutionStrategyOptimizer

    >>> def sphere(para):
    ...     return -(para["x"] ** 2 + para["y"] ** 2 + para["z"] ** 2)

    >>> search_space = {
    ...     "x": np.linspace(-5, 5, 100),
    ...     "y": np.linspace(-5, 5, 100),
    ...     "z": np.linspace(-5, 5, 100),
    ... }

    >>> opt = EvolutionStrategyOptimizer(
    ...     search_space, population=15, offspring=30, replace_parents=True
    ... )
    >>> opt.search(sphere, n_iter=500)
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
