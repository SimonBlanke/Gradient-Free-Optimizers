# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import List, Dict, Literal, Union

from ..search import Search
from ..optimizers import GeneticAlgorithmOptimizer as _GeneticAlgorithmOptimizer


class GeneticAlgorithmOptimizer(_GeneticAlgorithmOptimizer, Search):
    """
    Evolutionary optimizer inspired by natural selection and genetics.

    Genetic Algorithm (GA) is a population-based metaheuristic that mimics
    the process of natural evolution. The algorithm maintains a population of
    candidate solutions (individuals) that evolve over generations through
    selection, crossover (recombination), and mutation operators. Better
    solutions have higher probability of surviving and reproducing, gradually
    improving the population's fitness.

    Each generation follows these steps: (1) Selection - choosing parents based
    on fitness, (2) Crossover - combining parent genes to create offspring,
    (3) Mutation - introducing random changes for diversity, and (4) Replacement -
    forming the next generation from parents and offspring.

    The algorithm is well-suited for:

    - Combinatorial and discrete optimization problems
    - Multimodal optimization landscapes
    - Problems where the solution can be encoded as a chromosome-like structure
    - Situations requiring robust global search

    The balance between `crossover_rate` and `mutation_rate` controls
    exploration vs exploitation. Higher crossover promotes combining good
    solutions, while higher mutation maintains population diversity.

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
    population : int
        The number of individuals in the population. Larger populations
        provide more genetic diversity. Default is 10.
    offspring : int
        Number of offspring to generate each generation. Typically equal to
        or larger than population size. Default is 10.
    crossover : str
        The crossover operator for combining parent genes. Options include
        "discrete-recombination". Default is "discrete-recombination".
    n_parents : int
        Number of parents selected for each crossover operation.
        Default is 2.
    mutation_rate : float
        Probability of mutating each gene in an offspring. Higher values
        increase diversity but may disrupt good solutions. Default is 0.5.
    crossover_rate : float
        Probability of applying crossover to create offspring vs. cloning
        parents. Default is 0.5.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import GeneticAlgorithmOptimizer

    >>> def knapsack_like(para):
    ...     return para["item1"] * 10 + para["item2"] * 20 + para["item3"] * 15

    >>> search_space = {
    ...     "item1": np.array([0, 1]),
    ...     "item2": np.array([0, 1]),
    ...     "item3": np.array([0, 1]),
    ... }

    >>> opt = GeneticAlgorithmOptimizer(
    ...     search_space, population=20, mutation_rate=0.3, crossover_rate=0.7
    ... )
    >>> opt.search(knapsack_like, n_iter=100)
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
        offspring=10,
        crossover="discrete-recombination",
        n_parents=2,
        mutation_rate=0.5,
        crossover_rate=0.5,
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
            crossover=crossover,
            n_parents=n_parents,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
        )
