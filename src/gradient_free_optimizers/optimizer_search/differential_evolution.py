# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Differential evolution using vector differences for mutation."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers import (
    DifferentialEvolutionOptimizer as _DifferentialEvolutionOptimizer,
)
from ..search import Search


class DifferentialEvolutionOptimizer(_DifferentialEvolutionOptimizer, Search):
    """
    Evolutionary optimizer using vector differences for mutation.

    Differential Evolution (DE) is a powerful population-based optimizer
    particularly effective for continuous optimization problems. The key
    innovation is using weighted differences between population members to
    generate mutations, which automatically adapts the search scale to the
    current population distribution.

    For each individual, DE creates a trial vector by adding a weighted
    difference of two random population members to a third member (mutation),
    then applying crossover with the original individual. If the trial vector
    improves upon the original, it replaces it in the next generation. This
    simple yet effective scheme provides robust global optimization.

    The algorithm is well-suited for:

    - Continuous optimization problems with real-valued parameters
    - Multimodal functions with many local optima
    - Non-separable problems where parameters interact
    - Black-box optimization without gradient information

    DE is known for its simplicity, few control parameters, and robust
    performance across a wide range of problems. The `mutation_rate` (often
    called F) controls the amplification of differential variation.

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
        Number of individuals in the population. Should be at least 4 for
        the differential mutation to work. Default is 10.
    mutation_rate : float
        Scaling factor F for the differential mutation. Controls the
        amplification of the difference vector. Values typically range from
        0.5 to 1.0. Higher values increase exploration. Default is 0.9.
    crossover_rate : float
        Probability CR of inheriting each parameter from the mutant vector
        vs. the original. Higher values increase the influence of mutation.
        Default is 0.9.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import DifferentialEvolutionOptimizer

    >>> def rosenbrock(para):
    ...     x, y = para["x"], para["y"]
    ...     return -((1 - x) ** 2 + 100 * (y - x ** 2) ** 2)

    >>> search_space = {
    ...     "x": np.linspace(-5, 5, 100),
    ...     "y": np.linspace(-5, 5, 100),
    ... }

    >>> opt = DifferentialEvolutionOptimizer(
    ...     search_space, population=20, mutation_rate=0.8, crossover_rate=0.9
    ... )
    >>> opt.search(rosenbrock, n_iter=500)
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
        population=10,
        mutation_rate=0.9,
        crossover_rate=0.9,
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
            population=population,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
        )
