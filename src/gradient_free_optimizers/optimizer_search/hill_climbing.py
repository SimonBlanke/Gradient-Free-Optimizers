# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Hill climbing that iteratively moves toward better neighboring solutions."""

from typing import Literal

from ..optimizers import HillClimbingOptimizer as _HillClimbingOptimizer
from ..search import Search


class HillClimbingOptimizer(_HillClimbingOptimizer, Search):
    """
    Local search optimizer that iteratively moves towards better neighboring solutions.

    Hill Climbing is a simple yet effective local search algorithm that starts from
    an initial position and iteratively moves to neighboring positions that improve
    the objective function. At each step, the algorithm samples a set of neighboring
    positions and moves to the best one found. This greedy approach makes it fast
    and memory-efficient, but susceptible to getting stuck in local optima.

    The algorithm is well-suited for:

    - Unimodal optimization problems with a single global optimum
    - Fine-tuning solutions found by other optimizers
    - Problems where function evaluations are expensive (due to low overhead)
    - Situations requiring a simple, interpretable optimization process

    The `epsilon` parameter controls the step size: smaller values lead to finer
    local search but slower convergence, while larger values enable broader
    exploration but may overshoot optima. The `n_neighbours` parameter determines
    how many candidate positions are evaluated before making a move.

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
        The step-size for the climbing. Controls how far the optimizer looks
        for neighboring positions. Values typically range from 0.01 to 0.1.
    distribution : str
        The type of distribution to sample neighbors from. Options are
        "normal", "laplace", "gumbel", or "logistic". Different distributions
        affect the shape of the neighborhood sampling.
    n_neighbours : int
        The number of neighbours to sample and evaluate before moving to the best
        of those neighbours. Higher values increase exploration but require more
        function evaluations per iteration.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import HillClimbingOptimizer

    >>> def parabola(para):
    ...     return -(para["x"] ** 2 + para["y"] ** 2)

    >>> search_space = {
    ...     "x": np.linspace(-10, 10, 100),
    ...     "y": np.linspace(-10, 10, 100),
    ... }

    >>> opt = HillClimbingOptimizer(search_space)
    >>> opt.search(parabola, n_iter=100)
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
        )
