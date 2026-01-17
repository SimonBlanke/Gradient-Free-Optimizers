# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Stochastic hill climbing that probabilistically accepts worse solutions."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers_new import (
    StochasticHillClimbingOptimizer as _StochasticHillClimbingOptimizer,
)
from ..search import Search


class StochasticHillClimbingOptimizer(_StochasticHillClimbingOptimizer, Search):
    """Hill climbing variant that accepts worse solutions to escape local optima.

    Stochastic Hill Climbing extends the basic hill climbing algorithm by introducing
    a probability of accepting solutions that are worse than the current one. This
    stochastic acceptance mechanism helps the optimizer escape local optima and
    explore a broader region of the search space. Unlike standard hill climbing,
    which always moves to better positions, this variant can temporarily accept
    inferior solutions, enabling it to "climb down" from local peaks.

    The algorithm is well-suited for:

    - Multimodal optimization problems with multiple local optima
    - Problems where standard hill climbing gets stuck frequently
    - Situations requiring a balance between local refinement and exploration
    - Optimization landscapes with many plateaus or ridges

    The `p_accept` parameter controls the probability of accepting worse solutions.
    Higher values increase exploration but may slow convergence, while lower values
    make the algorithm behave more like standard hill climbing. A value of 0.0
    reduces this to deterministic hill climbing.

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
        for neighboring positions.
    distribution : str
        The type of distribution to sample neighbors from. Options are
        "normal", "laplace", "gumbel", or "logistic".
    n_neighbours : int
        The number of neighbours to sample and evaluate before moving to the best
        of those neighbours.
    p_accept : float
        Probability of accepting a worse solution. Values range from 0.0 (never
        accept worse) to 1.0 (always accept). Default is 0.5, providing moderate
        exploration capability.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import StochasticHillClimbingOptimizer

    >>> def rastrigin(para):
    ...     x, y = para["x"], para["y"]
    ...     return -(20 + x**2 + y**2 - 10 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))

    >>> search_space = {
    ...     "x": np.linspace(-5.12, 5.12, 100),
    ...     "y": np.linspace(-5.12, 5.12, 100),
    ... }

    >>> opt = StochasticHillClimbingOptimizer(search_space, p_accept=0.3)
    >>> opt.search(rastrigin, n_iter=200)
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
        p_accept: float = 0.5,
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
            p_accept=p_accept,
        )
