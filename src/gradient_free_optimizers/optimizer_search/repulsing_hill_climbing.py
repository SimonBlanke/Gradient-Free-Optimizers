# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Repulsing hill climbing that increases step size to escape local optima."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers import (
    RepulsingHillClimbingOptimizer as _RepulsingHillClimbingOptimizer,
)
from ..search import Search


class RepulsingHillClimbingOptimizer(_RepulsingHillClimbingOptimizer, Search):
    """
    Hill climbing variant that increases step size when stuck to escape local optima.

    Repulsing Hill Climbing is an adaptive variant of hill climbing that dynamically
    increases the search radius (epsilon) when no improvement is found. This
    "repulsion" mechanism helps the optimizer escape local optima by taking
    progressively larger steps when stuck, effectively being "pushed away" from
    the current region. Once a better solution is found, the step size resets
    to its original value for fine-grained local search.

    The algorithm is well-suited for:

    - Optimization landscapes with isolated local optima
    - Problems where the distance between optima is unknown
    - Scenarios requiring automatic adaptation of search radius
    - Balancing local exploitation and global exploration without manual tuning

    The `repulsion_factor` parameter controls how aggressively the step size
    increases when stuck. A factor of 5 means epsilon is multiplied by 5 each
    time no improvement is found. Higher values lead to faster escape from
    local optima but may overshoot good regions.

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
        The initial step-size for the climbing. This value increases by
        `repulsion_factor` when no improvement is found.
    distribution : str
        The type of distribution to sample neighbors from. Options are
        "normal", "laplace", "gumbel", or "logistic".
    n_neighbours : int
        The number of neighbours to sample and evaluate before moving to the best
        of those neighbours.
    repulsion_factor : float
        Multiplicative factor to increase epsilon when no better position is found.
        Higher values cause more aggressive exploration when stuck. Default is 5.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import RepulsingHillClimbingOptimizer

    >>> def ackley(para):
    ...     x, y = para["x"], para["y"]
    ...     return -(-20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    ...              - np.exp(0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))
    ...              + np.e + 20)

    >>> search_space = {
    ...     "x": np.linspace(-5, 5, 100),
    ...     "y": np.linspace(-5, 5, 100),
    ... }

    >>> opt = RepulsingHillClimbingOptimizer(search_space, repulsion_factor=3)
    >>> opt.search(ackley, n_iter=200)
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
        repulsion_factor: float = 5,
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
            repulsion_factor=repulsion_factor,
        )
