# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Downhill simplex (Nelder-Mead) using geometric simplex transformations."""

from typing import Literal

from ..optimizers import DownhillSimplexOptimizer as _DownhillSimplexOptimizer
from ..search import Search


class DownhillSimplexOptimizer(_DownhillSimplexOptimizer, Search):
    """
    Derivative-free optimizer using geometric simplex transformations.

    The Downhill Simplex method (also known as Nelder-Mead) is a classic
    derivative-free optimization algorithm that maintains a simplex of n+1 points
    in n-dimensional space. The algorithm iteratively transforms this simplex
    through reflection, expansion, contraction, and shrinking operations to
    move towards better regions of the search space.

    At each iteration, the worst point of the simplex is identified and replaced
    through one of four operations: reflection (moving away from the worst point),
    expansion (extending further in a promising direction), contraction (pulling
    back toward the centroid), or shrinking (contracting the entire simplex toward
    the best point). This adaptive behavior allows efficient navigation without
    gradient information.

    The algorithm is well-suited for:

    - Low to moderate dimensional problems (typically < 20 dimensions)
    - Smooth objective functions without too many local optima
    - Problems where derivatives are unavailable or expensive to compute
    - Initial exploration before applying gradient-based methods

    The simplex parameters (alpha, gamma, beta, sigma) control the aggressiveness
    of each transformation. The default values are well-established and work
    well for most problems.

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
    alpha : float
        Reflection coefficient. Controls how far the reflected point is placed
        from the centroid, opposite to the worst point. Default is 1.
    gamma : float
        Expansion coefficient. When reflection yields a new best point, expansion
        stretches further in that direction. Default is 2.
    beta : float
        Contraction coefficient. Used when reflection does not improve, pulling
        the worst point toward the centroid. Default is 0.5.
    sigma : float
        Shrinking coefficient. When contraction fails, the entire simplex shrinks
        toward the best point by this factor. Default is 0.5.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import DownhillSimplexOptimizer

    >>> def rosenbrock(para):
    ...     x, y = para["x"], para["y"]
    ...     return -((1 - x) ** 2 + 100 * (y - x ** 2) ** 2)

    >>> search_space = {
    ...     "x": np.linspace(-2, 2, 100),
    ...     "y": np.linspace(-1, 3, 100),
    ... }

    >>> opt = DownhillSimplexOptimizer(search_space)
    >>> opt.search(rosenbrock, n_iter=500)
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
        alpha: float = 1,
        gamma: float = 2,
        beta: float = 0.5,
        sigma: float = 0.5,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            sigma=sigma,
        )
