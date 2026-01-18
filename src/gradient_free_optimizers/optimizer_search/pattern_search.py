# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Pattern search using geometric patterns around the current best point."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers_new import PatternSearch as _PatternSearch
from ..search import Search


class PatternSearch(_PatternSearch, Search):
    """
    Direct search optimizer using geometric patterns around the current best point.

    Pattern Search (also known as Coordinate Search or Compass Search) is a
    derivative-free optimization method that explores the search space using
    a fixed geometric pattern of points around the current best solution.
    At each iteration, the algorithm evaluates all points in the pattern and
    moves to the best one found. If no improvement is found, the pattern size
    is reduced, allowing for finer local search.

    The pattern typically consists of points arranged symmetrically around
    the current position, such as along coordinate axes or in a more complex
    geometric arrangement. This systematic exploration ensures the algorithm
    does not miss improvements in any direction.

    The algorithm is well-suited for:

    - Problems with discontinuous or noisy objective functions
    - Situations requiring guaranteed descent (never accepts worse solutions)
    - Low-dimensional optimization problems
    - Problems where simplicity and robustness are valued over speed

    The `pattern_size` parameter controls the initial search radius as a fraction
    of the search space. The `reduction` factor determines how quickly the
    pattern shrinks when stuck, trading off between exploration and convergence.

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
    n_positions : int
        Number of positions in the search pattern. More positions provide
        better coverage but require more function evaluations. Default is 4.
    pattern_size : float
        Initial pattern size as a fraction of each dimension's range.
        Values between 0 and 1, where 0.25 means the pattern spans 25% of
        the search space initially. Default is 0.25.
    reduction : float
        Factor by which to reduce pattern size when no improvement is found.
        Values between 0 and 1, where 0.9 means the pattern shrinks to 90%
        of its current size. Default is 0.9.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import PatternSearch

    >>> def step_function(para):
    ...     x, y = para["x"], para["y"]
    ...     return -(np.floor(x) + np.floor(y))

    >>> search_space = {
    ...     "x": np.linspace(-5, 5, 100),
    ...     "y": np.linspace(-5, 5, 100),
    ... }

    >>> opt = PatternSearch(search_space, n_positions=8, pattern_size=0.3)
    >>> opt.search(step_function, n_iter=200)
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
        n_positions=4,
        pattern_size=0.25,
        reduction=0.9,
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
            n_positions=n_positions,
            pattern_size=pattern_size,
            reduction=reduction,
        )
