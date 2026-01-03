# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import List, Dict, Literal, Union

from ..search import Search
from ..optimizers import PowellsMethod as _PowellsMethod


class PowellsMethod(_PowellsMethod, Search):
    """
    A class implementing **Powell's conjugate direction method** for the public API.
    Inheriting from the `Search`-class to get the `search`-method and from
    the `PowellsMethod`-backend to get the underlying algorithm.

    Powell's method performs sequential line searches along a set of directions,
    updating the directions after each complete cycle to form conjugate directions.
    This leads to faster convergence than simple coordinate descent.

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
    epsilon : float
        The step-size for hill climbing line search.
    distribution : str
        The type of distribution to sample from for hill climbing.
    n_neighbours : int
        The number of neighbours to sample and evaluate before moving to the best
        of those neighbours.
    iters_per_direction : int
        Number of evaluations per direction during line search.
    line_search : str
        Line search method: "grid" (default), "golden", or "hill_climb".
    convergence_threshold : float
        Minimum total improvement per cycle to continue. If the sum of
        improvements across all directions falls below this threshold,
        the optimizer switches to random exploration.
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
        distribution: str = "normal",
        n_neighbours: int = 3,
        iters_per_direction: int = 10,
        line_search: Literal["grid", "golden", "hill_climb"] = "grid",
        convergence_threshold: float = 1e-8,
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
            iters_per_direction=iters_per_direction,
            line_search=line_search,
            convergence_threshold=convergence_threshold,
        )
