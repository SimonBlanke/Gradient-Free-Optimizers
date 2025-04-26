# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import List, Dict, Literal, Union

from ..search import Search
from ..optimizers import LipschitzOptimizer as _LipschitzOptimizer


class LipschitzOptimizer(_LipschitzOptimizer, Search):
    """
    A class implementing the **lipschitz optimizer** for the public API.
    Inheriting from the `Search`-class to get the `search`-method and from
    the `LipschitzOptimizer`-backend to get the underlying algorithm.

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
    warm_start_smbo
        The warm start for SMBO.
    max_sample_size : int
        The maximum number of points to sample.
    sampling : dict
        The sampling method to use.
    replacement : bool
        Whether to sample with replacement.
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
        warm_start_smbo=None,
        max_sample_size: int = 10000000,
        sampling: Dict[Literal["random"], int] = {"random": 1000000},
        replacement: bool = True,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            warm_start_smbo=warm_start_smbo,
            max_sample_size=max_sample_size,
            sampling=sampling,
            replacement=replacement,
        )
