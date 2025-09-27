# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import List, Dict, Literal, Union

from ..search import Search
from ..optimizers import COBYLA as _COBYLA


class COBYLA(_COBYLA, Search):
    def __init__(
        self,
        rho_beg,
        rho_end,
        x_0,
        search_space: Dict[str, list],
        initialize: Dict[
            Literal["grid", "vertices", "random", "warm_start"],
            Union[int, list[dict]],
        ] = {"grid": 4, "random": 2, "vertices": 4},
        constraints: List[callable] = [],
        random_state: int = None,
        rand_rest_p: float = 0,
        nth_process: int = None,
    ):
        initialize={"grid": 4, "random": 2, "vertices": 4},
        rand_rest_p=0,
        super().__init__(
            search_space=search_space,
            rho_beg=rho_beg,
            rho_end=rho_end,
            x_0=x_0,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )
