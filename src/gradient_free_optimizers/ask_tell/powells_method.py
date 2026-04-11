# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Powell's method optimizer with ask/tell interface."""

from typing import Literal

from .._ask_tell_mixin import AskTell
from ..optimizers import PowellsMethod as _PowellsMethod


class PowellsMethod(_PowellsMethod, AskTell):
    """Powell's Method optimizer with ask/tell interface.

    Parameters
    ----------
    search_space : dict[str, list]
        The search space to explore.
    initial_evaluations : list[tuple[dict, float]]
        Previously evaluated parameters and their scores to seed the optimizer.
    constraints : list, optional
        Constraint functions restricting the search space.
    random_state : int or None, default=None
        Seed for reproducibility.
    rand_rest_p : float, default=0
        Probability of random restart.
    epsilon : float, default=0.03
        Step size for hill climbing line search.
    distribution : str, default="normal"
        Distribution for sampling during hill climbing line search.
    n_neighbours : int, default=3
        Kept for backwards compatibility.
    iters_p_dim : int, default=10
        Number of function evaluations per direction during each line search.
    line_search : str, default="grid"
        Method used for one-dimensional optimization along each direction.
    convergence_threshold : float, default=1e-8
        Minimum total improvement per cycle required to continue optimizing.
    """

    def __init__(
        self,
        search_space: dict[str, list],
        initial_evaluations: list[tuple[dict, float]],
        constraints: list[callable] = None,
        conditions: list[callable] = None,
        random_state: int = None,
        rand_rest_p: float = 0,
        epsilon: float = 0.03,
        distribution: str = "normal",
        n_neighbours: int = 3,
        iters_p_dim: int = 10,
        line_search: Literal["grid", "golden", "hill_climb"] = "grid",
        convergence_threshold: float = 1e-8,
    ):
        if constraints is None:
            constraints = []
        if conditions is None:
            conditions = []

        super().__init__(
            search_space=search_space,
            initialize={"random": 0},
            constraints=constraints,
            conditions=conditions,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            epsilon=epsilon,
            distribution=distribution,
            n_neighbours=n_neighbours,
            iters_p_dim=iters_p_dim,
            line_search=line_search,
            convergence_threshold=convergence_threshold,
        )

        self._process_initial_evaluations(initial_evaluations)
