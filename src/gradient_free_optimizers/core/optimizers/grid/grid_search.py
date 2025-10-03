# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from ..base_optimizer import BaseOptimizer
from .diagonal_grid_search import DiagonalGridSearchOptimizer
from .orthogonal_grid_search import OrthogonalGridSearchOptimizer


class GridSearchOptimizer(BaseOptimizer):
    name = "Grid Search"
    _name_ = "grid_search"
    __name__ = "GridSearchOptimizer"

    optimizer_type = "global"
    computationally_expensive = False

    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=[],
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        step_size=1,
        direction="diagonal",
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )

        self.step_size = step_size
        self.direction = direction

        if direction == "orthogonal":
            self.grid_search_opt = OrthogonalGridSearchOptimizer(
                search_space=search_space,
                initialize=initialize,
                constraints=constraints,
                random_state=random_state,
                rand_rest_p=rand_rest_p,
                nth_process=nth_process,
                step_size=step_size,
            )
        elif direction == "diagonal":
            self.grid_search_opt = DiagonalGridSearchOptimizer(
                search_space=search_space,
                initialize=initialize,
                constraints=constraints,
                random_state=random_state,
                rand_rest_p=rand_rest_p,
                nth_process=nth_process,
                step_size=step_size,
            )
        else:
            msg = ""
            raise Exception(msg)

    @BaseOptimizer.track_new_pos
    def iterate(self):
        return self.grid_search_opt.iterate()

    @BaseOptimizer.track_new_score
    def evaluate(self, score_new):
        self.grid_search_opt.evaluate(score_new)
