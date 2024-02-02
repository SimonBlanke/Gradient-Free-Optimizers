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

    def __init__(self, *args, step_size=1, direction="diagonal", **kwargs):
        super().__init__(*args, **kwargs)

        self.step_size = step_size
        self.direction = direction

        if direction == "orthogonal":
            self.grid_search_opt = OrthogonalGridSearchOptimizer(
                *args, step_size=step_size, **kwargs
            )
        elif direction == "diagonal":
            self.grid_search_opt = DiagonalGridSearchOptimizer(
                *args, step_size=step_size, **kwargs
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
