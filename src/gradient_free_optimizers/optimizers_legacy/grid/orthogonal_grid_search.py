# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from gradient_free_optimizers._array_backend import array
from gradient_free_optimizers._init_utils import get_default_initialize

from ..base_optimizer import BaseOptimizer


class OrthogonalGridSearchOptimizer(BaseOptimizer):
    """Orthogonal grid search traversing dimensions sequentially.

    Traverses the search space in a nested loop fashion, exhausting each
    dimension before moving to the next. This provides systematic coverage
    but may take longer to explore distant regions.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to arrays of possible values.
    initialize : dict, default=None
        Strategy for generating initial positions.
        If None, uses {"grid": 4, "random": 2, "vertices": 4}.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random restart.
    nth_process : int, optional
        Process index for parallel optimization.
    step_size : int, default=1
        Step multiplier for grid traversal.
    """

    def __init__(
        self,
        search_space,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        step_size=1,
    ):
        if initialize is None:
            initialize = get_default_initialize()

        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )

        self.step_size = step_size

    def grid_move(self):
        mod_tmp = self.nth_trial * self.step_size + int(
            self.nth_trial * self.step_size / self.conv.search_space_size
        )
        div_tmp = self.nth_trial * self.step_size + int(
            self.nth_trial * self.step_size / self.conv.search_space_size
        )
        flipped_new_pos = []

        for dim_size in self.conv.dim_sizes:
            mod = mod_tmp % dim_size
            div = int(div_tmp / dim_size)

            flipped_new_pos.append(mod)

            mod_tmp = div
            div_tmp = div

        return array(flipped_new_pos)

    @BaseOptimizer.track_new_pos
    def iterate(self):
        """Generate next orthogonal grid position."""
        pos_new = self.grid_move()
        pos_new = self.conv2pos(pos_new)
        return pos_new

    @BaseOptimizer.track_new_score
    def evaluate(self, score_new):
        BaseOptimizer.evaluate(self, score_new)
