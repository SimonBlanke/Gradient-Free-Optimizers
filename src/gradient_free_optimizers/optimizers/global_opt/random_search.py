# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..base_optimizer import BaseOptimizer


class RandomSearchOptimizer(BaseOptimizer):
    """Random search optimizer that samples positions uniformly at random.

    Serves as a baseline optimizer and is useful when the objective function
    has no exploitable structure. Each iteration samples a completely random
    position from the search space.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to arrays of possible values.
    initialize : dict, default={"grid": 4, "random": 2, "vertices": 4}
        Strategy for generating initial positions.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random restart (always 1.0 effective for this optimizer).
    nth_process : int, optional
        Process index for parallel optimization.
    """

    name = "Random Search"
    _name_ = "random_search"
    __name__ = "RandomSearchOptimizer"

    optimizer_type = "global"
    computationally_expensive = False

    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )

    @BaseOptimizer.track_new_pos
    def iterate(self):
        """Generate a random position in the search space."""
        return self.move_random()

    @BaseOptimizer.track_new_score
    def evaluate(self, score_new):
        return super().evaluate(score_new)
