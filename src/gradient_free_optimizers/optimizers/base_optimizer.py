# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Base class for single-solution optimizers.

Single-solution optimizers maintain one current position and explore
the search space by perturbing it. This distinguishes them from
population-based optimizers that maintain multiple candidates.

See CoreOptimizer for the full architecture description.
"""

from .core_optimizer import CoreOptimizer


class BaseOptimizer(CoreOptimizer):
    """Base class for single-solution optimization algorithms.

    Inherits from CoreOptimizer and adds the single-optimizer convention:
    self.optimizers is always [self] (one candidate solution).

    Population-based optimizers use BasePopulationOptimizer instead,
    which manages multiple candidate solutions.

    Hierarchy:

        CoreOptimizer (ABC)           ← orchestration + state management
            ├── BaseOptimizer          ← this class (single-solution)
            │     ├── HillClimbing
            │     ├── SMBO
            │     └── ...
            └── BasePopulationOptimizer ← population-based
                  ├── DifferentialEvolution
                  └── ...
    """

    name = "Base Optimizer"
    _name_ = "base_optimizer"
    __name__ = "BaseOptimizer"

    optimizer_type = "local"
    computationally_expensive = False

    def __init__(
        self,
        search_space,
        initialize=None,
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

        # Single-solution optimizer: population is just [self]
        self.optimizers = [self]
