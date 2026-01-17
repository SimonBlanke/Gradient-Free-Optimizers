# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Base class for population-based optimizers.

Population-based optimizers maintain multiple candidate solutions
and may have different iteration patterns than single-solution optimizers.
"""

from ..core_optimizer import CoreOptimizer


class BasePopulationOptimizer(CoreOptimizer):
    """Base class for population-based optimization algorithms.

    Population-based optimizers maintain a population of individuals
    and typically have specialized iteration logic that operates on
    the entire population or pairs of individuals.

    Subclasses may override iterate() to implement population-specific
    logic (e.g., velocity updates in PSO, mutation in DE).
    """

    name = "Base Population Optimizer"
    _name_ = "base_population_optimizer"
    __name__ = "BasePopulationOptimizer"

    optimizer_type = "population"
    computationally_expensive = False

    def __init__(
        self,
        search_space,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        population=10,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )
        self.population = population

        # Population state
        self.individuals = None  # List of individual positions
        self.scores = None  # Scores for each individual

    def iterate(self):
        """Generate new positions for the population.

        Population-based optimizers may override this to implement
        specialized iteration logic (e.g., PSO velocity, DE mutation).

        The default implementation calls the batch methods for each
        individual, but subclasses often override for efficiency.
        """
        # TODO: Implement population iteration
        raise NotImplementedError("iterate() not yet implemented")

    def evaluate(self, score_new):
        """Evaluate the current individual and update population state."""
        # TODO: Implement population evaluation
        raise NotImplementedError("evaluate() not yet implemented")
