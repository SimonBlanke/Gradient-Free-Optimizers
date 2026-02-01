# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Base class for population-based optimizers.

Population-based optimizers maintain multiple candidate solutions
and may have different iteration patterns than single-solution optimizers.
"""

import math

import numpy as np

from ..core_optimizer import CoreOptimizer


def split(positions_l, population):
    """Distribute initial positions across population members.

    Interleaves positions to give each member a diverse starting set.
    For example, with 10 positions and 5 members:
    - Member 0: positions [0, 5]
    - Member 1: positions [1, 6]
    - etc.

    Args:
        positions_l: List of initial positions
        population: Number of population members

    Returns
    -------
        List of position lists, one per population member
    """
    div_int = math.ceil(len(positions_l) / population)
    dist_init_positions = []

    for nth_indiv in range(population):
        indiv_pos = []
        for nth_indiv_pos in range(div_int):
            idx = nth_indiv + nth_indiv_pos * population
            if idx < len(positions_l):
                indiv_pos.append(positions_l[idx])

        dist_init_positions.append(indiv_pos)

    return dist_init_positions


class BasePopulationOptimizer(CoreOptimizer):
    """Base class for population-based optimization algorithms.

    Manages a population of individual optimizers that explore the search
    space in parallel. Provides common functionality for creating populations,
    distributing initial positions, and tracking the best individual.

    Population-based optimizers maintain a population of individuals
    and typically have specialized iteration logic that operates on
    the entire population or pairs of individuals.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to arrays of possible values.
    initialize : dict or None, default=None
        Strategy for generating initial positions distributed across population.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random restart.
    nth_process : int, optional
        Process index for parallel optimization.
    population : int, default=10
        Number of individuals in the population.

    Attributes
    ----------
    optimizers : list
        List of individual optimizer instances in the population.
    systems : list
        Alias for optimizers (used by some subclasses).
    p_current : object
        The currently active optimizer in the population.
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
        self.systems = None  # List of sub-optimizers
        self.optimizers = None  # Alias for systems
        self.p_current = None  # Currently active optimizer

        self.eval_times = []
        self.iter_times = []

        self.init_done = False

    def _create_population(self, Optimizer):
        """Create population of individual optimizers.

        Distributes initial positions across population members and creates
        one optimizer instance per member using warm_start initialization.

        Args:
            Optimizer: The optimizer class to instantiate for each member

        Returns
        -------
            List of optimizer instances
        """
        if isinstance(self.population, int):
            pop_size = self.population
        else:
            pop_size = len(self.population)

        # Ensure we have enough initial positions for the population
        diff_init = pop_size - self.init.n_inits
        if diff_init > 0:
            self.init.add_n_random_init_pos(diff_init)

        if isinstance(self.population, int):
            # Distribute positions across population
            distributed_init_positions = split(
                self.init.init_positions_l, self.population
            )

            population = []
            for init_positions in distributed_init_positions:
                # Convert positions to parameter dicts for warm_start
                init_values = self.conv.positions2values(init_positions)
                init_paras = self.conv.values2paras(init_values)

                population.append(
                    Optimizer(
                        self.conv.search_space,
                        rand_rest_p=self.rand_rest_p,
                        initialize={"warm_start": init_paras},
                        constraints=self.constraints,
                    )
                )
        else:
            population = self.population

        return population

    def _iterations(self, positioners):
        """Count total iterations across all optimizers."""
        nth_iter = 0
        for p in positioners:
            nth_iter = nth_iter + len(p.pos_new_list)
        return nth_iter

    def sort_pop_best_score(self):
        """Sort population by current score (best first).

        Handles None scores by treating them as -infinity (worst).
        """
        scores_list = []
        for _p_ in self.optimizers:
            scores_list.append(_p_.score_current)

        # Sort indices by score descending (pure Python)
        # Handle None scores by treating them as -infinity
        indexed = list(enumerate(scores_list))
        indexed.sort(
            key=lambda x: x[1] if x[1] is not None else float("-inf"), reverse=True
        )
        idx_sorted_ind = [i for i, _ in indexed]

        self.pop_sorted = [self.optimizers[i] for i in idx_sorted_ind]

    def iterate(self):
        """Generate new positions for the population.

        Population-based optimizers should override this to implement
        their specific iteration logic (e.g., temperature swapping in PT).

        The default implementation raises NotImplementedError.
        """
        raise NotImplementedError("iterate() not yet implemented")

    def _evaluate(self, score_new):
        """Evaluate the current individual.

        Population-based optimizers typically delegate to the current
        individual optimizer's evaluate method.

        Args:
            score_new: Score of the most recently evaluated position
        """
        raise NotImplementedError("_evaluate() not yet implemented")

    def _iterate_continuous_batch(self) -> "np.ndarray":
        """Not used - population optimizers have specialized iterate()."""
        raise NotImplementedError("Population optimizers use specialized iterate()")

    def _iterate_categorical_batch(self) -> "np.ndarray":
        """Not used - population optimizers have specialized iterate()."""
        raise NotImplementedError("Population optimizers use specialized iterate()")

    def _iterate_discrete_batch(self) -> "np.ndarray":
        """Not used - population optimizers have specialized iterate()."""
        raise NotImplementedError("Population optimizers use specialized iterate()")

    def finish_initialization(self):
        """Transition from initialization to iteration phase."""
        self.search_state = "iter"
