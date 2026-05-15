# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Base class for population-based optimizers.

Population-based optimizers maintain multiple candidate solutions
and may have different iteration patterns than single-solution optimizers.
"""

from __future__ import annotations

import math

from gradient_free_optimizers._array_backend import ndarray

from ..core_optimizer import CoreOptimizer
from ._topology import VALID_TOPOLOGIES, get_neighbors


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
    topology : str, default="star"
        Neighborhood topology controlling information sharing between
        individuals. Currently supported by ParticleSwarmOptimizer and
        SpiralOptimization.

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
        boundary="clip",
        topology="star",
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            boundary=boundary,
        )
        self.population = population

        if topology not in VALID_TOPOLOGIES:
            raise ValueError(
                f"topology must be one of {VALID_TOPOLOGIES}, got {topology!r}"
            )
        self.topology = topology

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
                        boundary=self.boundary,
                    )
                )
        else:
            population = self.population

        return population

    def _iterations(self, positioners):
        """Count total iterations across all optimizers."""
        nth_iter = 0
        for p in positioners:
            nth_iter = nth_iter + len(p._pos_new_list)
        return nth_iter

    def _sort_pop_best_score(self):
        """Sort population by current score (best first).

        Handles None scores by treating them as -infinity (worst).
        """
        scores_list = []
        for _p_ in self.optimizers:
            scores_list.append(_p_._score_current)

        # Sort indices by score descending (pure Python)
        # Handle None scores by treating them as -infinity
        indexed = list(enumerate(scores_list))
        indexed.sort(
            key=lambda x: x[1] if x[1] is not None else float("-inf"), reverse=True
        )
        idx_sorted_ind = [i for i, _ in indexed]

        self.pop_sorted = [self.optimizers[i] for i in idx_sorted_ind]

    def _on_evaluate(self, score_new):
        """Evaluate the current individual.

        Population-based optimizers typically delegate to the current
        individual optimizer's evaluate method.

        Args:
            score_new: Score of the most recently evaluated position
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _on_evaluate()"
        )

    def _iterate_continuous_batch(self) -> ndarray:
        """Generate continuous values for the current iteration.

        Population optimizers must implement this to provide
        algorithm-specific continuous dimension handling.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _iterate_continuous_batch()"
        )

    def _iterate_categorical_batch(self) -> ndarray:
        """Generate categorical indices for the current iteration.

        Population optimizers must implement this to provide
        algorithm-specific categorical dimension handling.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _iterate_categorical_batch()"
        )

    def _iterate_discrete_batch(self) -> ndarray:
        """Generate discrete indices for the current iteration.

        Population optimizers must implement this to provide
        algorithm-specific discrete dimension handling.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _iterate_discrete_batch()"
        )

    def _get_neighbor_indices(self, individual_idx):
        """Return neighbor indices for an individual under the current topology."""
        return get_neighbors(self.topology, individual_idx, len(self.optimizers))

    def _get_best_neighbor(self, individual_idx):
        """Return the best-scoring neighbor by ``_score_current``.

        Returns ``None`` if no neighbor has been scored yet.
        """
        neighbors = self._get_neighbor_indices(individual_idx)
        best = None
        best_score = float("-inf")
        for idx in neighbors:
            p = self.optimizers[idx]
            if p._score_current is not None and p._score_current > best_score:
                best = p
                best_score = p._score_current
        return best

    def _get_best_neighbor_position(self, individual_idx, position_attr):
        """Return a position attribute from the best-scoring neighbor."""
        best_neighbor = self._get_best_neighbor(individual_idx)
        if best_neighbor is None:
            return None
        return getattr(best_neighbor, position_attr)

    def _on_finish_initialization(self):
        """Perform population-specific setup after init phase.

        Override in subclasses to perform algorithm-specific initialization
        after all init positions have been evaluated.

        Note: DO NOT set search_state here - CoreOptimizer.finish_initialization()
        handles that automatically after calling this hook.
        """
        pass
