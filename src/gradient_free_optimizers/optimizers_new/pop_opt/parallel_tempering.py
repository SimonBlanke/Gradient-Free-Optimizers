# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Parallel Tempering (Replica Exchange) Optimizer.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
(via SimulatedAnnealingOptimizer sub-instances)

Template Method Pattern Compliance:
    - Does NOT override iterate() - uses CoreOptimizer's orchestration
    - Implements _iterate_*_batch() for dimension-type-aware position generation
    - Overrides init_pos()/evaluate_init() for population management (acceptable)
"""

import copy
import math
import random

import numpy as np

from ..local_opt import SimulatedAnnealingOptimizer
from .base_population_optimizer import BasePopulationOptimizer

# Temperature initialization parameters for parallel tempering
# temp = TEMP_BASE ** uniform(0, TEMP_MAX_EXPONENT) gives range [1.0, ~9.3]
# This creates a geometric distribution of temperatures across replicas
TEMP_BASE = 1.1
TEMP_MAX_EXPONENT = 25


class ParallelTemperingOptimizer(BasePopulationOptimizer):
    """Parallel Tempering (Replica Exchange) optimizer.

    Runs multiple simulated annealing instances at different temperatures.
    Periodically swaps temperatures between systems to enable exploration
    at high temperatures and exploitation at low temperatures.

    Dimension Support:
        - Continuous: YES (via SimulatedAnnealingOptimizer)
        - Categorical: YES (via SimulatedAnnealingOptimizer)
        - Discrete: YES (via SimulatedAnnealingOptimizer)

    The key insight of parallel tempering is that high-temperature chains
    can escape local optima easily, while low-temperature chains can
    fine-tune solutions. By exchanging temperatures, good configurations
    discovered at high temperatures can be refined at low temperatures.

    The swap acceptance probability uses the Metropolis criterion:
        p = exp((score1 - score2) * (1/temp1 - 1/temp2))

    Template Method Pattern:
        This optimizer follows the Template Method Pattern by implementing
        _iterate_*_batch() methods instead of overriding iterate().
        The sub-optimizer's iterate() is called once per iteration, then
        portions are extracted for each dimension type.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to arrays of possible values.
    initialize : dict, default=None
        Strategy for generating initial positions.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random restart.
    nth_process : int, optional
        Process index for parallel optimization.
    population : int, default=5
        Number of parallel tempering systems (replicas).
    n_iter_swap : int, default=5
        Iterations between temperature swap attempts.
    """

    name = "Parallel Tempering"
    _name_ = "parallel_tempering"
    __name__ = "ParallelTemperingOptimizer"

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
        population=5,
        n_iter_swap=5,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            population=population,
        )

        self.n_iter_swap = n_iter_swap

        # Create population of SimulatedAnnealing optimizers
        self.systems = self._create_population(SimulatedAnnealingOptimizer)

        # Initialize each system with a different random temperature
        # Using geometric distribution for better temperature ladder coverage
        for system in self.systems:
            system.temp = TEMP_BASE ** random.uniform(0, TEMP_MAX_EXPONENT)

        # Required by Search for tracking all optimizers
        self.optimizers = self.systems

        # Iteration state for template method coordination
        self._iteration_setup_done = False
        self._current_new_pos = None

    def _swap_pos(self):
        """Attempt temperature swaps between all pairs using Metropolis criterion.

        For each system, randomly select another system and attempt to swap
        their temperatures. The swap is accepted with probability based on
        the energy difference and temperature difference (Metropolis criterion).

        This allows configurations to travel between temperature levels,
        enabling high-temperature exploration to benefit low-temperature refinement.
        """
        for _p1_ in self.systems:
            # Skip if this system hasn't been evaluated yet
            if _p1_.score_current is None:
                continue

            # Select a random partner (excluding self)
            _systems_temp = copy.copy(self.systems)
            if len(_systems_temp) > 1:
                _systems_temp.remove(_p1_)

            # Filter to systems that have been evaluated
            _systems_evaluated = [
                s for s in _systems_temp if s.score_current is not None
            ]
            if not _systems_evaluated:
                continue

            rand = random.uniform(0, 1) * 100
            _p2_ = random.choice(_systems_evaluated)

            p_accept = self._accept_swap(_p1_, _p2_)
            if p_accept > rand:
                # Swap temperatures between the two systems
                _p1_.temp, _p2_.temp = (_p2_.temp, _p1_.temp)

    def _accept_swap(self, _p1_, _p2_):
        """Calculate the acceptance probability for a temperature swap.

        Uses the Metropolis criterion for replica exchange:
            p = exp((score1 - score2) * (1/temp1 - 1/temp2))

        Normalized by the sum of scores to handle different score magnitudes.

        Args:
            _p1_: First system
            _p2_: Second system

        Returns
        -------
            Acceptance probability scaled to 0-100 range
        """
        denom = _p1_.score_current + _p2_.score_current

        if denom == 0:
            return 100
        elif np.isinf(abs(denom)):
            return 0
        else:
            score_diff_norm = (_p1_.score_current - _p2_.score_current) / denom

            temp = (1 / _p1_.temp) - (1 / _p2_.temp)
            exponent = score_diff_norm * temp
            try:
                return math.exp(exponent) * 100
            except OverflowError:
                return math.inf

    def init_pos(self):
        """Get next initialization position from round-robin system.

        Round-robins through the population, letting each system provide
        its next initialization position in turn.

        Note: This override is acceptable because population-based optimizers
        need to manage initialization across multiple sub-optimizers.

        Returns
        -------
        np.ndarray
            The next initialization position.
        """
        # Select current system using round-robin
        nth_pop = self.nth_trial % len(self.systems)
        self.p_current = self.systems[nth_pop]

        # Get init position from current system
        pos = self.p_current.init_pos()

        # Track in parent optimizer (property setter auto-appends)
        self.pos_new = pos

        return pos

    def evaluate_init(self, score_new):
        """Handle initialization phase evaluation.

        Delegates to the current system's evaluate_init and tracks
        best position across all systems.

        Note: This override is acceptable because population-based optimizers
        need to track scores across multiple sub-optimizers.

        Args:
            score_new: Score of the most recently evaluated init position
        """
        # Track the score on main optimizer (this increments nth_trial)
        self._track_score(score_new)

        # Delegate to current system
        self.p_current.evaluate_init(score_new)

        # Update parent's best from all systems
        for system in self.systems:
            if system.score_best is not None and system.pos_best is not None:
                self._update_best(system.pos_best, system.score_best)

        # Initialize current if first evaluation
        if self.pos_current is None and self.p_current.pos_current is not None:
            self.pos_current = self.p_current.pos_current.copy()
            self.score_current = self.p_current.score_current

    # =========================================================================
    # Template Method Implementation - NO iterate() override!
    # =========================================================================

    def _setup_iteration(self):
        """Set up current iteration by selecting system and getting its position.

        Called lazily by the first _iterate_*_batch() method.
        Delegates to the current sub-optimizer's iterate() and stores
        the result for extraction by the batch methods.
        """
        if self._iteration_setup_done:
            return

        # Select current system using round-robin
        self.p_current = self.systems[self.nth_trial % len(self.systems)]

        # Delegate iteration to the sub-optimizer
        # The sub-optimizer (SimulatedAnnealing) implements the template methods
        self._current_new_pos = self.p_current.iterate()

        self._iteration_setup_done = True

    def _iterate_continuous_batch(self) -> np.ndarray:
        """Generate continuous values by delegating to current sub-optimizer.

        Returns the continuous portion of the sub-optimizer's position.

        Returns
        -------
        np.ndarray
            New continuous values from the current sub-optimizer.
        """
        self._setup_iteration()
        return self._current_new_pos[self._continuous_mask]

    def _iterate_categorical_batch(self) -> np.ndarray:
        """Generate categorical indices by delegating to current sub-optimizer.

        Returns the categorical portion of the sub-optimizer's position.

        Returns
        -------
        np.ndarray
            New category indices from the current sub-optimizer.
        """
        self._setup_iteration()
        return self._current_new_pos[self._categorical_mask]

    def _iterate_discrete_batch(self) -> np.ndarray:
        """Generate discrete indices by delegating to current sub-optimizer.

        Returns the discrete portion of the sub-optimizer's position.

        Returns
        -------
        np.ndarray
            New discrete indices from the current sub-optimizer.
        """
        self._setup_iteration()
        return self._current_new_pos[self._discrete_mask]

    def _evaluate(self, score_new):
        """Evaluate with temperature swapping and delegation to sub-optimizer.

        Periodically performs temperature swaps between systems, then
        delegates evaluation to the current sub-optimizer.

        Also resets iteration state for the next iteration.

        Args:
            score_new: Score of the most recently evaluated position
        """
        # Periodically attempt temperature swaps
        notZero = self.n_iter_swap != 0
        modZero = self.nth_trial % self.n_iter_swap == 0

        if notZero and modZero:
            self._swap_pos()

        # Note: We do NOT set p_current.pos_new here because the sub-optimizer
        # already tracked the position when we called p_current.iterate() in
        # _setup_iteration(). Setting it again would double-count positions.

        # Delegate to current system's evaluate
        self.p_current.evaluate(score_new)

        # Update parent's best tracking from all systems
        for system in self.systems:
            if system.score_best is not None and system.pos_best is not None:
                self._update_best(system.pos_best, system.score_best)

        # Update current tracking from the active sub-optimizer
        if self.p_current.pos_current is not None:
            self._update_current(
                self.p_current.pos_current, self.p_current.score_current
            )

        # Reset iteration setup for next iteration
        self._iteration_setup_done = False
        self._current_new_pos = None
