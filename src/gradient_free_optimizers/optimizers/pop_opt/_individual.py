# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Individual class for Evolutionary Algorithms.

An Individual extends HillClimbingOptimizer to add self-adaptive mutation
step sizes (sigma). Used by Genetic Algorithm, Evolution Strategy, and
Differential Evolution.

Template Method Pattern Compliance:
    - Does NOT override iterate() - uses CoreOptimizer's orchestration
    - Overrides _iterate_*_batch() for self-adaptive sigma mutation
    - Uses lazy setup pattern to mutate sigma once per iteration
"""

import math
import random

import numpy as np

from ..local_opt import HillClimbingOptimizer


class Individual(HillClimbingOptimizer):
    """Individual in an evolutionary population.

    Each individual maintains:
    - Position (inherited from HillClimbingOptimizer)
    - Self-adaptive mutation step size (sigma)
    - Personal best position and score

    The sigma self-adaptation follows Schwefel (1981):
        sigma_new = sigma * exp(tau * N(0, 1))

    Where tau = 1 / sqrt(n_dimensions) is the learning rate.

    Template Method Pattern:
        This class follows the Template Method Pattern by implementing
        _iterate_*_batch() methods instead of overriding iterate().
        Uses lazy setup to mutate sigma exactly once per iteration.

    Parameters
    ----------
    search_space : dict
        Search space definition.
    rand_rest_p : float, default=0.03
        Probability of random restart.
    """

    def __init__(self, *args, rand_rest_p=0.03, **kwargs):
        super().__init__(*args, **kwargs)
        self.rand_rest_p = rand_rest_p

        # Initialize sigma self-adaptation (Schwefel, 1981)
        self.sigma = self.epsilon
        self.sigma_new = self.sigma

        # Learning rate: tau = 1/sqrt(n) where n = number of dimensions
        n_dimensions = len(self.conv.search_space)
        self.tau = 1.0 / math.sqrt(n_dimensions) if n_dimensions > 0 else 1.0

        # Bounds to prevent sigma collapse or divergence
        self.sigma_min = 0.001
        self.sigma_max = 0.5

        # Lazy setup state for template method pattern
        self._iteration_setup_done = False
        self._use_random_restart = False
        self._original_epsilon = None

    # =========================================================================
    # Template Method Implementation - NO iterate() override!
    # =========================================================================

    def _setup_iteration(self):
        """Set up current iteration by mutating sigma.

        Called lazily by the first _iterate_*_batch() method.
        Mutates sigma using log-normal self-adaptation and temporarily
        sets epsilon to the new sigma for this iteration.
        """
        if self._iteration_setup_done:
            return

        # Random restart check
        if random.random() < self.rand_rest_p:
            self._use_random_restart = True
            self._iteration_setup_done = True
            return

        # Guard against None positions during early iterations
        if self._pos_current is None:
            self._use_random_restart = True
            self._iteration_setup_done = True
            return

        self._use_random_restart = False

        # Mutate sigma (log-normal distribution) - Schwefel (1981)
        self.sigma_new = self.sigma * math.exp(self.tau * self._rng.normal())
        self.sigma_new = max(self.sigma_min, min(self.sigma_new, self.sigma_max))

        # Temporarily set epsilon to new sigma for this iteration
        self._original_epsilon = self.epsilon
        self.epsilon = self.sigma_new

        self._iteration_setup_done = True

    def _iterate_continuous_batch(self) -> np.ndarray:
        """Generate continuous values with self-adaptive sigma.

        Calls lazy setup first, then either returns random values
        (for random restart) or uses parent's implementation with
        the mutated sigma (via self.epsilon).

        Returns
        -------
        np.ndarray
            New continuous values.
        """
        self._setup_iteration()

        if self._use_random_restart:
            # Generate random continuous values
            bounds = self._continuous_bounds
            return self._rng.uniform(bounds[:, 0], bounds[:, 1])

        # Use parent's implementation with mutated epsilon (sigma_new)
        return super()._iterate_continuous_batch()

    def _iterate_categorical_batch(self) -> np.ndarray:
        """Generate categorical indices with self-adaptive mutation rate.

        Calls lazy setup first, then either returns random values
        (for random restart) or uses parent's implementation.

        Returns
        -------
        np.ndarray
            New category indices.
        """
        self._setup_iteration()

        if self._use_random_restart:
            # Generate random categorical indices
            n_categories = self._categorical_sizes
            return np.floor(self._rng.random(len(n_categories)) * n_categories).astype(
                np.int64
            )

        # Use parent's implementation
        return super()._iterate_categorical_batch()

    def _iterate_discrete_batch(self) -> np.ndarray:
        """Generate discrete indices with self-adaptive sigma.

        Calls lazy setup first, then either returns random values
        (for random restart) or uses parent's implementation with
        the mutated sigma (via self.epsilon).

        Returns
        -------
        np.ndarray
            New discrete indices.
        """
        self._setup_iteration()

        if self._use_random_restart:
            # Generate random discrete indices
            bounds = self._discrete_bounds
            return self._rng.integers(
                bounds[:, 0], bounds[:, 1] + 1, size=len(bounds)
            ).astype(float)

        # Use parent's implementation with mutated epsilon (sigma_new)
        return super()._iterate_discrete_batch()

    def _on_evaluate(self, score_new):
        """Evaluate and update sigma based on success.

        If the mutation improved the score, adopt the new sigma.
        Also restores original epsilon and resets iteration state.

        Parameters
        ----------
        score_new : float
            Score of the most recently evaluated position.
        """
        # Adopt new sigma only if mutation was successful
        if self._score_current is not None and score_new > self._score_current:
            self.sigma = self.sigma_new

        # Restore original epsilon if it was modified
        if self._original_epsilon is not None:
            self.epsilon = self._original_epsilon
            self._original_epsilon = None

        # Reset iteration setup for next iteration
        self._iteration_setup_done = False
        self._use_random_restart = False

        # Update current position and score
        self._update_current(self._pos_new, score_new)

        # Update best if improved
        self._update_best(self._pos_new, score_new)

    # =========================================================================
    # Helper methods for external use (e.g., by EvolutionStrategyOptimizer)
    # =========================================================================

    def move_climb_typed(self, pos, epsilon_mod=1.0):
        """Move using hill climbing with optional epsilon modifier.

        Used by GA/ES/DE for constraint handling when initial position
        violates constraints.

        Parameters
        ----------
        pos : np.ndarray
            Current position.
        epsilon_mod : float, default=1.0
            Multiplier for epsilon (larger = bigger steps).

        Returns
        -------
        np.ndarray
            New position after hill climbing.
        """
        original_epsilon = self.epsilon
        self.epsilon = self.epsilon * epsilon_mod

        pos_new = self._iterate_typed(pos)

        self.epsilon = original_epsilon
        return pos_new

    def _iterate_typed(self, pos_current):
        """Perform typed iteration using dimension masks.

        Temporarily sets self._pos_current so the parameterless template methods
        can access the current position via state.
        Uses the dimension masks set up by _setup_dimension_masks() in CoreOptimizer.

        Note: This is a helper method for move_climb_typed(), not for iterate().
        """
        n_dims = len(pos_current)
        new_pos = np.empty(n_dims, dtype=object)

        # Temporarily set _pos_current so template methods can access it
        # Bypass the property setter to avoid appending to _pos_current_list;
        # backing field uses name mangling:
        # __pos_current -> _CoreOptimizer__pos_current
        old_pos_current = self._pos_current
        self.__dict__["_CoreOptimizer__pos_current"] = pos_current

        # Also need to disable the lazy setup since we're calling directly
        old_setup_done = self._iteration_setup_done
        self._iteration_setup_done = True  # Prevent _setup_iteration from running
        old_use_random = self._use_random_restart
        self._use_random_restart = False

        try:
            # Handle continuous dimensions
            if self._continuous_bounds is not None:
                cont_mask = self._continuous_mask
                cont_bounds = self._continuous_bounds
                # Call parent's method directly (bypass our override's setup)
                cont_new = HillClimbingOptimizer._iterate_continuous_batch(self)
                # Clip to bounds
                cont_new = np.clip(cont_new, cont_bounds[:, 0], cont_bounds[:, 1])
                new_pos[cont_mask] = cont_new

            # Handle categorical dimensions
            if self._categorical_sizes is not None:
                cat_mask = self._categorical_mask
                n_cats = self._categorical_sizes
                # Call parent's method directly
                cat_new = HillClimbingOptimizer._iterate_categorical_batch(self)
                # Clip to valid range
                cat_new = np.clip(cat_new, 0, n_cats - 1).astype(int)
                new_pos[cat_mask] = cat_new

            # Handle discrete dimensions
            if self._discrete_bounds is not None:
                disc_mask = self._discrete_mask
                disc_bounds = self._discrete_bounds
                # Call parent's method directly
                disc_new = HillClimbingOptimizer._iterate_discrete_batch(self)
                # Round and clip to bounds
                disc_new = np.round(disc_new)
                disc_new = np.clip(
                    disc_new, disc_bounds[:, 0], disc_bounds[:, 1]
                ).astype(int)
                new_pos[disc_mask] = disc_new
        finally:
            # Restore original state (bypass property setter)
            self.__dict__["_CoreOptimizer__pos_current"] = old_pos_current
            self._iteration_setup_done = old_setup_done
            self._use_random_restart = old_use_random

        return new_pos.astype(float)
