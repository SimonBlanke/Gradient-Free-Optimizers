# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Individual class for Evolutionary Algorithms.

An Individual extends HillClimbingOptimizer to add self-adaptive mutation
step sizes (sigma). Used by Genetic Algorithm, Evolution Strategy, and
Differential Evolution.
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

    def iterate(self):
        """Generate new position via self-adaptive mutation.

        1. Mutate sigma using log-normal distribution
        2. Generate new position using hill climbing with new sigma
        3. Track the new position

        Returns
        -------
        np.ndarray
            New position after mutation.
        """
        # Random restart check
        if random.random() < self.rand_rest_p:
            pos_new = self.init.move_random_typed()
            self.pos_new = pos_new  # Property setter auto-appends
            return pos_new

        # Guard against None positions during early iterations
        if self.pos_current is None:
            pos_new = self.init.move_random_typed()
            self.pos_new = pos_new  # Property setter auto-appends
            return pos_new

        # Mutate sigma (log-normal distribution)
        self.sigma_new = self.sigma * math.exp(self.tau * self._rng.normal())
        self.sigma_new = max(self.sigma_min, min(self.sigma_new, self.sigma_max))

        # Temporarily set epsilon to new sigma for move_climb
        original_epsilon = self.epsilon
        self.epsilon = self.sigma_new

        # Use inherited iterate() which calls _iterate_*_batch() methods
        # This handles dimension types correctly
        pos_new = self._move_climb_typed(self.pos_current)

        # Restore original epsilon
        self.epsilon = original_epsilon

        # Track position (property setter auto-appends)
        self.pos_new = pos_new

        return pos_new

    def _move_climb_typed(self, pos):
        """Move using dimension-type-aware hill climbing.

        This method performs the actual mutation by calling the
        inherited _iterate_*_batch() methods from HillClimbingOptimizer.

        Parameters
        ----------
        pos : np.ndarray
            Current position.

        Returns
        -------
        np.ndarray
            New position after mutation.
        """
        # Use the CoreOptimizer's _iterate_typed() method which calls
        # _iterate_continuous_batch, _iterate_categorical_batch, _iterate_discrete_batch
        # These are inherited from HillClimbingOptimizer
        return self._iterate_typed(pos)

    def _iterate_typed(self, pos_current):
        """Perform typed iteration using dimension masks.

        Temporarily sets self._pos_current so the parameterless template methods
        can access the current position via state.
        Uses the dimension masks set up by _setup_dimension_masks() in CoreOptimizer.
        """
        n_dims = len(pos_current)
        new_pos = np.empty(n_dims, dtype=object)

        # Temporarily set _pos_current so template methods can access it
        old_pos_current = getattr(self, "_pos_current", None)
        self._pos_current = pos_current

        try:
            # Handle continuous dimensions
            if self._continuous_bounds is not None:
                cont_mask = self._continuous_mask
                cont_bounds = self._continuous_bounds
                # Call parameterless method - it accesses self._pos_current
                cont_new = self._iterate_continuous_batch()
                # Clip to bounds
                cont_new = np.clip(cont_new, cont_bounds[:, 0], cont_bounds[:, 1])
                new_pos[cont_mask] = cont_new

            # Handle categorical dimensions
            if self._categorical_sizes is not None:
                cat_mask = self._categorical_mask
                n_cats = self._categorical_sizes
                # Call parameterless method
                cat_new = self._iterate_categorical_batch()
                # Clip to valid range
                cat_new = np.clip(cat_new, 0, n_cats - 1).astype(int)
                new_pos[cat_mask] = cat_new

            # Handle discrete dimensions
            if self._discrete_bounds is not None:
                disc_mask = self._discrete_mask
                disc_bounds = self._discrete_bounds
                # Call parameterless method
                disc_new = self._iterate_discrete_batch()
                # Round and clip to bounds
                disc_new = np.round(disc_new)
                disc_new = np.clip(
                    disc_new, disc_bounds[:, 0], disc_bounds[:, 1]
                ).astype(int)
                new_pos[disc_mask] = disc_new
        finally:
            # Restore original _pos_current
            self._pos_current = old_pos_current

        return new_pos.astype(float)

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

    def _evaluate(self, score_new):
        """Evaluate and update sigma based on success.

        If the mutation improved the score, adopt the new sigma.

        Parameters
        ----------
        score_new : float
            Score of the most recently evaluated position.
        """
        # Adopt new sigma only if mutation was successful
        if self.score_current is not None and score_new > self.score_current:
            self.sigma = self.sigma_new

        # Update current position and score
        self._update_current(self.pos_new, score_new)

        # Update best if improved
        self._update_best(self.pos_new, score_new)
