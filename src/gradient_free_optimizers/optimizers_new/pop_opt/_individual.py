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
            self.pos_new = pos_new
            self.pos_new_list.append(pos_new)
            return pos_new

        # Guard against None positions during early iterations
        if self.pos_current is None:
            pos_new = self.init.move_random_typed()
            self.pos_new = pos_new
            self.pos_new_list.append(pos_new)
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

        # Track position
        self.pos_new = pos_new
        self.pos_new_list.append(pos_new)

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

        This is copied from CoreOptimizer.iterate() but takes a position argument.
        """
        n_dims = len(pos_current)
        new_pos = np.empty(n_dims, dtype=object)

        # Handle continuous dimensions
        if self.conv.continuous_mask is not None and len(self.conv.continuous_mask) > 0:
            cont_mask = self.conv.continuous_mask
            cont_bounds = self.conv.continuous_bounds
            cont_current = pos_current[cont_mask].astype(float)
            cont_new = self._iterate_continuous_batch(cont_current, cont_bounds)
            # Clip to bounds
            cont_new = np.clip(cont_new, cont_bounds[:, 0], cont_bounds[:, 1])
            new_pos[cont_mask] = cont_new

        # Handle categorical dimensions
        if self.conv.categorical_mask is not None and len(self.conv.categorical_mask) > 0:
            cat_mask = self.conv.categorical_mask
            n_cats = self.conv.n_categories
            cat_current = pos_current[cat_mask].astype(int)
            cat_new = self._iterate_categorical_batch(cat_current, n_cats)
            # Clip to valid range
            cat_new = np.clip(cat_new, 0, n_cats - 1).astype(int)
            new_pos[cat_mask] = cat_new

        # Handle discrete dimensions
        if self.conv.discrete_mask is not None and len(self.conv.discrete_mask) > 0:
            disc_mask = self.conv.discrete_mask
            disc_bounds = self.conv.discrete_bounds
            disc_current = pos_current[disc_mask].astype(float)
            disc_new = self._iterate_discrete_batch(disc_current, disc_bounds)
            # Round and clip to bounds
            disc_new = np.round(disc_new)
            disc_new = np.clip(disc_new, disc_bounds[:, 0], disc_bounds[:, 1]).astype(int)
            new_pos[disc_mask] = disc_new

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
