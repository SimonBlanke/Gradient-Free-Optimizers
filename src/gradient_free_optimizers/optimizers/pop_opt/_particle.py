# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Particle class for Particle Swarm Optimization.

A Particle extends HillClimbingOptimizer to add velocity-based movement
and awareness of global best positions.
"""

import random

import numpy as np

from ..local_opt import HillClimbingOptimizer


class Particle(HillClimbingOptimizer):
    """Individual particle in PSO swarm.

    Each particle maintains:
    - Position (inherited from HillClimbingOptimizer)
    - Velocity (for momentum-based movement)
    - Personal best position and score
    - Reference to global best position

    The velocity update equation is:
        v = inertia * v
          + cognitive * r1 * (personal_best - position)
          + social * r2 * (global_best - position)

    Parameters
    ----------
    search_space : dict
        Search space definition.
    inertia : float, default=0.5
        Weight for maintaining current velocity direction.
    cognitive_weight : float, default=0.5
        Weight for attraction toward personal best.
    social_weight : float, default=0.5
        Weight for attraction toward global best.
    temp_weight : float, default=0.2
        Temperature weight for exploration randomness.
    rand_rest_p : float, default=0.03
        Probability of random restart.
    """

    def __init__(
        self,
        *args,
        inertia=0.5,
        cognitive_weight=0.5,
        social_weight=0.5,
        temp_weight=0.2,
        rand_rest_p=0.03,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.global_pos_best = None
        self.velo = None  # Velocity vector

        self.inertia = inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.temp_weight = temp_weight
        self.rand_rest_p = rand_rest_p

    def _move_part(self, pos, velo):
        """Apply velocity to position with type-aware handling.

        For continuous dimensions: velocity addition preserving float precision.
        For discrete dimensions: velocity addition with rounding.
        For categorical dimensions: velocity magnitude determines switch probability.

        Parameters
        ----------
        pos : np.ndarray
            Current position.
        velo : np.ndarray
            Velocity vector to apply.

        Returns
        -------
        np.ndarray
            New position after applying velocity.
        """
        # Fast path for legacy mode (all discrete-numerical)
        if self.conv.is_legacy_mode:
            pos_new = (np.array(pos) + np.array(velo)).astype(int)
            n_zeros = [0] * len(self.conv.max_positions)
            return np.clip(pos_new, n_zeros, self.conv.max_positions)

        # Type-aware movement for mixed dimension types
        from gradient_free_optimizers._dimension_types import DimensionType

        pos_new = []
        for idx, dim_type in enumerate(self.conv.dim_types):
            if dim_type == DimensionType.CONTINUOUS:
                # Keep as float, will be clipped by _conv2pos_typed
                new_val = float(pos[idx]) + float(velo[idx])
                pos_new.append(new_val)
            elif dim_type == DimensionType.CATEGORICAL:
                # Velocity magnitude determines switch probability
                max_idx = self.conv.dim_infos[idx].bounds[1]
                switch_prob = min(1.0, abs(velo[idx]) / (max_idx + 1))
                if random.random() < switch_prob:
                    pos_new.append(random.randint(0, int(max_idx)))
                else:
                    pos_new.append(int(pos[idx]))
            else:  # DISCRETE_NUMERICAL
                new_val = int(round(pos[idx] + velo[idx]))
                pos_new.append(new_val)

        return self._conv2pos_typed(np.array(pos_new))

    def move_linear(self):
        """Compute velocity update and move particle.

        The velocity update follows the standard PSO equation:
            v = w*v + c1*r1*(pbest - pos) + c2*r2*(gbest - pos)

        Where:
            w = inertia (momentum term)
            c1 = cognitive_weight (attraction to personal best)
            c2 = social_weight (attraction to global best)
            r1, r2 = random values in [0, 1]

        Returns
        -------
        np.ndarray
            New position after velocity update.
        """
        # Random restart check
        if random.random() < self.rand_rest_p:
            pos_new = self.init.move_random_typed()
            self._pos_new = pos_new  # Property setter auto-appends
            return pos_new

        # Guard against None positions during early iterations
        if (
            self._pos_current is None
            or self._pos_best is None
            or self.global_pos_best is None
        ):
            # Fall back to random move during early iterations
            pos_new = self.init.move_random_typed()
            self._pos_new = pos_new  # Property setter auto-appends
            return pos_new

        r1, r2 = random.random(), random.random()

        pos_current = np.array(self._pos_current)
        pos_best = np.array(self._pos_best)
        global_pos_best = np.array(self.global_pos_best)

        # Inertia term: maintain current direction
        A = self.inertia * np.array(self.velo)

        # Cognitive term: attract toward personal best
        B = self.cognitive_weight * r1 * (pos_best - pos_current)

        # Social term: attract toward global best
        C = self.social_weight * r2 * (global_pos_best - pos_current)

        # Update velocity
        self.velo = A + B + C

        # Apply velocity to get new position
        pos_new = self._move_part(pos_current, self.velo)

        # Track new position (property setter auto-appends)
        self._pos_new = pos_new

        return pos_new

    def move_climb_typed(self, pos_new):
        """Fallback movement using hill climbing when constraints violated."""
        return self._iterate()

    def _conv2pos_typed(self, pos):
        """Convert position array to valid position with proper types.

        Clips values to bounds and ensures correct data types for each dimension.
        """
        if self.conv.is_legacy_mode:
            n_zeros = [0] * len(self.conv.max_positions)
            return np.clip(pos, n_zeros, self.conv.max_positions).astype(int)

        from gradient_free_optimizers._dimension_types import DimensionType

        pos_new = []
        for idx, dim_type in enumerate(self.conv.dim_types):
            bounds = self.conv.dim_infos[idx].bounds
            val = pos[idx]

            if dim_type == DimensionType.CONTINUOUS:
                # Clip to bounds, keep as float
                pos_new.append(np.clip(val, bounds[0], bounds[1]))
            else:
                # Discrete or categorical: clip and convert to int
                pos_new.append(int(np.clip(round(val), bounds[0], bounds[1])))

        return np.array(pos_new)
