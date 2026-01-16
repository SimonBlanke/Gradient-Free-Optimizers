# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random

from gradient_free_optimizers._array_backend import array, clip
from gradient_free_optimizers._dimension_types import DimensionType

from ..local_opt import HillClimbingOptimizer


class Particle(HillClimbingOptimizer):
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

        self.inertia = inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.temp_weight = temp_weight
        self.rand_rest_p = rand_rest_p

    def _move_part(self, pos, velo):
        """Apply velocity to position with type-aware handling.

        For discrete-numerical dimensions: standard velocity addition with rounding.
        For continuous dimensions: velocity addition preserving float precision.
        For categorical dimensions: velocity magnitude determines switch probability.
        """
        # Fast path for legacy mode (all discrete-numerical)
        if self.conv.is_legacy_mode:
            pos_new = (array(pos) + array(velo)).astype(int)
            n_zeros = [0] * len(self.conv.max_positions)
            return clip(pos_new, n_zeros, self.conv.max_positions)

        # Type-aware movement for mixed dimension types
        pos_new = []
        for idx, dim_type in enumerate(self.conv.dim_types):
            if dim_type == DimensionType.CONTINUOUS:
                # Keep as float, will be clipped by conv2pos_typed
                new_val = float(pos[idx]) + float(velo[idx])
                pos_new.append(new_val)
            elif dim_type == DimensionType.CATEGORICAL:
                # Velocity magnitude determines switch probability
                # Higher velocity = higher chance of random category jump
                max_idx = self.conv.dim_infos[idx].bounds[1]
                switch_prob = min(1.0, abs(velo[idx]) / (max_idx + 1))
                if random.random() < switch_prob:
                    pos_new.append(random.randint(0, max_idx))
                else:
                    pos_new.append(int(pos[idx]))
            else:  # DISCRETE_NUMERICAL
                new_val = int(round(pos[idx] + velo[idx]))
                pos_new.append(new_val)

        return self.conv2pos_typed(array(pos_new))

    @HillClimbingOptimizer.track_new_pos
    @HillClimbingOptimizer.random_iteration
    def move_linear(self):
        r1, r2 = random.random(), random.random()

        A = self.inertia * array(self.velo)
        B = (
            self.cognitive_weight
            * r1
            * (array(self.pos_best) - array(self.pos_current))
        )
        C = (
            self.social_weight
            * r2
            * (array(self.global_pos_best) - array(self.pos_current))
        )

        new_velocity = A + B + C
        return self._move_part(self.pos_current, new_velocity)
