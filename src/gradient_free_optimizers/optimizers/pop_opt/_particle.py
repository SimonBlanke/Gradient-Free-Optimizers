# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np
from scipy.spatial.transform import Rotation as R

from ..local_opt import HillClimbingOptimizer
from ..._search_space.dimensions import CategoricalDimension, FixedDimension


class Particle(HillClimbingOptimizer):
    def __init__(
        self,
        *args,
        inertia=0.5,
        cognitive_weight=0.5,
        social_weight=0.5,
        temp_weight=0.2,
        rand_rest_p=0.03,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.global_pos_best = None

        self.inertia = inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.temp_weight = temp_weight
        self.rand_rest_p = rand_rest_p

    def _move_part(self, pos, velo):
        # v1 (discrete lattice) fallback
        if not getattr(self, "_v2_active", False):
            pos_new = (pos + velo).astype(int)
            n_zeros = [0] * len(self.conv.max_positions)
            return np.clip(pos_new, n_zeros, self.conv.max_positions)

        # v2: arithmetic in z-space with special handling for categorical and fixed
        z_new = np.asarray(pos, dtype=float) + np.asarray(velo, dtype=float)
        z_new = np.clip(z_new, 0.0, 1.0)

        # Snap categorical dims to nearest bucket center; keep fixed dims constant
        try:
            dims = self.conv.dims  # type: ignore[attr-defined]
        except Exception:
            return z_new

        out = z_new.copy()
        eps = np.finfo(float).eps
        for i, dim in enumerate(dims):
            kind = getattr(dim, "kind", "")
            if kind == "categorical":
                n = int(getattr(dim, "size", 1) or 1)
                if n <= 1:
                    out[i] = float(dim.value_to_z(dim.sample()))
                else:
                    zz = min(max(out[i], 0.0), 1.0 - eps)
                    idx = int(np.floor(zz * n))
                    idx = max(0, min(n - 1, idx))
                    out[i] = (idx + 0.5) / n
            elif kind == "fixed":
                # keep exact fixed z
                out[i] = float(dim.value_to_z(dim.z_to_value(out[i])))
            else:
                # real/integer/distribution already clamped in z
                pass
        return out

    @HillClimbingOptimizer.track_new_pos
    @HillClimbingOptimizer.random_iteration
    def move_linear(self):
        r1, r2 = random.random(), random.random()

        A = self.inertia * self.velo
        B = self.cognitive_weight * r1 * np.subtract(self.pos_best, self.pos_current)
        C = (
            self.social_weight
            * r2
            * np.subtract(self.global_pos_best, self.pos_current)
        )

        new_velocity = A + B + C
        return self._move_part(self.pos_current, new_velocity)
