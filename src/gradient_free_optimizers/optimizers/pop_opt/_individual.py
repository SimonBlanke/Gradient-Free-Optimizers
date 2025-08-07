# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from collections import deque

from ..local_opt import HillClimbingOptimizer


class Individual(HillClimbingOptimizer):
    def __init__(
        self,
        *args,
        rand_rest_p=0.03,
        self_adaptation=True,
        adaptation_window=10,
        adaptation_factor=1.22,  # 5th root of 10, standard ES factor
        target_success_rate=0.2,  # 1/5th rule
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rand_rest_p = rand_rest_p

        # Self-adaptation parameters
        self.self_adaptation = self_adaptation
        self.adaptation_window = adaptation_window
        self.adaptation_factor = adaptation_factor
        self.target_success_rate = target_success_rate

        # Individual's mutation strength (starts with inherited epsilon)
        self.mutation_strength = self.epsilon

        # Track success/failure history for adaptation
        self.success_history = deque(maxlen=self.adaptation_window)
        self.adaptation_counter = 0

    @HillClimbingOptimizer.track_new_pos
    @HillClimbingOptimizer.random_iteration
    def iterate(self):
        """Override to use individual's mutation strength and apply self-adaptation."""
        return self.move_climb(
            self.pos_current,
            epsilon=self.mutation_strength,
            distribution=self.distribution,
        )

    # @HillClimbingOptimizer.track_new_score
    def evaluate(self, score_new):
        """Override to track success and adapt mutation strength."""
        # Track whether this mutation was successful
        if self.self_adaptation and hasattr(self, "score_current"):
            success = score_new > self.score_current
            self.success_history.append(success)

            # Adapt mutation strength periodically
            self.adaptation_counter += 1
            if (
                self.adaptation_counter % self.adaptation_window == 0
                and len(self.success_history) >= self.adaptation_window
            ):
                self._adapt_mutation_strength()

        # Call parent evaluation
        HillClimbingOptimizer.evaluate(self, score_new)

    def _adapt_mutation_strength(self):
        """Apply 1/5th success rule to adapt mutation strength."""
        if not self.success_history:
            return

        # Calculate success rate over the window
        success_rate = sum(self.success_history) / len(self.success_history)

        # Apply 1/5th rule
        if success_rate > self.target_success_rate:
            # Too many successes - increase mutation strength (more exploration)
            self.mutation_strength *= self.adaptation_factor
        elif success_rate < self.target_success_rate:
            # Too few successes - decrease mutation strength (more exploitation)
            self.mutation_strength /= self.adaptation_factor
        # If success_rate ≈ target_success_rate, keep current mutation strength

        # Ensure mutation strength stays within reasonable bounds
        self.mutation_strength = np.clip(self.mutation_strength, 1e-6, 1.0)

    def get_adaptation_info(self):
        """Return current adaptation state for debugging/monitoring."""
        if not self.self_adaptation:
            return {"self_adaptation": False}

        success_rate = (
            sum(self.success_history) / len(self.success_history)
            if self.success_history
            else 0.0
        )

        return {
            "self_adaptation": True,
            "mutation_strength": self.mutation_strength,
            "success_rate": success_rate,
            "success_history_length": len(self.success_history),
            "adaptation_counter": self.adaptation_counter,
        }
