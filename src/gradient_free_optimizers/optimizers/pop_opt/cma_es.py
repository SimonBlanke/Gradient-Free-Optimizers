# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) Optimizer.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL

Template Method Pattern Compliance:
    - Does NOT override iterate() - uses CoreOptimizer's orchestration
    - Implements _iterate_*_batch() for dimension-type-aware position generation
    - Uses compute-once-extract-thrice pattern (like DE)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base_population_optimizer import BasePopulationOptimizer

if TYPE_CHECKING:
    pass


class CMAESOptimizer(BasePopulationOptimizer):
    pass
