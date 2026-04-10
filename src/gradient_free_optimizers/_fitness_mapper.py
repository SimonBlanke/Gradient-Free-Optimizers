"""Fitness mapping strategies for multi-objective optimization.

A FitnessMapper converts one or more objective values into a single scalar
fitness that the optimizer uses for all internal comparisons and decisions.
For single-objective optimization, the identity mapper passes the score
through unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FitnessMapper(ABC):
    """Convert objective value(s) to scalar fitness.

    Subclass this to implement custom scalarization strategies.
    The mapper is called once per evaluation with the raw objective
    value(s) and must return a single float.
    """

    @abstractmethod
    def __call__(self, objectives: float | list[float]) -> float: ...


class ScalarIdentity(FitnessMapper):
    """Pass-through for single-objective optimization.

    Returns the objective value unchanged. This is the default mapper
    when ``n_objectives=1``.
    """

    def __call__(self, objectives: float | list[float]) -> float:
        return float(objectives)


class WeightedSum(FitnessMapper):
    """Weighted linear combination of objectives.

    Computes ``sum(w_i * obj_i)`` as a scalar fitness. When no weights
    are provided, all objectives are weighted equally (1/n).

    Note that this scalarization cannot find solutions in concave
    regions of the Pareto front. For such cases, Tchebycheff
    decomposition or a true multi-objective algorithm (NSGA-II) is
    more appropriate.

    Parameters
    ----------
    weights : list[float] or None
        Per-objective weights. Must have the same length as the
        number of objectives. ``None`` means equal weights.
    n_objectives : int
        Number of objectives. Only used when ``weights`` is None.
    """

    def __init__(self, weights: list[float] | None = None, n_objectives: int = 2):
        if weights is not None:
            self.weights = np.array(weights, dtype=float)
        else:
            self.weights = np.ones(n_objectives, dtype=float) / n_objectives

    def __call__(self, objectives: float | list[float]) -> float:
        obj = np.asarray(objectives, dtype=float)
        return float(np.dot(self.weights, obj))
