# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Selection strategies for the AutoOptimizer.

Strategies control which sub-optimizer receives the next iteration.
They observe evaluation results and adapt their selection over time.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class SelectionContext:
    """Runtime context tracking per-optimizer performance statistics.

    Updated by the AutoOptimizer after each evaluation and passed
    to the strategy for selection decisions.

    Parameters
    ----------
    n_optimizers : int
        Number of sub-optimizers in the portfolio.
    """

    def __init__(self, n_optimizers):
        self.n_optimizers = n_optimizers

        self.n_evals = [0] * n_optimizers
        self.total_time = [0.0] * n_optimizers
        self.best_scores = [None] * n_optimizers
        self.total_improvement = [0.0] * n_optimizers

        self.global_best_score = None
        self.nth_iter = 0


class SelectionStrategy(ABC):
    """Base class for optimizer selection strategies.

    Subclasses implement ``select`` to choose which sub-optimizer
    gets the next iteration, and ``update`` to process evaluation
    results.
    """

    @abstractmethod
    def select(self, context: SelectionContext) -> int:
        """Return the index of the sub-optimizer for the next iteration."""

    @abstractmethod
    def update(
        self,
        optimizer_idx: int,
        score: float,
        elapsed: float,
        context: SelectionContext,
    ) -> None:
        """Process an evaluation result.

        Parameters
        ----------
        optimizer_idx : int
            Index of the sub-optimizer that produced this result.
        score : float
            Score returned by the objective function.
        elapsed : float
            Wall-clock seconds for the full iteration (optimizer
            overhead plus objective evaluation).
        context : SelectionContext
            Shared context to update with the new observation.
        """


class DefaultStrategy(SelectionStrategy):
    """Time-weighted selection with UCB1 exploration bonus.

    Gathers baseline data via round-robin during a warmup phase,
    then selects the sub-optimizer with the highest time-weighted
    efficiency score. The efficiency of optimizer *i* is defined as
    the total improvement it achieved divided by the total wall-clock
    time it consumed. A UCB1-style exploration term prevents any
    optimizer from being permanently starved.

    The time-weighting is the key property: an optimizer that produces
    small improvements very quickly (like HillClimbing on cheap
    objective functions) will be preferred over one that produces
    larger improvements but takes much longer per step (like
    BayesianOptimizer with its surrogate model fitting overhead).

    Parameters
    ----------
    min_rounds : int, default=3
        Minimum evaluations per optimizer before switching from
        round-robin warmup to adaptive selection.
    exploration : float, default=0.5
        UCB1 exploration coefficient. Higher values explore
        underused optimizers more aggressively.
    """

    def __init__(self, min_rounds=3, exploration=0.5):
        self.min_rounds = min_rounds
        self.exploration = exploration

    def select(self, context):
        under_min = [
            i
            for i in range(context.n_optimizers)
            if context.n_evals[i] < max(self.min_rounds, 1)
        ]
        if under_min:
            return min(under_min, key=lambda i: context.n_evals[i])

        total_n = sum(context.n_evals)
        best_idx = 0
        best_ucb = float("-inf")

        for i in range(context.n_optimizers):
            n_i = context.n_evals[i]
            t_i = context.total_time[i]

            efficiency = context.total_improvement[i] / t_i if t_i > 0 else 0.0

            exploration_bonus = self.exploration * math.sqrt(math.log(total_n) / n_i)

            ucb = efficiency + exploration_bonus
            if ucb > best_ucb:
                best_ucb = ucb
                best_idx = i

        return best_idx

    def update(self, optimizer_idx, score, elapsed, context):
        context.n_evals[optimizer_idx] += 1
        context.total_time[optimizer_idx] += elapsed
        context.nth_iter += 1

        prev_best = context.best_scores[optimizer_idx]
        if prev_best is not None and score > prev_best:
            context.total_improvement[optimizer_idx] += score - prev_best

        if prev_best is None or score > prev_best:
            context.best_scores[optimizer_idx] = score

        if context.global_best_score is None or score > context.global_best_score:
            context.global_best_score = score
