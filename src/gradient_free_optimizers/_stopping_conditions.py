import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class StoppingContext:
    """
    Encapsulates all relevant data for stopping condition evaluation.
    This creates a clear contract for what data stopping conditions can access.
    """

    iteration: int
    score_current: float
    score_best: float
    score_history: List[float]
    start_time: float
    current_time: float

    @property
    def elapsed_time(self) -> float:
        """Time elapsed since optimization started."""
        return self.current_time - self.start_time

    @property
    def iterations_since_improvement(self) -> int:
        """Number of iterations since the best score was found."""
        if not self.score_history:
            return 0

        best_score_idx = np.argmax(self.score_history)
        return len(self.score_history) - best_score_idx - 1


class StoppingCondition(ABC):
    """
    Abstract base class for all stopping conditions.
    Each condition is responsible for a single stopping criterion.
    """

    def __init__(self, name: str):
        self.name = name
        self.triggered = False
        self.trigger_reason = ""
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    @abstractmethod
    def should_stop(self, context: StoppingContext) -> bool:
        """Check if the optimization should stop based on this condition."""
        pass

    @abstractmethod
    def get_debug_info(self, context: StoppingContext) -> Dict[str, Any]:
        """Return detailed information for debugging purposes."""
        pass

    def reset(self):
        """Reset the condition to its initial state."""
        self.triggered = False
        self.trigger_reason = ""


class TimeExceededCondition(StoppingCondition):
    """Stops when maximum time limit is exceeded."""

    def __init__(self, max_time: Optional[float]):
        super().__init__("TimeExceeded")
        self.max_time = max_time

    def should_stop(self, context: StoppingContext) -> bool:
        if self.max_time is None:
            return False

        if context.elapsed_time > self.max_time:
            self.triggered = True
            self.trigger_reason = f"Time limit exceeded: {context.elapsed_time:.2f}s > {self.max_time:.2f}s"
            self.logger.info(self.trigger_reason)
            return True
        return False

    def get_debug_info(self, context: StoppingContext) -> Dict[str, Any]:
        return {
            "condition": self.name,
            "max_time": self.max_time,
            "elapsed_time": context.elapsed_time,
            "time_remaining": (
                self.max_time - context.elapsed_time if self.max_time else None
            ),
            "triggered": self.triggered,
            "reason": self.trigger_reason,
        }


class ScoreExceededCondition(StoppingCondition):
    """Stops when target score is reached or exceeded."""

    def __init__(self, max_score: Optional[float]):
        super().__init__("ScoreExceeded")
        self.max_score = max_score

    def should_stop(self, context: StoppingContext) -> bool:
        if self.max_score is None:
            return False

        if context.score_best >= self.max_score:
            self.triggered = True
            self.trigger_reason = f"Target score reached: {context.score_best:.6f} >= {self.max_score:.6f}"
            self.logger.info(self.trigger_reason)
            return True
        return False

    def get_debug_info(self, context: StoppingContext) -> Dict[str, Any]:
        return {
            "condition": self.name,
            "max_score": self.max_score,
            "current_best_score": context.score_best,
            "score_gap": (
                self.max_score - context.score_best if self.max_score else None
            ),
            "triggered": self.triggered,
            "reason": self.trigger_reason,
        }


class NoImprovementCondition(StoppingCondition):
    """Stops when no improvement is observed for a specified number of iterations."""

    def __init__(
        self,
        n_iter_no_change: int,
        tol_abs: Optional[float] = None,
        tol_rel: Optional[float] = None,
    ):
        super().__init__("NoImprovement")
        self.n_iter_no_change = n_iter_no_change
        self.tol_abs = tol_abs
        self.tol_rel = tol_rel

    def should_stop(self, context: StoppingContext) -> bool:
        if len(context.score_history) <= self.n_iter_no_change:
            return False

        iterations_stale = context.iterations_since_improvement

        if iterations_stale >= self.n_iter_no_change:
            self.triggered = True
            self.trigger_reason = f"No improvement for {iterations_stale} iterations"
            self.logger.info(self.trigger_reason)
            return True

        # Check tolerance-based early stopping
        first_n = len(context.score_history) - self.n_iter_no_change
        scores_before = context.score_history[:first_n]

        if not scores_before:
            return False

        max_score_before = max(scores_before)
        current_best = context.score_best

        # Absolute tolerance check
        if self.tol_abs is not None:
            improvement = abs(current_best - max_score_before)
            if improvement < self.tol_abs:
                self.triggered = True
                self.trigger_reason = f"Improvement below absolute tolerance: {improvement:.6f} < {self.tol_abs:.6f}"
                self.logger.info(self.trigger_reason)
                return True

        # Relative tolerance check
        if self.tol_rel is not None and max_score_before != 0:
            improvement_pct = (
                (current_best - max_score_before) / abs(max_score_before)
            ) * 100
            if improvement_pct < self.tol_rel:
                self.triggered = True
                self.trigger_reason = f"Improvement below relative tolerance: {improvement_pct:.2f}% < {self.tol_rel:.2f}%"
                self.logger.info(self.trigger_reason)
                return True

        return False

    def get_debug_info(self, context: StoppingContext) -> Dict[str, Any]:
        iterations_stale = context.iterations_since_improvement

        debug_info = {
            "condition": self.name,
            "n_iter_no_change": self.n_iter_no_change,
            "iterations_since_improvement": iterations_stale,
            "tol_abs": self.tol_abs,
            "tol_rel": self.tol_rel,
            "triggered": self.triggered,
            "reason": self.trigger_reason,
        }

        if len(context.score_history) > self.n_iter_no_change:
            first_n = len(context.score_history) - self.n_iter_no_change
            scores_before = context.score_history[:first_n]
            if scores_before:
                max_score_before = max(scores_before)
                improvement = context.score_best - max_score_before
                debug_info["improvement_abs"] = improvement
                if max_score_before != 0:
                    debug_info["improvement_rel_pct"] = (
                        improvement / abs(max_score_before)
                    ) * 100

        return debug_info


class CompositeStoppingCondition(StoppingCondition):
    """Combines multiple stopping conditions with OR logic."""

    def __init__(self, conditions: List[StoppingCondition]):
        super().__init__("Composite")
        self.conditions = conditions

    def should_stop(self, context: StoppingContext) -> bool:
        for condition in self.conditions:
            if condition.should_stop(context):
                self.triggered = True
                self.trigger_reason = (
                    f"Stopped by {condition.name}: {condition.trigger_reason}"
                )
                self.logger.info(self.trigger_reason)
                return True
        return False

    def get_debug_info(self, context: StoppingContext) -> Dict[str, Any]:
        return {
            "condition": self.name,
            "triggered": self.triggered,
            "reason": self.trigger_reason,
            "sub_conditions": [
                condition.get_debug_info(context) for condition in self.conditions
            ],
        }

    def reset(self):
        super().reset()
        for condition in self.conditions:
            condition.reset()


class OptimizationStopper:
    """
    Main class for managing optimization stopping conditions.
    Provides a clean interface and comprehensive debugging capabilities.
    """

    def __init__(
        self,
        start_time: float,
        max_time: Optional[float] = None,
        max_score: Optional[float] = None,
        early_stopping: Optional[Dict[str, Any]] = None,
    ):
        self.start_time = start_time
        self.conditions: List[StoppingCondition] = []
        self.score_history: List[float] = []
        self.score_best = -np.inf
        self.iteration = 0
        self.logger = logging.getLogger(f"{__name__}.OptimizationStopper")

        # Build stopping conditions
        if max_time is not None:
            self.conditions.append(TimeExceededCondition(max_time))

        if max_score is not None:
            self.conditions.append(ScoreExceededCondition(max_score))

        if early_stopping is not None:
            n_iter = early_stopping.get("n_iter_no_change")
            if n_iter is not None:
                self.conditions.append(
                    NoImprovementCondition(
                        n_iter_no_change=n_iter,
                        tol_abs=early_stopping.get("tol_abs"),
                        tol_rel=early_stopping.get("tol_rel"),
                    )
                )

        self.composite_condition = CompositeStoppingCondition(self.conditions)

    def update(self, score_current: float, score_best: float, iteration: int):
        """Update the stopper with new optimization state."""
        self.score_history.append(score_current)
        self.score_best = score_best
        self.iteration = iteration

    def should_stop(self) -> bool:
        """Check if optimization should stop."""
        context = StoppingContext(
            iteration=self.iteration,
            score_current=self.score_history[-1] if self.score_history else -np.inf,
            score_best=self.score_best,
            score_history=self.score_history,
            start_time=self.start_time,
            current_time=time.time(),
        )

        return self.composite_condition.should_stop(context)

    def get_debug_info(self) -> Dict[str, Any]:
        """Get comprehensive debugging information about stopping conditions."""
        context = StoppingContext(
            iteration=self.iteration,
            score_current=self.score_history[-1] if self.score_history else -np.inf,
            score_best=self.score_best,
            score_history=self.score_history,
            start_time=self.start_time,
            current_time=time.time(),
        )

        return self.composite_condition.get_debug_info(context)

    def get_stop_reason(self) -> str:
        """Get a human-readable reason for why optimization stopped."""
        if self.composite_condition.triggered:
            return self.composite_condition.trigger_reason
        return "Optimization not stopped by stopper"
