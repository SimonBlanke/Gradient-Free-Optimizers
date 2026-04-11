# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
AskTell mixin providing a batch-capable optimization interface.

This is the counterpart to search.py (the Search mixin). While Search
orchestrates a complete optimization loop internally, AskTell lets the
user drive the loop: ask() proposes positions, the user evaluates them
externally, and tell() feeds back the scores.

Initialization is separated from the optimization loop. The constructor
receives ``initial_evaluations`` (pre-evaluated points) and transitions
the optimizer to iteration state. After that, ask() and tell() operate
exclusively in the iteration phase, delegating to _iterate_batch() and
_evaluate_batch() on the optimizer.
"""

from __future__ import annotations

import math


class AskTell:
    """Mixin that provides batch ask/tell methods on top of CoreOptimizer.

    Combined with an optimizer class via multiple inheritance, this mixin
    exposes ``ask(n)`` and ``tell(params_list, scores)`` methods that
    drive the optimizer in batches.

    The optimizer must be initialized with ``initial_evaluations`` in the
    constructor. After initialization, the ask/tell loop operates
    exclusively in the iteration phase.

    Usage (via a concrete ask_tell optimizer class)::

        from gradient_free_optimizers.ask_tell import HillClimbingOptimizer

        opt = HillClimbingOptimizer(
            search_space,
            initial_evaluations=[
                ({"x": 0.5, "y": 1.0}, 0.8),
                ({"x": -3.0, "y": 0.0}, 0.2),
            ],
        )

        for _ in range(25):
            params_list = opt.ask(n=4)
            scores = [objective(p) for p in params_list]
            opt.tell(scores)

        print(opt.best_para, opt.best_score)

    Expects the host class to provide: conv, search_state, nth_init,
        _pos_new (setter), _on_init_pos(), _evaluate_init(),
        _finish_initialization(), _iterate_batch(), _evaluate_batch()
    """

    def __init__(self):
        super().__init__()
        self._pending_positions = None
        # User-visible best, tracked separately from the optimizer's internal
        # _pos_best/_score_best to avoid inflating their auto-appending
        # history lists with extra entries that the optimizer didn't produce.
        self._at_score_best = -math.inf
        self._at_pos_best = None

    def _process_initial_evaluations(self, initial_evaluations):
        """Feed pre-evaluated (params, score) pairs through the init machinery.

        Must be called after CoreOptimizer.__init__ has completed.
        Replays each evaluation through _pos_new setter, _on_init_pos hook,
        and _evaluate_init, then transitions to iteration state via
        _finish_initialization.

        Parameters
        ----------
        initial_evaluations : list[tuple[dict, float]]
            Each element is a (params_dict, score) pair. The params dict
            must contain keys matching the search space parameter names.
        """
        if not initial_evaluations:
            raise ValueError(
                "initial_evaluations must not be empty. Provide at least "
                "one (params_dict, score) tuple."
            )

        min_required = self._min_initial_evaluations()
        if len(initial_evaluations) < min_required:
            raise ValueError(
                f"{self.__class__.__name__} requires at least "
                f"{min_required} initial evaluations (got "
                f"{len(initial_evaluations)}). Population-based "
                f"optimizers need one evaluation per member."
            )

        for params, score in initial_evaluations:
            value = self.conv.para2value(params)
            pos = self.conv.value2position(value)
            self._pos_new = pos
            self.nth_init += 1
            self._on_init_pos(pos)
            self._evaluate_init(score)

            if not math.isnan(score):
                if self._at_pos_best is None or score > self._at_score_best:
                    self._at_score_best = score
                    self._at_pos_best = pos.copy()

        self._finish_initialization()

        if self._at_pos_best is not None:
            best_value = self.conv.position2value(self._at_pos_best)
            self._best_para = self.conv.value2para(best_value)
            self._best_value = best_value

    def _min_initial_evaluations(self):
        """Minimum number of initial evaluations this optimizer requires.

        Population-based optimizers need at least one evaluation per
        member so every sub-optimizer has a starting point.
        """
        if hasattr(self, "optimizers") and self.optimizers:
            return len(self.optimizers)
        return 1

    def ask(self, n: int = 1) -> list[dict]:
        """Propose n parameter sets to evaluate.

        Parameters
        ----------
        n : int, default=1
            Number of parameter sets to generate. The optimizer uses
            ``_iterate_batch(n)`` internally, which may leverage
            algorithm-specific batch generation (e.g. diverse acquisition
            points for SMBO, population cycling for PSO/GA).

        Returns
        -------
        list[dict]
            List of n parameter dictionaries to evaluate.

        Raises
        ------
        RuntimeError
            If the optimizer has not been initialized (no
            initial_evaluations provided) or if ask() is called
            before tell() returned results for the previous ask().
        """
        if self.search_state != "iter":
            raise RuntimeError(
                "Optimizer not initialized. Provide initial_evaluations "
                "in the constructor before calling ask()."
            )
        if self._pending_positions is not None:
            raise RuntimeError(
                "ask() called again before tell(). Each ask() must be "
                "followed by exactly one tell() before the next ask()."
            )

        positions = self._iterate_batch(n)
        self._pending_positions = positions

        params_list = []
        for pos in positions:
            if self.conv.conditions:
                filtered, _ = self.conv.get_active_params(pos)
                params_list.append(filtered)
            else:
                value = self.conv.position2value(pos)
                params_list.append(self.conv.value2para(value))

        return params_list

    def tell(self, scores: list[float]) -> None:
        """Report evaluation results for the most recently asked positions.

        The scores must correspond to the parameter sets returned by the
        preceding ask(), in the same order.

        Parameters
        ----------
        scores : list[float]
            The objective function scores for each parameter set.
            Higher is better. For minimization problems, negate the
            scores before passing them here.

        Raises
        ------
        RuntimeError
            If called without a preceding ask().
        ValueError
            If the number of scores doesn't match the preceding ask().
        """
        if self._pending_positions is None:
            raise RuntimeError("tell() called without a preceding ask().")

        if len(scores) != len(self._pending_positions):
            raise ValueError(
                f"Expected {len(self._pending_positions)} scores, "
                f"got {len(scores)}."
            )

        self._evaluate_batch(self._pending_positions, scores)

        for pos, score in zip(self._pending_positions, scores):
            if not math.isnan(score):
                if self._at_pos_best is None or score > self._at_score_best:
                    self._at_score_best = score
                    self._at_pos_best = pos.copy()
                    best_value = self.conv.position2value(pos)
                    self._best_para = self.conv.value2para(best_value)
                    self._best_value = best_value

        self._pending_positions = None

    @property
    def best_score(self) -> float:
        """Best score found so far.

        Returns -inf if no evaluations have been performed yet.
        """
        return self._at_score_best
