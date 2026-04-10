# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
AskTell mixin providing a step-by-step optimization interface.

This is the counterpart to search.py (the Search mixin). While Search
orchestrates a complete optimization loop internally, AskTell lets the
user drive the loop: ask() proposes a point, the user evaluates it
externally, and tell() feeds back the score.

The mixin delegates to CoreOptimizer's existing state machine
(_init_pos, _iterate, _evaluate_init, _evaluate, _finish_initialization)
without adding any of Search's orchestration overhead (progress bars,
memory caching, callbacks, stopping conditions, distributed execution).
"""

from __future__ import annotations

import math

import numpy as np


class AskTell:
    """Mixin that provides ask/tell methods on top of CoreOptimizer.

    Combined with an optimizer class via multiple inheritance, this mixin
    exposes two public methods that drive the optimizer one step at a time.

    The calls must alternate strictly: each ask() must be followed by
    exactly one tell() before the next ask(). Violating this raises
    RuntimeError.

    Usage (via a concrete ask_tell optimizer class)::

        from gradient_free_optimizers.ask_tell import HillClimbingOptimizer

        opt = HillClimbingOptimizer(search_space)
        for _ in range(100):
            params = opt.ask()
            score = my_function(params)
            opt.tell(params, score)

        print(opt.best_para, opt.best_score)
    """

    def __init__(self):
        super().__init__()
        self._pending_ask = False
        # User-visible best, tracked separately from the optimizer's internal
        # _pos_best/_score_best to avoid inflating their auto-appending
        # history lists with extra entries that the optimizer didn't produce.
        self._at_score_best = -math.inf
        self._at_pos_best = None

    def ask(self) -> dict:
        """Propose the next set of parameters to evaluate.

        During the initialization phase, returns positions from the
        initialization strategy (grid, random, vertices, warm start).
        After initialization is exhausted, switches to the optimizer's
        iteration strategy.

        Returns
        -------
        dict
            Parameter dictionary mapping names to values.

        Raises
        ------
        RuntimeError
            If called again before tell() was called for the previous ask().
        """
        if self._pending_ask:
            raise RuntimeError(
                "ask() called again before tell(). Each ask() must be "
                "followed by exactly one tell() before the next ask()."
            )

        if self.search_state == "init" and self.nth_init < self.init.n_inits:
            pos = self._init_pos()
        else:
            if self.search_state == "init":
                self._finish_initialization()
            pos = self._iterate()

        value = self.conv.position2value(pos)
        params = self.conv.value2para(value)
        self._pending_ask = True
        return params

    def tell(self, params: dict, score: float) -> None:
        """Report the evaluation result for the most recently asked parameters.

        Parameters
        ----------
        params : dict
            The parameter dictionary returned by the preceding ask() call.
            Must match exactly; passing different params raises ValueError.
        score : float
            The objective function value for those parameters. The optimizer
            always maximizes, so higher is better. For minimization problems,
            negate the score before passing it here.

        Raises
        ------
        RuntimeError
            If called without a preceding ask().
        ValueError
            If params does not match the preceding ask() result.
        """
        if not self._pending_ask:
            raise RuntimeError("tell() called without a preceding ask().")

        value = self.conv.para2value(params)
        pos = self.conv.value2position(value)
        if not np.array_equal(pos, self._pos_new):
            raise ValueError(
                "params passed to tell() do not match the params returned "
                "by the preceding ask(). You must pass back the exact dict."
            )

        if self.search_state == "init":
            self._evaluate_init(score)
        else:
            self._evaluate(score)

        # Track user-visible best on separate attributes so we don't
        # inflate _pos_best_list/_score_best_list with entries that the
        # optimizer's own logic didn't produce. We also set _best_para
        # and _best_value so CoreOptimizer.best_para/.best_value return
        # the true running maximum (some optimizers batch their internal
        # best-update and may lag between batches).
        if not math.isnan(score):
            if self._at_pos_best is None or score > self._at_score_best:
                self._at_score_best = score
                self._at_pos_best = self._pos_new.copy()
                best_value = self.conv.position2value(self._pos_new)
                self._best_para = self.conv.value2para(best_value)
                self._best_value = best_value

        self._pending_ask = False

    @property
    def best_score(self) -> float:
        """Best score found so far.

        Returns -inf if no evaluations have been performed yet.
        """
        return self._at_score_best
