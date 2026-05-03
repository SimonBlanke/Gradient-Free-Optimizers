# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""AutoOptimizer with adaptive, time-aware algorithm selection."""

from __future__ import annotations

import time

from gradient_free_optimizers._array_backend import ndarray

from ..global_opt import RandomSearchOptimizer
from ..local_opt import HillClimbingOptimizer, RepulsingHillClimbingOptimizer
from ._selection_strategy import DefaultStrategy, SelectionContext
from .base_population_optimizer import BasePopulationOptimizer, split

DEFAULT_PORTFOLIO = [
    RandomSearchOptimizer,
    HillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
]


class AutoOptimizer(BasePopulationOptimizer):
    """Automatic optimizer that selects among a portfolio of algorithms.

    Maintains a heterogeneous population of sub-optimizers and uses
    a configurable selection strategy to allocate iterations to the
    most effective algorithm at any given time. The default strategy
    uses time-weighted UCB1 selection, which naturally adapts to the
    cost of the objective function: cheap evaluations favor fast
    algorithms like HillClimbing, while expensive evaluations allow
    surrogate-based methods to compete.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to arrays of possible values.
    initialize : dict, default=None
        Strategy for generating initial positions.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random restart.
    nth_process : int, optional
        Process index for parallel optimization.
    portfolio : list of optimizer classes, optional
        Optimizer types to include in the portfolio. Each class is
        instantiated once with shared search_space and constraints.
        Defaults to RandomSearch, HillClimbing, RepulsingHillClimbing.
    strategy : SelectionStrategy, optional
        Strategy controlling which optimizer gets each iteration.
        Defaults to DefaultStrategy with time-weighted UCB1.
    """

    name = "Auto Optimizer"
    _name_ = "auto_optimizer"
    __name__ = "AutoOptimizer"

    optimizer_type = "population"
    computationally_expensive = False

    def __init__(
        self,
        search_space,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        portfolio=None,
        strategy=None,
    ):
        if portfolio is None:
            portfolio = list(DEFAULT_PORTFOLIO)

        if not portfolio:
            raise ValueError("portfolio must contain at least one optimizer class")

        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            population=len(portfolio),
        )

        if strategy is None:
            strategy = DefaultStrategy()

        self.strategy = strategy
        self._portfolio_types = portfolio

        self.systems = self._create_heterogeneous_population(portfolio)
        self.optimizers = self.systems

        self._selection_context = SelectionContext(len(self.systems))

        self._iteration_setup_done = False
        self._current_new_pos = None
        self._current_optimizer_idx = 0
        self._iter_start_time = None

    def _create_heterogeneous_population(self, optimizer_types):
        """Create one sub-optimizer per type with distributed init positions."""
        pop_size = len(optimizer_types)

        diff_init = pop_size - self.init.n_inits
        if diff_init > 0:
            self.init.add_n_random_init_pos(diff_init)

        distributed = split(self.init.init_positions_l, pop_size)

        population = []
        for OptimizerClass, init_positions in zip(optimizer_types, distributed):
            init_values = self.conv.positions2values(init_positions)
            init_paras = self.conv.values2paras(init_values)

            population.append(
                OptimizerClass(
                    self.conv.search_space,
                    rand_rest_p=self.rand_rest_p,
                    initialize={"warm_start": init_paras},
                    constraints=self.constraints,
                )
            )

        return population

    def _on_init_pos(self, position):
        """Assign initialization position to sub-optimizers via round-robin."""
        nth_pop = (self.nth_init - 1) % len(self.systems)
        self.p_current = self.systems[nth_pop]

        self.p_current._pos_new = position.copy()
        self.p_current._pos_current = position.copy()

    def _on_evaluate_init(self, score_new):
        """Delegate init evaluation to the current sub-optimizer."""
        self.p_current._score_new = score_new

        if self.p_current._pos_best is None or score_new > self.p_current._score_best:
            self.p_current._pos_best = self.p_current._pos_new.copy()
            self.p_current._score_best = score_new

        self.p_current._score_current = score_new

    def _setup_iteration(self):
        """Select a sub-optimizer and delegate iteration (lazy, called once)."""
        if self._iteration_setup_done:
            return

        idx = self.strategy.select(self._selection_context)
        self.p_current = self.systems[idx]
        self._current_optimizer_idx = idx

        self._iter_start_time = time.perf_counter()

        self._current_new_pos = self.p_current._iterate()
        self._iteration_setup_done = True

    def _iterate_continuous_batch(self) -> ndarray:
        """Generate continuous values from the selected sub-optimizer."""
        self._setup_iteration()
        return self._current_new_pos[self._continuous_mask]

    def _iterate_categorical_batch(self) -> ndarray:
        """Generate categorical indices from the selected sub-optimizer."""
        self._setup_iteration()
        return self._current_new_pos[self._categorical_mask]

    def _iterate_discrete_batch(self) -> ndarray:
        """Generate discrete indices from the selected sub-optimizer."""
        self._setup_iteration()
        return self._current_new_pos[self._discrete_mask]

    def _on_evaluate(self, score_new):
        """Update strategy, delegate to sub-optimizer, track global state.

        In single-iteration mode (_iter_start_time is set), computes elapsed
        time and updates the strategy. In batch mode (_iter_start_time is
        None), the strategy was already updated by _evaluate_batch.
        """
        if self._iter_start_time is not None:
            elapsed = time.perf_counter() - self._iter_start_time
            self.strategy.update(
                self._current_optimizer_idx,
                score_new,
                elapsed,
                self._selection_context,
            )

        self.p_current._evaluate(score_new)

        for system in self.systems:
            if system._score_best is not None and system._pos_best is not None:
                self._update_best(system._pos_best, system._score_best)

        if self.p_current._pos_current is not None:
            self._update_current(
                self.p_current._pos_current, self.p_current._score_current
            )

        self._iteration_setup_done = False
        self._current_new_pos = None
        self._iter_start_time = None

    def _iterate_batch(self, n):
        """Generate n positions using strategy-selected sub-optimizers."""
        positions = []
        self._batch_optimizer_indices = []
        self._batch_start_time = time.perf_counter()

        for _ in range(n):
            idx = self.strategy.select(self._selection_context)
            self._batch_optimizer_indices.append(idx)

            self.p_current = self.systems[idx]
            pos = self.p_current._iterate()
            positions.append(self._clip_position(pos))

        return positions

    def _evaluate_batch(self, positions, scores):
        """Process batch results, distributing elapsed time evenly."""
        total_elapsed = time.perf_counter() - self._batch_start_time
        per_eval = total_elapsed / len(scores) if scores else 0.0

        for i, (pos, score) in enumerate(zip(positions, scores)):
            idx = self._batch_optimizer_indices[i]
            self.p_current = self.systems[idx]
            self._current_optimizer_idx = idx

            self.strategy.update(idx, score, per_eval, self._selection_context)

            # Restore correct position on sub-optimizer before evaluation,
            # because _iterate_batch may have selected the same optimizer
            # multiple times, overwriting its _pos_new each time.
            self.p_current._pos_new = pos.copy()

            self._pos_new = pos
            self._evaluate(score)
