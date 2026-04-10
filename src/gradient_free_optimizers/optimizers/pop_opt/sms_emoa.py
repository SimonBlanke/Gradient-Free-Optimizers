"""SMS-EMOA: S-Metric Selection Evolutionary Multi-Objective Algorithm.

Steady-state evolutionary multi-objective optimizer that uses the
hypervolume indicator (S-metric) for environmental selection.

Reference:
    Beume, Naujoks, Emmerich (2007). "SMS-EMOA: Multiobjective selection
    based on dominated hypervolume." European Journal of Operational Research.
"""

from __future__ import annotations

import math

import numpy as np

from ._individual import Individual
from .base_population_optimizer import BasePopulationOptimizer
from .nsga2 import non_dominated_sort


def hypervolume_2d(objectives: list[list[float]], ref: list[float]) -> float:
    """Exact 2D hypervolume for maximization.

    Computes the area of the union of rectangles [ref, p] for p in
    the point set. Works with any set of points (dominated or not).

    Parameters
    ----------
    objectives : list of [f1, f2]
        Objective vectors (higher is better).
    ref : [r1, r2]
        Reference point, must be dominated by all points.
    """
    if not objectives:
        return 0.0

    sorted_pts = sorted(objectives, key=lambda p: -p[0])

    hv = 0.0
    prev_f2 = ref[1]

    for f1, f2 in sorted_pts:
        if f2 > prev_f2:
            hv += (f1 - ref[0]) * (f2 - prev_f2)
            prev_f2 = f2

    return hv


def hypervolume_nd(objectives: list[list[float]], ref: list[float]) -> float:
    """General hypervolume via recursive slicing (HSO algorithm).

    Efficient for 2D (dispatches to sweep), reasonable for 3D on
    small fronts. Complexity is O(n^(d-1) log n) where d is the
    number of objectives.

    Parameters
    ----------
    objectives : list of list[float]
        Objective vectors (higher is better).
    ref : list[float]
        Reference point.
    """
    if not objectives:
        return 0.0

    n_obj = len(objectives[0])

    if n_obj == 1:
        return max(o[0] for o in objectives) - ref[0]

    if n_obj == 2:
        return hypervolume_2d(objectives, ref)

    # Recursive slicing on the last objective
    # Sort by last objective descending
    sorted_obj = sorted(objectives, key=lambda o: -o[n_obj - 1])

    hv = 0.0
    prev_last = None
    accumulated = []

    for point in sorted_obj:
        if prev_last is not None:
            slice_height = prev_last - point[n_obj - 1]
            if slice_height > 0 and accumulated:
                # Compute (d-1)-dimensional hypervolume of accumulated points
                proj = [o[: n_obj - 1] for o in accumulated]
                ref_proj = ref[: n_obj - 1]
                hv += slice_height * hypervolume_nd(proj, ref_proj)

        accumulated.append(point)
        prev_last = point[n_obj - 1]

    # Final slice from the lowest point to the reference
    if prev_last is not None and prev_last > ref[n_obj - 1]:
        proj = [o[: n_obj - 1] for o in accumulated]
        ref_proj = ref[: n_obj - 1]
        hv += (prev_last - ref[n_obj - 1]) * hypervolume_nd(proj, ref_proj)

    return hv


def hypervolume_contributions(
    front_objectives: list[list[float]], ref: list[float]
) -> list[float]:
    r"""Compute exclusive hypervolume contribution of each point.

    For 2D, uses an O(n log n) exact formula. For 3D+, computes
    HV(S) - HV(S \\ {p}) for each point p.

    Parameters
    ----------
    front_objectives : list of list[float]
        Mutually non-dominated objective vectors.
    ref : list[float]
        Reference point.

    Returns
    -------
    list[float]
        Contribution of each point (same order as input).
    """
    n = len(front_objectives)

    if n <= 1:
        return [math.inf] * n

    n_obj = len(front_objectives[0])

    if n_obj == 2:
        return _contributions_2d(front_objectives, ref)

    # General case: brute-force per-point removal
    total_hv = hypervolume_nd(front_objectives, ref)
    contributions = []
    for i in range(n):
        reduced = [o for j, o in enumerate(front_objectives) if j != i]
        contributions.append(total_hv - hypervolume_nd(reduced, ref))

    return contributions


def _contributions_2d(
    front_objectives: list[list[float]], ref: list[float]
) -> list[float]:
    """Exact 2D contributions via neighbor distances.

    For a non-dominated front sorted by f1 ascending (f2 descending),
    the exclusive contribution of point i is:
        (f1_i - f1_{i-1}) * (f2_i - f2_{i+1})
    with boundary values taken from the reference point.
    """
    n = len(front_objectives)
    indexed = sorted(range(n), key=lambda i: front_objectives[i][0])

    contributions = [0.0] * n

    for local_i, orig_i in enumerate(indexed):
        f1 = front_objectives[orig_i][0]
        f2 = front_objectives[orig_i][1]

        left_f1 = front_objectives[indexed[local_i - 1]][0] if local_i > 0 else ref[0]
        right_f2 = (
            front_objectives[indexed[local_i + 1]][1] if local_i < n - 1 else ref[1]
        )

        contributions[orig_i] = (f1 - left_f1) * (f2 - right_f2)

    return contributions


class _SMSEMOAOptimizer(BasePopulationOptimizer):
    """SMS-EMOA multi-objective optimizer.

    Steady-state algorithm that generates one offspring per iteration.
    After evaluation, the offspring is added to the population (size
    N+1), non-dominated sorting is applied, and the individual with
    the smallest hypervolume contribution in the worst front is
    removed, bringing the population back to size N.

    The hypervolume contribution measures how much dominated space a
    point exclusively covers. Removing the point with the smallest
    contribution preserves the overall hypervolume as much as possible,
    which drives the population toward a well-spread Pareto front.
    """

    name = "SMS-EMOA"
    _name_ = "sms_emoa"
    __name__ = "SMSEMOAOptimizer"

    def __init__(
        self,
        search_space,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        population=20,
        crossover_rate=0.9,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            population=population,
        )
        self.crossover_rate = crossover_rate

        self.individuals: list = []
        self._pop_objectives: list[list[float]] = []
        self._sms_new_pos = None
        self._batch_individual_refs: list = []

    def _on_init_pos(self, position):
        if not self.individuals:
            self.individuals = self._create_population(Individual)
            self.optimizers = self.individuals
            self.systems = self.individuals

        idx = (self.nth_init - 1) % len(self.individuals)
        self.p_current = self.individuals[idx]

    def _on_evaluate_init(self, score_new):
        self.p_current._pos_new = self._pos_new.copy()
        self.p_current._evaluate(score_new)
        objectives = getattr(self, "_last_objectives", None)
        self.p_current._sms_objectives = (
            list(objectives) if objectives is not None else [score_new]
        )

    def _on_finish_initialization(self):
        n_obj = getattr(self, "_n_objectives", 2)
        self._pop_objectives = []
        for ind in self.individuals:
            obj = getattr(ind, "_sms_objectives", None)
            if obj is not None and len(obj) == n_obj:
                self._pop_objectives.append(list(obj))
            else:
                score = (
                    ind._score_current if ind._score_current is not None else -math.inf
                )
                self._pop_objectives.append([score] * n_obj)

    def _setup_iteration(self):
        """Select parents via binary tournament on score, crossover, mutate."""
        rng = np.random.default_rng()
        pop_size = len(self.individuals)

        # Binary tournament (on scalar fitness / score_current)
        def _tournament():
            i, j = rng.choice(pop_size, size=2, replace=False)
            si = self.individuals[i]._score_current or -math.inf
            sj = self.individuals[j]._score_current or -math.inf
            return self.individuals[i] if si >= sj else self.individuals[j]

        parent_a = _tournament()
        parent_b = _tournament()

        pos_a = parent_a._pos_current
        pos_b = parent_b._pos_current

        if pos_a is None or pos_b is None:
            self.p_current = parent_a
            fallback = parent_a._pos_current
            self._sms_new_pos = (
                fallback.copy()
                if fallback is not None
                else self._clip_position(self.init.move_random_typed())
            )
            return

        n_dims = len(self.search_space)

        if rng.random() < self.crossover_rate:
            new_pos = np.empty(n_dims)
            mask = rng.random(n_dims) < 0.5
            new_pos[mask] = pos_a[mask]
            new_pos[~mask] = pos_b[~mask]
        else:
            new_pos = pos_a.copy()

        # Mutation
        self.p_current = parent_a
        self._sms_new_pos = parent_a.move_climb_typed(new_pos)

    def _iterate_continuous_batch(self):
        if self._sms_new_pos is None:
            self._setup_iteration()
        return self._sms_new_pos[self._continuous_mask]

    def _iterate_categorical_batch(self):
        if self._sms_new_pos is None:
            self._setup_iteration()
        return self._sms_new_pos[self._categorical_mask]

    def _iterate_discrete_batch(self):
        if self._sms_new_pos is None:
            self._setup_iteration()
        return self._sms_new_pos[self._discrete_mask]

    def _on_evaluate(self, score_new):
        """Steady-state selection: add offspring, remove worst by HV contribution."""
        n_obj = getattr(self, "_n_objectives", 2)
        objectives = getattr(self, "_last_objectives", None)
        if objectives is None or len(objectives) != n_obj:
            objectives = [score_new] * n_obj

        offspring_obj = list(objectives)
        offspring_pos = self._pos_new.copy()

        # Combined population: existing individuals + offspring
        all_obj = list(self._pop_objectives) + [offspring_obj]
        pop_size = len(self.individuals)
        offspring_idx = pop_size

        # Non-dominated sort on combined set (size N+1)
        fronts = non_dominated_sort(all_obj)

        # Identify which index to remove from the worst front
        worst_front = fronts[-1]

        if len(worst_front) == 1:
            remove_idx = worst_front[0]
        else:
            # Reference point: worst value per objective minus margin
            front_obj = [all_obj[i] for i in worst_front]
            ref = self._compute_reference(front_obj)
            contribs = hypervolume_contributions(front_obj, ref)

            # Remove the point with smallest contribution
            min_local = min(range(len(contribs)), key=lambda i: contribs[i])
            remove_idx = worst_front[min_local]

        if remove_idx == offspring_idx:
            # Offspring rejected
            pass
        else:
            # Replace individual at remove_idx with offspring
            ind = self.individuals[remove_idx]
            ind.__dict__["_CoreOptimizer__pos_current"] = offspring_pos.copy()
            ind.__dict__["_CoreOptimizer__score_current"] = score_new
            self._pop_objectives[remove_idx] = offspring_obj

        self._update_best(self._pos_new, score_new)
        self._update_current(self._pos_new, score_new)
        self._sms_new_pos = None

    def _compute_reference(self, front_obj: list[list[float]]) -> list[float]:
        """Reference point for hypervolume: worst per objective with margin."""
        n_obj = len(front_obj[0])
        ref = [math.inf] * n_obj
        for obj in front_obj:
            for j in range(n_obj):
                if obj[j] < ref[j]:
                    ref[j] = obj[j]

        # Small margin below the worst point to ensure all points
        # have positive hypervolume contribution
        for j in range(n_obj):
            span = max(o[j] for o in front_obj) - ref[j]
            ref[j] -= max(span * 0.1, 1e-6)

        return ref

    def _iterate_batch(self, n):
        positions = []
        self._batch_individual_refs = []
        for _ in range(n):
            self._setup_iteration()
            pos = self._generate_position()
            positions.append(pos)
            self._batch_individual_refs.append(self.p_current)
            self._sms_new_pos = None
        return positions

    def _evaluate_batch(self, positions, scores):
        for pos, score, indiv_ref in zip(
            positions, scores, self._batch_individual_refs
        ):
            self.p_current = indiv_ref
            self._pos_new = pos
            self._evaluate(score)
