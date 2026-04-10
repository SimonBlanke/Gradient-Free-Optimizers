"""NSGA-II: Non-dominated Sorting Genetic Algorithm II.

Implements the algorithm from:
    Deb, Pratap, Agarwal, Meyarivan (2002).
    "A fast and elitist multiobjective genetic algorithm: NSGA-II."

Uses SBX (Simulated Binary Crossover) and Polynomial Mutation as the
canonical reproduction operators from the original paper, combined
with non-dominated sorting and crowding distance for environmental
selection.
"""

from __future__ import annotations

import math

import numpy as np

from ._individual import Individual
from .base_population_optimizer import BasePopulationOptimizer


def non_dominated_sort(objectives_list):
    """Assign non-dominated ranks (fronts) to a set of solutions.

    Parameters
    ----------
    objectives_list : list[list[float]]
        Objective vectors for each solution. All objectives are
        maximized (higher is better), matching GFO's internal convention.

    Returns
    -------
    list[list[int]]
        List of fronts, each front is a list of solution indices.
        fronts[0] is the Pareto-optimal front.
    """
    n = len(objectives_list)
    if n == 0:
        return []

    obj = np.array(objectives_list)
    domination_count = np.zeros(n, dtype=int)
    dominated_by = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(obj[i], obj[j]):
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif _dominates(obj[j], obj[i]):
                dominated_by[j].append(i)
                domination_count[i] += 1

    fronts = []
    current_front = [i for i in range(n) if domination_count[i] == 0]

    while current_front:
        fronts.append(current_front)
        next_front = []
        for i in current_front:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front = next_front

    return fronts


def _dominates(a, b):
    """True if solution a dominates b (all >=, at least one >)."""
    return np.all(a >= b) and np.any(a > b)


def crowding_distance(objectives_list, front_indices):
    """Compute crowding distance for solutions within a single front.

    Parameters
    ----------
    objectives_list : np.ndarray, shape (n_total, n_objectives)
        All objective vectors.
    front_indices : list[int]
        Indices of solutions in this front.

    Returns
    -------
    dict[int, float]
        Mapping from solution index to crowding distance.
    """
    obj = np.array(objectives_list)
    n_front = len(front_indices)
    if n_front <= 2:
        return {idx: math.inf for idx in front_indices}

    distances = {idx: 0.0 for idx in front_indices}
    n_objectives = obj.shape[1]

    for m in range(n_objectives):
        sorted_indices = sorted(front_indices, key=lambda i: obj[i, m])
        distances[sorted_indices[0]] = math.inf
        distances[sorted_indices[-1]] = math.inf

        obj_min = obj[sorted_indices[0], m]
        obj_max = obj[sorted_indices[-1], m]
        if not np.isfinite(obj_min) or not np.isfinite(obj_max):
            continue
        obj_range = obj_max - obj_min
        if obj_range == 0:
            continue

        for k in range(1, n_front - 1):
            distances[sorted_indices[k]] += (
                obj[sorted_indices[k + 1], m] - obj[sorted_indices[k - 1], m]
            ) / obj_range

    return distances


class _NSGA2Optimizer(BasePopulationOptimizer):
    """NSGA-II multi-objective optimizer.

    Maintains a population evolved using tournament selection (based on
    non-dominated rank and crowding distance), SBX crossover, and
    polynomial mutation.

    After each generation (population_size evaluations), the combined
    parent+offspring population is reduced back to population_size using
    non-dominated sorting and crowding distance selection.
    """

    name = "NSGA-II"
    _name_ = "nsga2"
    __name__ = "NSGA2Optimizer"

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
        crossover_eta=20.0,
        mutation_eta=20.0,
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
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta

        self.individuals = []
        self._offspring_buffer = []
        self._offspring_refs = []
        self._gen_counter = 0

        self._pop_objectives = []

        self._nsga2_new_pos = None
        self._batch_individual_refs = []

    def _on_init_pos(self, position):
        """Route init positions to individuals round-robin."""
        if not self.individuals:
            self.individuals = self._create_population(Individual)
            self.optimizers = self.individuals
            self.systems = self.individuals

        idx = (self.nth_init - 1) % len(self.individuals)
        self.p_current = self.individuals[idx]

    def _on_evaluate_init(self, score_new):
        """Feed init score to the current individual and store objectives."""
        self.p_current._pos_new = self._pos_new.copy()
        self.p_current._evaluate(score_new)
        # Store objectives on the individual for later use in selection
        objectives = getattr(self, "_last_objectives", None)
        self.p_current._nsga2_objectives = (
            objectives if objectives is not None else [score_new]
        )

    def _on_finish_initialization(self):
        """After init: collect per-individual objectives."""
        n_obj = getattr(self, "_n_objectives", 1)
        self._pop_objectives = []
        for ind in self.individuals:
            obj = getattr(ind, "_nsga2_objectives", None)
            if obj is not None and len(obj) == n_obj:
                self._pop_objectives.append(obj)
            else:
                # Fallback: fill with score repeated to match n_objectives
                score = (
                    ind._score_current if ind._score_current is not None else -math.inf
                )
                self._pop_objectives.append([score] * n_obj)
        self._gen_counter = 0

    def _sbx(self, p1, p2, bl, bu, rng):
        """Simulated Binary Crossover (vectorized).

        Produces one child from two parents. The distribution index
        (crossover_eta) controls spread: higher values keep children
        closer to parents. Standard value is 20.
        """
        eta = self.crossover_eta
        n = len(p1)

        # Per-variable: 50% chance to apply SBX, otherwise copy a random parent
        do_xover = rng.random(n) < 0.5
        same = np.abs(p1 - p2) < 1e-14

        pick_a = rng.random(n) < 0.5
        child = np.where(pick_a, p1, p2)

        active = do_xover & ~same
        if not active.any():
            return child

        u = rng.random(n)
        beta = np.where(
            u <= 0.5,
            (2.0 * u) ** (1.0 / (eta + 1.0)),
            (1.0 / (2.0 * (1.0 - u + 1e-30))) ** (1.0 / (eta + 1.0)),
        )

        c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)

        pick_c1 = rng.random(n) < 0.5
        sbx_child = np.clip(np.where(pick_c1, c1, c2), bl, bu)

        child[active] = sbx_child[active]
        return child

    def _poly_mut(self, x, bl, bu, rng):
        """Polynomial Mutation (vectorized).

        Each variable is mutated independently with probability 1/n.
        The distribution index (mutation_eta) controls perturbation
        magnitude. Standard value is 20.
        """
        eta = self.mutation_eta
        n = len(x)
        prob = 1.0 / max(n, 1)

        result = x.copy()
        do_mutate = rng.random(n) < prob
        delta_max = bu - bl
        active = do_mutate & (delta_max > 1e-14)

        if not active.any():
            return result

        u = rng.random(n)
        delta = np.where(
            u < 0.5,
            (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0,
            1.0 - (2.0 * (1.0 - u + 1e-30)) ** (1.0 / (eta + 1.0)),
        )

        result[active] = x[active] + delta[active] * delta_max[active]
        return np.clip(result, bl, bu)

    def _integer_mutation(self, x, bl, bu, rng):
        """Integer-aware mutation for discrete dimensions.

        Uses a geometric distribution for step sizes (expected ~1-2 indices),
        guaranteeing the mutated value actually differs from the original.
        Polynomial mutation on discrete spaces with high eta often produces
        perturbations that round to zero, making it ineffective.
        """
        n = len(x)
        prob = 1.0 / max(n, 1)
        result = x.copy()

        for i in range(n):
            if rng.random() >= prob:
                continue
            step = int(rng.geometric(0.5)) * (1 if rng.random() < 0.5 else -1)
            result[i] = np.clip(result[i] + step, bl[i], bu[i])
            # If clipping cancelled the step, try the other direction
            if result[i] == x[i]:
                if x[i] < bu[i]:
                    result[i] = x[i] + 1
                elif x[i] > bl[i]:
                    result[i] = x[i] - 1

        return result

    def _setup_iteration(self):
        """Select parents, apply SBX crossover and polynomial mutation."""
        parent_a = self._binary_tournament()
        parent_b = self._binary_tournament()

        pos_a = parent_a._pos_current
        pos_b = parent_b._pos_current

        if pos_a is None or pos_b is None:
            self.p_current = parent_a
            fallback = parent_a._pos_current
            self._nsga2_new_pos = (
                fallback.copy()
                if fallback is not None
                else self._clip_position(self.init.move_random_typed())
            )
            return

        rng = np.random.default_rng()
        n_dims = len(self.search_space)
        new_pos = np.empty(n_dims)

        # Continuous dimensions: SBX crossover + polynomial mutation
        if self._continuous_mask is not None and self._continuous_mask.any():
            ca = pos_a[self._continuous_mask]
            cb = pos_b[self._continuous_mask]
            bl = self._continuous_bounds[:, 0]
            bu = self._continuous_bounds[:, 1]

            child = self._sbx(ca, cb, bl, bu, rng)
            new_pos[self._continuous_mask] = self._poly_mut(child, bl, bu, rng)

        # Discrete dimensions: SBX on index space + integer-aware mutation
        if self._discrete_mask is not None and self._discrete_mask.any():
            da = pos_a[self._discrete_mask].astype(float)
            db = pos_b[self._discrete_mask].astype(float)
            bl = self._discrete_bounds[:, 0].astype(float)
            bu = self._discrete_bounds[:, 1].astype(float)

            child = self._sbx(da, db, bl, bu, rng)
            child = np.round(child).astype(int)
            bl_int = self._discrete_bounds[:, 0]
            bu_int = self._discrete_bounds[:, 1]
            new_pos[self._discrete_mask] = self._integer_mutation(
                child,
                bl_int,
                bu_int,
                rng,
            )

        # Categorical dimensions: uniform crossover (SBX not applicable)
        if self._categorical_mask is not None and self._categorical_mask.any():
            ka = pos_a[self._categorical_mask]
            kb = pos_b[self._categorical_mask]
            pick = rng.random(len(ka)) < 0.5
            new_pos[self._categorical_mask] = np.where(pick, ka, kb)

        self.p_current = parent_a
        self._nsga2_new_pos = new_pos

    def _iterate_continuous_batch(self):
        if self._nsga2_new_pos is None:
            self._setup_iteration()
        return self._nsga2_new_pos[self._continuous_mask]

    def _iterate_categorical_batch(self):
        if self._nsga2_new_pos is None:
            self._setup_iteration()
        return self._nsga2_new_pos[self._categorical_mask]

    def _iterate_discrete_batch(self):
        if self._nsga2_new_pos is None:
            self._setup_iteration()
        return self._nsga2_new_pos[self._discrete_mask]

    def _on_evaluate(self, score_new):
        """Process one offspring evaluation.

        Triggers selection after a full generation.
        """
        n_obj = getattr(self, "_n_objectives", 1)
        objectives = getattr(self, "_last_objectives", None)
        if objectives is None or len(objectives) != n_obj:
            objectives = [score_new] * n_obj

        self._offspring_buffer.append((self._pos_new.copy(), score_new, objectives))
        self._gen_counter += 1

        self._update_best(self._pos_new, score_new)
        self._update_current(self._pos_new, score_new)

        # Reset so next _iterate triggers a fresh _setup_iteration
        self._nsga2_new_pos = None

        # Full generation complete: environmental selection
        if self._gen_counter >= self.population:
            self._environmental_selection()
            self._gen_counter = 0
            self._offspring_buffer = []

    def _environmental_selection(self):
        """Combine parents + offspring, non-dominated sort, select top N."""
        pop_size = self.population
        n_obj = getattr(self, "_n_objectives", 1)
        combined_pos = []
        combined_scores = []
        combined_objectives = []

        for i, ind in enumerate(self.individuals):
            combined_pos.append(
                ind._pos_current.copy()
                if ind._pos_current is not None
                else self.init.move_random_typed()
            )
            score = ind._score_current if ind._score_current is not None else -math.inf
            combined_scores.append(score)
            if i < len(self._pop_objectives) and len(self._pop_objectives[i]) == n_obj:
                combined_objectives.append(self._pop_objectives[i])
            else:
                combined_objectives.append([score] * n_obj)

        # Offspring
        for pos, score, obj in self._offspring_buffer:
            combined_pos.append(pos)
            combined_scores.append(score)
            combined_objectives.append(obj)

        # Non-dominated sorting on combined population
        fronts = non_dominated_sort(combined_objectives)

        # Select the best pop_size solutions
        selected_indices = []
        for front in fronts:
            if len(selected_indices) + len(front) <= pop_size:
                selected_indices.extend(front)
            else:
                remaining = pop_size - len(selected_indices)
                cd = crowding_distance(np.array(combined_objectives), front)
                front_sorted = sorted(front, key=lambda i: cd.get(i, 0), reverse=True)
                selected_indices.extend(front_sorted[:remaining])
                break

        # Update individuals with selected solutions
        for i, sel_idx in enumerate(selected_indices):
            if i < len(self.individuals):
                ind = self.individuals[i]
                ind.__dict__["_CoreOptimizer__pos_current"] = combined_pos[
                    sel_idx
                ].copy()
                ind.__dict__["_CoreOptimizer__score_current"] = combined_scores[sel_idx]

        self._pop_objectives = [combined_objectives[idx] for idx in selected_indices]

    def _binary_tournament(self):
        """Select one individual via binary tournament on rank + crowding."""
        rng = np.random.default_rng()
        pop_size = len(self.individuals)

        if not self._pop_objectives or len(self._pop_objectives) < pop_size:
            return self.individuals[rng.integers(pop_size)]

        fronts = non_dominated_sort(self._pop_objectives)
        rank = [0] * pop_size
        for front_idx, front in enumerate(fronts):
            for i in front:
                if i < pop_size:
                    rank[i] = front_idx

        i, j = rng.choice(pop_size, size=2, replace=False)

        if rank[i] < rank[j]:
            return self.individuals[i]
        elif rank[j] < rank[i]:
            return self.individuals[j]
        else:
            # Same rank: compare crowding distance
            front = [k for k in range(pop_size) if rank[k] == rank[i]]
            cd = crowding_distance(np.array(self._pop_objectives), front)
            if cd.get(i, 0) >= cd.get(j, 0):
                return self.individuals[i]
            return self.individuals[j]

    def _iterate_batch(self, n):
        positions = []
        self._batch_individual_refs = []
        for _ in range(n):
            self._setup_iteration()
            pos = self._generate_position()
            positions.append(pos)
            self._batch_individual_refs.append(self.p_current)
        return positions

    def _evaluate_batch(self, positions, scores):
        for pos, score, indiv_ref in zip(
            positions, scores, self._batch_individual_refs
        ):
            self.p_current = indiv_ref
            self._pos_new = pos
            self._evaluate(score)
