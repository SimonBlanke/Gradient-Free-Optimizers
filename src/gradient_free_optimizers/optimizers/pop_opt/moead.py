"""MOEA/D: Multi-Objective Evolutionary Algorithm based on Decomposition.

Implements the algorithm from:
    Zhang & Li (2007). "MOEA/D: A Multiobjective Evolutionary Algorithm
    Based on Decomposition." IEEE Trans. on Evolutionary Computation.

Decomposes a multi-objective problem into N scalar subproblems via
Tchebycheff scalarization with uniformly distributed weight vectors.
Each subproblem is optimized by one individual. Neighboring subproblems
share solutions: when an offspring improves a neighbor's subproblem,
it replaces that neighbor's solution immediately.
"""

from __future__ import annotations

import math

import numpy as np

from ._individual import Individual
from .base_population_optimizer import BasePopulationOptimizer


def generate_weight_vectors(n_objectives: int, n_partitions: int) -> np.ndarray:
    """Uniformly distributed weight vectors via Das-Dennis simplex-lattice.

    Enumerates all non-negative integer tuples (h_1, ..., h_m) summing
    to ``n_partitions``, then normalizes by ``n_partitions``.

    Returns
    -------
    np.ndarray, shape (n_vectors, n_objectives)
        n_vectors = C(n_partitions + m - 1, m - 1).
    """
    if n_objectives == 1:
        return np.array([[1.0]])

    if n_objectives == 2:
        weights = np.empty((n_partitions + 1, 2))
        for i in range(n_partitions + 1):
            weights[i] = [i / n_partitions, 1.0 - i / n_partitions]
        return weights

    points: list[list[int]] = []
    _simplex_lattice(n_objectives, n_partitions, [], points)
    return np.array(points, dtype=float) / n_partitions


def _simplex_lattice(m: int, H: int, current: list[int], result: list[list[int]]):
    """Recursively enumerate m-tuples of non-negative ints summing to H."""
    if len(current) == m - 1:
        result.append(current + [H - sum(current)])
        return
    for i in range(H - sum(current) + 1):
        _simplex_lattice(m, H, current + [i], result)


def find_n_partitions(n_objectives: int, target_population: int) -> int:
    """Find H such that C(H + m - 1, m - 1) is closest to target."""
    if n_objectives <= 2:
        return max(1, target_population - 1)

    best_h, best_diff = 1, float("inf")
    for h in range(1, 200):
        n_vec = math.comb(h + n_objectives - 1, n_objectives - 1)
        diff = abs(n_vec - target_population)
        if diff < best_diff:
            best_diff = diff
            best_h = h
        if n_vec > target_population * 2:
            break
    return best_h


def compute_neighborhoods(
    weight_vectors: np.ndarray, n_neighbors: int
) -> list[list[int]]:
    """T-nearest neighbors for each weight vector (Euclidean distance)."""
    diff = weight_vectors[:, np.newaxis, :] - weight_vectors[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))

    neighborhoods = []
    for i in range(len(weight_vectors)):
        neighborhoods.append(np.argsort(distances[i])[:n_neighbors].tolist())
    return neighborhoods


def tchebycheff_fitness(
    objectives: list[float],
    weight: np.ndarray,
    reference_point: list[float],
    eps: float = 1e-6,
) -> float:
    """Tchebycheff scalarization adapted for maximization.

    fitness = -max_j { max(w_j, eps) * (z*_j - f_j) }

    Higher fitness means closer to the ideal reference point.
    A solution AT the ideal gets fitness 0 (the maximum possible).
    """
    worst_gap = -math.inf
    for w, f, z in zip(weight, objectives, reference_point):
        gap = max(w, eps) * (z - f)
        if gap > worst_gap:
            worst_gap = gap
    return -worst_gap


class _MOEADOptimizer(BasePopulationOptimizer):
    """MOEA/D multi-objective optimizer.

    Each individual is assigned a weight vector defining its scalar
    subproblem. During iteration, parents are selected from the
    weight-vector neighborhood, offspring are produced via crossover
    and mutation, and neighbors are replaced if the offspring
    improves their subproblem fitness.

    This immediate-replacement mechanism lets good solutions
    propagate through the population within a single generation,
    which distinguishes MOEA/D from generational algorithms like
    NSGA-II.
    """

    name = "MOEA/D"
    _name_ = "moead"
    __name__ = "MOEADOptimizer"

    def __init__(
        self,
        search_space,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        population=20,
        n_neighbors=None,
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
        self._n_neighbors_param = n_neighbors

        self.individuals: list = []
        self._weight_vectors: np.ndarray | None = None
        self._neighborhoods: list[list[int]] | None = None
        self._reference_point: list[float] | None = None
        self._pop_objectives: list[list[float]] = []
        self._current_subproblem: int = 0
        self._current_subproblem_idx: int = 0

        self._moead_new_pos = None
        self._batch_individual_refs: list = []
        self._batch_subproblem_indices: list[int] = []

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
        self.p_current._moead_objectives = (
            list(objectives) if objectives is not None else [score_new]
        )

    def _on_finish_initialization(self):
        n_obj = getattr(self, "_n_objectives", 2)
        pop_size = len(self.individuals)

        # Collect per-individual objectives
        self._pop_objectives = []
        for ind in self.individuals:
            obj = getattr(ind, "_moead_objectives", None)
            if obj is not None and len(obj) == n_obj:
                self._pop_objectives.append(list(obj))
            else:
                score = (
                    ind._score_current if ind._score_current is not None else -math.inf
                )
                self._pop_objectives.append([score] * n_obj)

        # Weight vectors via simplex-lattice, adjusted to match population size
        n_partitions = find_n_partitions(n_obj, pop_size)
        self._weight_vectors = generate_weight_vectors(n_obj, n_partitions)

        n_weights = len(self._weight_vectors)
        if n_weights > pop_size:
            self._weight_vectors = self._weight_vectors[:pop_size]
        elif n_weights < pop_size:
            rng = np.random.default_rng()
            extra = rng.dirichlet(np.ones(n_obj), size=pop_size - n_weights)
            self._weight_vectors = np.vstack([self._weight_vectors, extra])

        # Neighborhoods
        n_neighbors = self._n_neighbors_param
        if n_neighbors is None:
            n_neighbors = max(3, pop_size // 5)
        n_neighbors = min(n_neighbors, pop_size)
        self._neighborhoods = compute_neighborhoods(self._weight_vectors, n_neighbors)

        # Reference point: best observed value per objective
        obj_array = np.array(self._pop_objectives)
        self._reference_point = obj_array.max(axis=0).tolist()

        self._current_subproblem = 0

    def _setup_iteration(self):
        """Select parents from neighborhood, crossover, mutate."""
        pop_size = len(self.individuals)
        i = self._current_subproblem % pop_size
        self._current_subproblem_idx = i
        self._current_subproblem += 1

        rng = np.random.default_rng()
        neighborhood = self._neighborhoods[i]
        idx_a, idx_b = rng.choice(neighborhood, size=2, replace=len(neighborhood) < 2)

        parent_a = self.individuals[idx_a]
        parent_b = self.individuals[idx_b]
        pos_a = parent_a._pos_current
        pos_b = parent_b._pos_current

        if pos_a is None or pos_b is None:
            self.p_current = self.individuals[i]
            fallback = self.p_current._pos_current
            self._moead_new_pos = (
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

        # Mutation via Individual's hill-climbing mechanism.
        # Use backing fields to avoid polluting tracking lists.
        mutator = self.individuals[i]
        self.p_current = mutator
        self._moead_new_pos = mutator.move_climb_typed(new_pos)

    def _iterate_continuous_batch(self):
        if self._moead_new_pos is None:
            self._setup_iteration()
        return self._moead_new_pos[self._continuous_mask]

    def _iterate_categorical_batch(self):
        if self._moead_new_pos is None:
            self._setup_iteration()
        return self._moead_new_pos[self._categorical_mask]

    def _iterate_discrete_batch(self):
        if self._moead_new_pos is None:
            self._setup_iteration()
        return self._moead_new_pos[self._discrete_mask]

    def _on_evaluate(self, score_new):
        """Update reference point and replace neighbors if improved."""
        n_obj = getattr(self, "_n_objectives", 2)
        objectives = getattr(self, "_last_objectives", None)
        if objectives is None or len(objectives) != n_obj:
            objectives = [score_new] * n_obj

        offspring_obj = list(objectives)
        offspring_pos = self._pos_new.copy()

        # Update reference point (ideal per objective)
        for j in range(n_obj):
            if offspring_obj[j] > self._reference_point[j]:
                self._reference_point[j] = offspring_obj[j]

        # Replace neighbors whose subproblem fitness improves
        i = self._current_subproblem_idx
        for j_idx in self._neighborhoods[i]:
            if j_idx >= len(self._pop_objectives):
                continue

            fit_offspring = tchebycheff_fitness(
                offspring_obj,
                self._weight_vectors[j_idx],
                self._reference_point,
            )
            fit_current = tchebycheff_fitness(
                self._pop_objectives[j_idx],
                self._weight_vectors[j_idx],
                self._reference_point,
            )

            if fit_offspring > fit_current:
                ind = self.individuals[j_idx]
                ind.__dict__["_CoreOptimizer__pos_current"] = offspring_pos.copy()
                ind.__dict__["_CoreOptimizer__score_current"] = score_new
                self._pop_objectives[j_idx] = offspring_obj[:]

        self._update_best(self._pos_new, score_new)
        self._update_current(self._pos_new, score_new)

        # Reset so next _iterate triggers a fresh _setup_iteration
        self._moead_new_pos = None

    def _iterate_batch(self, n):
        positions = []
        self._batch_individual_refs = []
        self._batch_subproblem_indices = []
        for _ in range(n):
            self._setup_iteration()
            pos = self._generate_position()
            positions.append(pos)
            self._batch_individual_refs.append(self.p_current)
            self._batch_subproblem_indices.append(self._current_subproblem_idx)
            self._moead_new_pos = None
        return positions

    def _evaluate_batch(self, positions, scores):
        for pos, score, indiv_ref, sub_idx in zip(
            positions,
            scores,
            self._batch_individual_refs,
            self._batch_subproblem_indices,
        ):
            self.p_current = indiv_ref
            self._current_subproblem_idx = sub_idx
            self._pos_new = pos
            self._evaluate(score)
