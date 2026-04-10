"""Tests for multi-objective optimization support (Phase 1 + Phase 2)."""

import math

import numpy as np
import pytest

from gradient_free_optimizers import (
    HillClimbingOptimizer,
    MOEADOptimizer,
    NSGA2Optimizer,
    ObjectiveResult,
    ParticleSwarmOptimizer,
    RandomSearchOptimizer,
    SMSEMOAOptimizer,
    WeightedSum,
)
from gradient_free_optimizers._fitness_mapper import ScalarIdentity
from gradient_free_optimizers._result import (
    negate_objectives,
    objectives_as_list,
    unpack_objective_result,
)
from gradient_free_optimizers.optimizers.pop_opt.moead import (
    generate_weight_vectors,
    tchebycheff_fitness,
)
from gradient_free_optimizers.optimizers.pop_opt.sms_emoa import (
    hypervolume_2d,
    hypervolume_contributions,
)

SEARCH_SPACE = {"x": np.arange(-5, 6, 1), "y": np.arange(-5, 6, 1)}


def bi_objective(params):
    """Two conflicting objectives: minimize x^2, minimize (x-3)^2."""
    x = params["x"]
    return [-(x**2), -((x - 3) ** 2)]


def bi_objective_with_metrics(params):
    x = params["x"]
    return ObjectiveResult(
        score=[-(x**2), -((x - 3) ** 2)],
        metrics={"x_val": x},
    )


class TestFitnessMapper:
    def test_scalar_identity(self):
        mapper = ScalarIdentity()
        assert mapper(3.14) == pytest.approx(3.14)

    def test_weighted_sum_equal_weights(self):
        mapper = WeightedSum(n_objectives=2)
        assert mapper([4.0, 6.0]) == pytest.approx(5.0)

    def test_weighted_sum_custom_weights(self):
        mapper = WeightedSum(weights=[0.8, 0.2])
        assert mapper([10.0, 5.0]) == pytest.approx(9.0)

    def test_weighted_sum_single_objective(self):
        mapper = WeightedSum(weights=[1.0])
        assert mapper([7.0]) == pytest.approx(7.0)


class TestUnpackMultiObjective:
    def test_list_return(self):
        objectives, metrics = unpack_objective_result([0.5, 0.3])
        assert objectives == [0.5, 0.3]
        assert metrics == {}

    def test_ndarray_return(self):
        objectives, metrics = unpack_objective_result(np.array([1.0, 2.0]))
        assert objectives == [1.0, 2.0]
        assert metrics == {}

    def test_objective_result_with_list_score(self):
        raw = ObjectiveResult(score=[0.9, 0.1], metrics={"k": 1})
        objectives, metrics = unpack_objective_result(raw)
        assert objectives == [0.9, 0.1]
        assert metrics == {"k": 1}

    def test_tuple_with_list_score(self):
        objectives, metrics = unpack_objective_result(([1.0, 2.0], {"m": 3}))
        assert objectives == [1.0, 2.0]
        assert metrics == {"m": 3}


class TestNegateObjectives:
    def test_negate_scalar(self):
        assert negate_objectives(5.0) == -5.0

    def test_negate_list(self):
        assert negate_objectives([1.0, -2.0, 3.0]) == [-1.0, 2.0, -3.0]


class TestObjectivesAsList:
    def test_single_objective_returns_none(self):
        assert objectives_as_list(0.5, n_objectives=1) is None

    def test_multi_objective_list(self):
        assert objectives_as_list([0.5, 0.3], n_objectives=2) == [0.5, 0.3]

    def test_multi_objective_scalar(self):
        assert objectives_as_list(0.5, n_objectives=2) == [0.5]


class TestMultiObjectiveSearch:
    """Integration tests for multi-objective through the search pipeline."""

    def test_basic_mo_search(self):
        opt = RandomSearchOptimizer(SEARCH_SPACE)
        opt.search(bi_objective, n_iter=30, verbosity=False, n_objectives=2)

        assert opt.best_score is not None
        assert "objective_0" in opt.search_data.columns
        assert "objective_1" in opt.search_data.columns

    def test_mo_with_objective_result(self):
        opt = RandomSearchOptimizer(SEARCH_SPACE)
        opt.search(
            bi_objective_with_metrics,
            n_iter=30,
            verbosity=False,
            n_objectives=2,
        )

        cols = opt.search_data.columns
        assert "objective_0" in cols
        assert "objective_1" in cols
        assert "x_val" in cols

    def test_mo_with_custom_weights(self):
        mapper = WeightedSum(weights=[0.9, 0.1])
        opt = HillClimbingOptimizer(SEARCH_SPACE, random_state=42)
        opt.search(
            bi_objective,
            n_iter=50,
            verbosity=False,
            n_objectives=2,
            fitness_mapper=mapper,
        )
        assert opt.best_score is not None

    def test_mo_with_minimization(self):
        def objective(params):
            x = params["x"]
            return [x**2, (x - 3) ** 2]

        opt = RandomSearchOptimizer(SEARCH_SPACE, random_state=0)
        opt.search(
            objective,
            n_iter=50,
            verbosity=False,
            n_objectives=2,
            optimum="minimum",
        )
        assert "objective_0" in opt.search_data.columns
        assert "objective_1" in opt.search_data.columns

    def test_mo_callbacks_receive_objectives(self):
        received = []

        def cb(info):
            received.append(info.objectives)

        opt = RandomSearchOptimizer(SEARCH_SPACE)
        opt.search(
            bi_objective,
            n_iter=20,
            verbosity=False,
            n_objectives=2,
            callbacks=[cb],
        )
        assert len(received) == 20
        assert all(isinstance(o, list) and len(o) == 2 for o in received)

    def test_single_objective_no_objective_columns(self):
        """Single-objective search should NOT add objective_* columns."""

        def objective(params):
            return -(params["x"] ** 2)

        opt = RandomSearchOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=20, verbosity=False)

        obj_cols = [c for c in opt.search_data.columns if c.startswith("objective_")]
        assert len(obj_cols) == 0

    def test_single_objective_callbacks_objectives_none(self):
        received = []

        def cb(info):
            received.append(info.objectives)

        def objective(params):
            return -(params["x"] ** 2)

        opt = RandomSearchOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=10, verbosity=False, callbacks=[cb])
        assert all(o is None for o in received)


class TestParetoFront:
    def test_pareto_front_basic(self):
        opt = RandomSearchOptimizer(SEARCH_SPACE, random_state=42)
        opt.search(bi_objective, n_iter=50, verbosity=False, n_objectives=2)

        pf = opt.pareto_front
        assert len(pf) > 0
        assert "objective_0" in pf.columns
        assert "objective_1" in pf.columns

        # Every solution in the Pareto front should be non-dominated
        obj_vals = pf[["objective_0", "objective_1"]].values
        for i in range(len(obj_vals)):
            for j in range(len(obj_vals)):
                if i == j:
                    continue
                # j should NOT dominate i
                assert not (
                    all(obj_vals[j] >= obj_vals[i]) and any(obj_vals[j] > obj_vals[i])
                ), f"Solution {j} dominates {i} in Pareto front"

    def test_pareto_front_raises_for_single_objective(self):
        def objective(params):
            return -(params["x"] ** 2)

        opt = RandomSearchOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=20, verbosity=False)

        with pytest.raises(ValueError, match="No objective columns"):
            _ = opt.pareto_front

    def test_pareto_front_with_population_optimizer(self):
        opt = ParticleSwarmOptimizer(
            SEARCH_SPACE,
            population=10,
            random_state=0,
        )
        opt.search(bi_objective, n_iter=50, verbosity=False, n_objectives=2)

        pf = opt.pareto_front
        assert len(pf) > 0


class TestNSGA2:
    """Tests for the NSGA-II multi-objective optimizer."""

    search_space = {"x": np.arange(-5, 6, 1)}

    def test_nsga2_basic_run(self):
        def objective(params):
            x = params["x"]
            return [-(x**2), -((x - 3) ** 2)]

        opt = NSGA2Optimizer(self.search_space, population=10, random_state=0)
        opt.search(objective, n_iter=50, n_objectives=2, verbosity=False)
        assert opt.best_score is not None
        assert "objective_0" in opt.search_data.columns
        assert "objective_1" in opt.search_data.columns

    def test_nsga2_pareto_front_non_dominated(self):
        def objective(params):
            x = params["x"]
            return [-(x**2), -((x - 3) ** 2)]

        opt = NSGA2Optimizer(self.search_space, population=10, random_state=42)
        opt.search(objective, n_iter=100, n_objectives=2, verbosity=False)

        pf = opt.pareto_front
        assert len(pf) >= 1

        # Verify non-domination
        obj_vals = pf[["objective_0", "objective_1"]].values
        for i in range(len(obj_vals)):
            for j in range(len(obj_vals)):
                if i == j:
                    continue
                assert not (
                    all(obj_vals[j] >= obj_vals[i]) and any(obj_vals[j] > obj_vals[i])
                )

    def test_nsga2_with_objective_result(self):
        def objective(params):
            x = params["x"]
            return ObjectiveResult(
                score=[-(x**2), -((x - 3) ** 2)],
                metrics={"raw_x": x},
            )

        opt = NSGA2Optimizer(self.search_space, population=10, random_state=0)
        opt.search(objective, n_iter=50, n_objectives=2, verbosity=False)
        assert "raw_x" in opt.search_data.columns

    def test_nsga2_2d_search_space(self):
        space = {"x": np.arange(-3, 4, 1), "y": np.arange(-3, 4, 1)}

        def objective(params):
            x, y = params["x"], params["y"]
            return [-(x**2 + y**2), -((x - 2) ** 2 + (y - 2) ** 2)]

        opt = NSGA2Optimizer(space, population=15, random_state=0)
        opt.search(objective, n_iter=150, n_objectives=2, verbosity=False)
        assert len(opt.pareto_front) >= 1

    def test_nsga2_three_objectives(self):
        def objective(params):
            x = params["x"]
            return [-(x**2), -((x - 2) ** 2), -((x + 2) ** 2)]

        opt = NSGA2Optimizer(self.search_space, population=10, random_state=0)
        opt.search(objective, n_iter=80, n_objectives=3, verbosity=False)
        assert "objective_0" in opt.search_data.columns
        assert "objective_2" in opt.search_data.columns


class TestWeightVectors:
    def test_2d_weights_sum_to_one(self):
        w = generate_weight_vectors(2, 10)
        assert len(w) == 11
        for row in w:
            assert pytest.approx(sum(row)) == 1.0

    def test_3d_weight_count(self):
        # C(5+2, 2) = 21
        w = generate_weight_vectors(3, 5)
        assert len(w) == 21
        for row in w:
            assert pytest.approx(sum(row)) == 1.0
            assert all(v >= 0 for v in row)


class TestTchebycheffFitness:
    def test_at_reference_point(self):
        """Solution at the ideal has fitness 0 (best possible)."""
        fit = tchebycheff_fitness([10, 20], [0.5, 0.5], [10, 20])
        assert fit == pytest.approx(0.0)

    def test_worse_solution_has_lower_fitness(self):
        ref = [10, 20]
        w = [0.5, 0.5]
        fit_close = tchebycheff_fitness([9, 19], w, ref)
        fit_far = tchebycheff_fitness([5, 10], w, ref)
        assert fit_close > fit_far


class TestMOEAD:
    """Tests for the MOEA/D optimizer."""

    search_space = {"x": np.arange(-5, 6, 1)}

    def test_moead_basic_run(self):
        def objective(params):
            x = params["x"]
            return [-(x**2), -((x - 3) ** 2)]

        opt = MOEADOptimizer(self.search_space, population=11, random_state=0)
        opt.search(objective, n_iter=100, n_objectives=2, verbosity=False)
        assert opt.best_score is not None
        assert "objective_0" in opt.search_data.columns
        assert "objective_1" in opt.search_data.columns

    def test_moead_finds_pareto_front(self):
        def objective(params):
            x = params["x"]
            return [-(x**2), -((x - 3) ** 2)]

        opt = MOEADOptimizer(self.search_space, population=11, random_state=42)
        opt.search(objective, n_iter=200, n_objectives=2, verbosity=False)

        pf = opt.pareto_front
        unique = pf[["objective_0", "objective_1"]].drop_duplicates()
        assert len(unique) >= 2

        # Verify non-domination
        obj_vals = pf[["objective_0", "objective_1"]].values
        for i in range(len(obj_vals)):
            for j in range(len(obj_vals)):
                if i == j:
                    continue
                assert not (
                    all(obj_vals[j] >= obj_vals[i]) and any(obj_vals[j] > obj_vals[i])
                )

    def test_moead_with_objective_result(self):
        def objective(params):
            x = params["x"]
            return ObjectiveResult(
                score=[-(x**2), -((x - 3) ** 2)],
                metrics={"raw_x": x},
            )

        opt = MOEADOptimizer(self.search_space, population=11, random_state=0)
        opt.search(objective, n_iter=100, n_objectives=2, verbosity=False)
        assert "raw_x" in opt.search_data.columns

    def test_moead_2d_search_space(self):
        space = {"x": np.arange(-3, 4, 1), "y": np.arange(-3, 4, 1)}

        def objective(params):
            x, y = params["x"], params["y"]
            return [-(x**2 + y**2), -((x - 2) ** 2 + (y - 2) ** 2)]

        opt = MOEADOptimizer(space, population=15, random_state=0)
        opt.search(objective, n_iter=200, n_objectives=2, verbosity=False)
        assert len(opt.pareto_front) >= 1

    def test_moead_three_objectives(self):
        def objective(params):
            x = params["x"]
            return [-(x**2), -((x - 2) ** 2), -((x + 2) ** 2)]

        opt = MOEADOptimizer(self.search_space, population=15, random_state=0)
        opt.search(objective, n_iter=150, n_objectives=3, verbosity=False)
        assert "objective_2" in opt.search_data.columns

    def test_moead_custom_n_neighbors(self):
        def objective(params):
            x = params["x"]
            return [-(x**2), -((x - 3) ** 2)]

        opt = MOEADOptimizer(
            self.search_space,
            population=11,
            n_neighbors=5,
            random_state=0,
        )
        opt.search(objective, n_iter=100, n_objectives=2, verbosity=False)
        assert opt.best_score is not None


class TestHypervolume:
    def test_2d_single_point(self):
        hv = hypervolume_2d([[5, 3]], [0, 0])
        assert hv == pytest.approx(15.0)

    def test_2d_two_points(self):
        hv = hypervolume_2d([[1, 5], [5, 1]], [0, 0])
        assert hv == pytest.approx(9.0)

    def test_2d_three_points(self):
        hv = hypervolume_2d([[1, 5], [3, 3], [5, 1]], [0, 0])
        assert hv == pytest.approx(13.0)

    def test_contributions_2d(self):
        front = [[1, 5], [3, 3], [5, 1]]
        ref = [0, 0]
        contribs = hypervolume_contributions(front, ref)
        assert contribs[0] == pytest.approx(2.0)  # (1,5)
        assert contribs[1] == pytest.approx(4.0)  # (3,3)
        assert contribs[2] == pytest.approx(2.0)  # (5,1)

    def test_contributions_single_point(self):
        contribs = hypervolume_contributions([[3, 4]], [0, 0])
        assert contribs[0] == math.inf


class TestSMSEMOA:
    """Tests for the SMS-EMOA optimizer."""

    search_space = {"x": np.arange(-5, 6, 1)}

    def test_sms_emoa_basic_run(self):
        def objective(params):
            x = params["x"]
            return [-(x**2), -((x - 3) ** 2)]

        opt = SMSEMOAOptimizer(self.search_space, population=11, random_state=0)
        opt.search(objective, n_iter=100, n_objectives=2, verbosity=False)
        assert opt.best_score is not None
        assert "objective_0" in opt.search_data.columns

    def test_sms_emoa_pareto_front_non_dominated(self):
        def objective(params):
            x = params["x"]
            return [-(x**2), -((x - 3) ** 2)]

        opt = SMSEMOAOptimizer(self.search_space, population=11, random_state=42)
        opt.search(objective, n_iter=200, n_objectives=2, verbosity=False)

        pf = opt.pareto_front
        unique = pf[["objective_0", "objective_1"]].drop_duplicates()
        assert len(unique) >= 2

        obj_vals = pf[["objective_0", "objective_1"]].values
        for i in range(len(obj_vals)):
            for j in range(len(obj_vals)):
                if i == j:
                    continue
                assert not (
                    all(obj_vals[j] >= obj_vals[i]) and any(obj_vals[j] > obj_vals[i])
                )

    def test_sms_emoa_with_objective_result(self):
        def objective(params):
            x = params["x"]
            return ObjectiveResult(
                score=[-(x**2), -((x - 3) ** 2)],
                metrics={"raw_x": x},
            )

        opt = SMSEMOAOptimizer(self.search_space, population=11, random_state=0)
        opt.search(objective, n_iter=100, n_objectives=2, verbosity=False)
        assert "raw_x" in opt.search_data.columns

    def test_sms_emoa_2d_search_space(self):
        space = {"x": np.arange(-3, 4, 1), "y": np.arange(-3, 4, 1)}

        def objective(params):
            x, y = params["x"], params["y"]
            return [-(x**2 + y**2), -((x - 2) ** 2 + (y - 2) ** 2)]

        opt = SMSEMOAOptimizer(space, population=15, random_state=0)
        opt.search(objective, n_iter=200, n_objectives=2, verbosity=False)
        assert len(opt.pareto_front) >= 1

    def test_sms_emoa_three_objectives(self):
        def objective(params):
            x = params["x"]
            return [-(x**2), -((x - 2) ** 2), -((x + 2) ** 2)]

        opt = SMSEMOAOptimizer(self.search_space, population=10, random_state=0)
        opt.search(objective, n_iter=100, n_objectives=3, verbosity=False)
        assert "objective_2" in opt.search_data.columns
