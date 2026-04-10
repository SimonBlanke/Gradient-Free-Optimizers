"""Tests for various objective function formats and return types."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from gradient_free_optimizers import ObjectiveResult, RandomSearchOptimizer
from gradient_free_optimizers._result import unpack_objective_result


def test_function():
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {
        "x1": np.arange(-100, 101, 1),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=30)


def test_sklearn():
    data = load_iris()
    X, y = data.data, data.target

    def model(para):
        knr = KNeighborsClassifier(n_neighbors=para["n_neighbors"])
        scores = cross_val_score(knr, X, y, cv=5)
        score = scores.mean()

        return score

    search_space = {
        "n_neighbors": np.arange(1, 51, 1),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(model, n_iter=30)


def test_obj_func_return_dictionary_0():
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score, {"_x1_": para["x1"]}

    search_space = {
        "x1": np.arange(-100, 101, 1),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=30)

    assert "_x1_" in list(opt.search_data.columns)


def test_obj_func_return_dictionary_1():
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score, {"_x1_": para["x1"], "_x1_*2": para["x1"] * 2}

    search_space = {
        "x1": np.arange(-100, 101, 1),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=30)

    assert "_x1_" in list(opt.search_data.columns)
    assert "_x1_*2" in list(opt.search_data.columns)


class TestUnpackObjectiveResult:
    """Unit tests for the centralized unpacking function."""

    def test_plain_float(self):
        score, metrics = unpack_objective_result(0.5)
        assert score == 0.5
        assert metrics == {}

    def test_plain_int(self):
        score, metrics = unpack_objective_result(3)
        assert score == 3.0
        assert metrics == {}

    def test_tuple_score_and_metrics(self):
        score, metrics = unpack_objective_result((0.9, {"loss": 0.1}))
        assert score == 0.9
        assert metrics == {"loss": 0.1}

    def test_objective_result_with_metrics(self):
        raw = ObjectiveResult(score=0.7, metrics={"acc": 0.7})
        score, metrics = unpack_objective_result(raw)
        assert score == 0.7
        assert metrics == {"acc": 0.7}

    def test_objective_result_without_metrics(self):
        raw = ObjectiveResult(score=-1.5)
        score, metrics = unpack_objective_result(raw)
        assert score == -1.5
        assert metrics == {}

    def test_objective_result_takes_precedence_over_tuple(self):
        """ObjectiveResult is checked before tuple, so no ambiguity."""
        raw = ObjectiveResult(score=0.42, metrics={"k": "v"})
        score, metrics = unpack_objective_result(raw)
        assert score == 0.42
        assert metrics["k"] == "v"

    def test_numpy_float(self):
        score, metrics = unpack_objective_result(np.float64(0.123))
        assert pytest.approx(score) == 0.123
        assert metrics == {}


class TestObjectiveResultIntegration:
    """Integration tests: ObjectiveResult through the full search pipeline."""

    search_space = {"x1": np.arange(-100, 101, 1)}

    def test_objective_result_score_only(self):
        def objective(para):
            return ObjectiveResult(score=-(para["x1"] ** 2))

        opt = RandomSearchOptimizer(self.search_space)
        opt.search(objective, n_iter=30, verbosity=False)
        assert opt.best_score is not None

    def test_objective_result_with_metrics_in_search_data(self):
        def objective(para):
            score = -(para["x1"] ** 2)
            return ObjectiveResult(score=score, metrics={"squared": para["x1"] ** 2})

        opt = RandomSearchOptimizer(self.search_space)
        opt.search(objective, n_iter=30, verbosity=False)
        assert "squared" in opt.search_data.columns

    def test_objective_result_with_minimization(self):
        def objective(para):
            loss = para["x1"] ** 2
            return ObjectiveResult(score=loss, metrics={"raw_loss": loss})

        opt = RandomSearchOptimizer(self.search_space)
        opt.search(objective, n_iter=50, verbosity=False, optimum="minimum")
        # best_score is negated internally for minimization,
        # so it should be <= 0 for a non-negative objective
        assert opt.best_score <= 0
        assert "raw_loss" in opt.search_data.columns

    def test_objective_result_with_callbacks(self):
        received_metrics = []

        def callback(info):
            received_metrics.append(info.metrics)

        def objective(para):
            return ObjectiveResult(
                score=-(para["x1"] ** 2), metrics={"doubled": para["x1"] * 2}
            )

        opt = RandomSearchOptimizer(self.search_space)
        opt.search(objective, n_iter=20, verbosity=False, callbacks=[callback])
        assert len(received_metrics) == 20
        assert all("doubled" in m for m in received_metrics)
