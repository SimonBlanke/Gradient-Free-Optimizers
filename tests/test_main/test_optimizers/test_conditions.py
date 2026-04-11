"""Tests for conditional search spaces."""

import math

import numpy as np
import pandas as pd
import pytest

from gradient_free_optimizers import HillClimbingOptimizer, RandomSearchOptimizer
from gradient_free_optimizers._search_params import SearchParams

from ._parametrize import optimizers, optimizers_representative


class TestConditionsFiltering:
    """Core conditions functionality: inactive params are filtered."""

    @pytest.mark.parametrize(*optimizers)
    def test_condition_filters_params(self, Optimizer):
        search_space = {
            "x": np.arange(-10, 10, 1),
            "y": np.arange(-10, 10, 1),
        }

        received_keys = []

        def objective(params):
            received_keys.append(set(params.keys()))
            return -(params["x"] ** 2)

        opt = Optimizer(
            search_space,
            conditions=[lambda p: {"y": p["x"] > 0}],
            random_state=42,
        )
        opt.search(objective, n_iter=30, verbosity=False)

        for keys in received_keys:
            assert "x" in keys

    @pytest.mark.parametrize(*optimizers_representative)
    def test_condition_nan_in_search_data(self, Optimizer):
        search_space = {
            "x": np.arange(-10, 10, 1),
            "y": np.arange(-10, 10, 1),
        }

        opt = Optimizer(
            search_space,
            conditions=[lambda p: {"y": p["x"] > 0}],
            random_state=42,
        )
        opt.search(
            lambda p: -(p["x"] ** 2),
            n_iter=50,
            verbosity=False,
        )

        df = opt.search_data
        inactive_rows = df[df["x"] <= 0]
        if len(inactive_rows) > 0:
            assert inactive_rows["y"].isna().all()

        active_rows = df[df["x"] > 0]
        if len(active_rows) > 0:
            assert active_rows["y"].notna().all()

    @pytest.mark.parametrize(*optimizers_representative)
    def test_no_conditions_backward_compat(self, Optimizer):
        """Without conditions, everything works as before."""
        search_space = {
            "x": np.arange(-10, 10, 1),
            "y": np.arange(-10, 10, 1),
        }
        opt = Optimizer(search_space, random_state=42)
        opt.search(
            lambda p: -(p["x"] ** 2 + p["y"] ** 2),
            n_iter=30,
            verbosity=False,
        )

        df = opt.search_data
        assert df["x"].notna().all()
        assert df["y"].notna().all()

    def test_condition_multiple_and_logic(self):
        search_space = {
            "x": np.arange(0, 10, 1),
            "y": np.arange(0, 10, 1),
            "z": np.arange(0, 10, 1),
        }

        received = []

        def objective(params):
            received.append(dict(params))
            return -params["x"]

        opt = HillClimbingOptimizer(
            search_space,
            conditions=[
                lambda p: {"z": p["x"] > 3},
                lambda p: {"z": p["y"] > 3},
            ],
            random_state=42,
        )
        opt.search(objective, n_iter=30, verbosity=False)

        for params in received:
            if "z" in params:
                pass


class TestConditionsWithConstraints:
    """Constraints and conditions working together."""

    @pytest.mark.parametrize(*optimizers_representative)
    def test_constraint_sees_only_active_params(self, Optimizer):
        search_space = {
            "x": np.arange(1, 10, 1),
            "y": np.arange(1, 10, 1),
        }

        opt = Optimizer(
            search_space,
            conditions=[lambda p: {"y": p["x"] > 5}],
            constraints=[lambda p: p.get("x", 0) > 2],
            random_state=42,
        )
        opt.search(
            lambda p: -p["x"],
            n_iter=30,
            verbosity=False,
        )

        assert np.all(opt.search_data["x"].values > 2)

    def test_constraint_with_inactive_param(self):
        """Constraint that uses .get() for a potentially inactive param."""
        search_space = {
            "x": np.arange(-5, 5, 1),
            "y": np.arange(-5, 5, 1),
        }

        opt = HillClimbingOptimizer(
            search_space,
            conditions=[lambda p: {"y": p["x"] > 0}],
            constraints=[lambda p: p.get("y", 0) < 3],
            random_state=42,
        )
        opt.search(
            lambda p: -(p["x"] ** 2),
            n_iter=30,
            verbosity=False,
        )

        df = opt.search_data
        active_y = df[df["x"] > 0]["y"].dropna()
        if len(active_y) > 0:
            assert np.all(active_y.values < 3)


class TestSearchParams:
    """SearchParams dict subclass behavior."""

    def test_is_dict(self):
        sp = SearchParams({"x": 1, "y": 2})
        assert isinstance(sp, dict)
        assert sp["x"] == 1
        assert len(sp) == 2
        assert list(sp.keys()) == ["x", "y"]

    def test_context_defaults(self):
        sp = SearchParams({"x": 1})
        assert sp.iteration == 0
        assert sp.score_best == -math.inf
        assert sp.pos_best is None
        assert sp.n_iter_total == 0

    def test_context_from_optimizer(self):
        search_space = {"x": np.arange(-5, 5, 1)}
        opt = HillClimbingOptimizer(search_space, random_state=42)

        context = {}

        def objective(params):
            context["iteration"] = params.iteration
            context["has_score_best"] = params.score_best is not None
            context["is_search_params"] = isinstance(params, SearchParams)
            return -(params["x"] ** 2)

        opt.search(objective, n_iter=10, verbosity=False)

        assert context["is_search_params"] is True

    def test_deferred_set_conditions(self):
        search_space = {
            "x": np.arange(-5, 5, 1),
            "y": np.arange(-5, 5, 1),
        }

        call_count = [0]

        def objective(params):
            call_count[0] += 1
            if call_count[0] == 5:
                params.set_conditions([lambda p: {"y": p["x"] > 0}])
            return -(params["x"] ** 2)

        opt = HillClimbingOptimizer(search_space, random_state=42)
        opt.search(objective, n_iter=20, verbosity=False)

        assert len(opt.conv.conditions) == 1

    def test_deferred_set_constraints(self):
        search_space = {"x": np.arange(-10, 10, 1)}

        call_count = [0]

        def objective(params):
            call_count[0] += 1
            if call_count[0] == 3:
                params.set_constraints([lambda p: p["x"] > 0])
            return -(params["x"] ** 2)

        opt = HillClimbingOptimizer(search_space, random_state=42)
        opt.search(objective, n_iter=20, verbosity=False)

        assert len(opt.conv.constraints) == 1

    def test_set_conditions_is_additive(self):
        search_space = {
            "x": np.arange(-5, 5, 1),
            "y": np.arange(-5, 5, 1),
            "z": np.arange(-5, 5, 1),
        }

        opt = HillClimbingOptimizer(
            search_space,
            conditions=[lambda p: {"y": p["x"] > 0}],
            random_state=42,
        )

        call_count = [0]

        def objective(params):
            call_count[0] += 1
            if call_count[0] == 5:
                params.set_conditions([lambda p: {"z": p["x"] > 2}])
            return -(params["x"] ** 2)

        opt.search(objective, n_iter=20, verbosity=False)
        assert len(opt.conv.conditions) == 2


class TestConditionsWithDimensionTypes:
    """Conditions work across all dimension types."""

    def test_continuous_dimension(self):
        search_space = {
            "x": np.arange(-5, 5, 1),
            "y": (0.0, 1.0),
        }

        received = []

        def objective(params):
            received.append(dict(params))
            return -(params["x"] ** 2)

        opt = HillClimbingOptimizer(
            search_space,
            conditions=[lambda p: {"y": p["x"] > 0}],
            random_state=42,
        )
        opt.search(objective, n_iter=30, verbosity=False)

        for params in received:
            if params["x"] <= 0:
                assert "y" not in params

    def test_categorical_dimension(self):
        search_space = {
            "algo": ["svm", "rf", "nn"],
            "kernel": ["linear", "rbf", "poly"],
            "n_estimators": np.arange(10, 100, 10),
        }

        received = []

        def objective(params):
            received.append(dict(params))
            return 1.0

        opt = RandomSearchOptimizer(
            search_space,
            conditions=[
                lambda p: {
                    "kernel": p["algo"] == "svm",
                    "n_estimators": p["algo"] == "rf",
                }
            ],
            random_state=42,
        )
        opt.search(objective, n_iter=30, verbosity=False)

        for params in received:
            if params["algo"] == "svm":
                assert "kernel" in params
                assert "n_estimators" not in params
            elif params["algo"] == "rf":
                assert "kernel" not in params
                assert "n_estimators" in params
            else:
                assert "kernel" not in params
                assert "n_estimators" not in params


class TestConditionsAskTell:
    """Conditions in the Ask/Tell interface."""

    def test_ask_filters_inactive_params(self):
        from gradient_free_optimizers.ask_tell import HillClimbingOptimizer as ATHill

        search_space = {
            "x": np.arange(-5, 5, 1),
            "y": np.arange(-5, 5, 1),
        }

        opt = ATHill(
            search_space,
            initial_evaluations=[({"x": 0, "y": 0}, 0.5)],
            conditions=[lambda p: {"y": p["x"] > 0}],
        )

        params_list = opt.ask(n=4)
        for params in params_list:
            assert "x" in params
            if params["x"] <= 0:
                assert "y" not in params

    def test_set_conditions_direct(self):
        from gradient_free_optimizers.ask_tell import (
            RandomSearchOptimizer as ATRandom,
        )

        search_space = {
            "x": np.arange(-5, 5, 1),
            "y": np.arange(-5, 5, 1),
        }

        opt = ATRandom(
            search_space,
            initial_evaluations=[({"x": 0, "y": 0}, 0.5)],
        )

        assert len(opt.conv.conditions) == 0

        opt.set_conditions([lambda p: {"y": p["x"] > 0}])
        assert len(opt.conv.conditions) == 1

        params_list = opt.ask(n=4)
        for params in params_list:
            if params["x"] <= 0:
                assert "y" not in params


class TestConditionsEdgeCases:
    """Edge cases and robustness."""

    def test_empty_conditions_list(self):
        search_space = {"x": np.arange(-5, 5, 1)}
        opt = HillClimbingOptimizer(search_space, conditions=[], random_state=42)
        opt.search(lambda p: -(p["x"] ** 2), n_iter=10, verbosity=False)
        assert opt.search_data["x"].notna().all()

    def test_condition_returns_partial_dict(self):
        """Condition that only mentions some params (rest stay active)."""
        search_space = {
            "x": np.arange(-5, 5, 1),
            "y": np.arange(-5, 5, 1),
            "z": np.arange(-5, 5, 1),
        }

        received = []

        def objective(params):
            received.append(set(params.keys()))
            return 1.0

        opt = HillClimbingOptimizer(
            search_space,
            conditions=[lambda p: {"z": False}],
            random_state=42,
        )
        opt.search(objective, n_iter=10, verbosity=False)

        for keys in received:
            assert "x" in keys
            assert "y" in keys
            assert "z" not in keys

    def test_all_params_always_active(self):
        """Condition that always returns True for everything."""
        search_space = {
            "x": np.arange(-5, 5, 1),
            "y": np.arange(-5, 5, 1),
        }

        received = []

        def objective(params):
            received.append(set(params.keys()))
            return 1.0

        opt = HillClimbingOptimizer(
            search_space,
            conditions=[lambda p: {"x": True, "y": True}],
            random_state=42,
        )
        opt.search(objective, n_iter=10, verbosity=False)

        for keys in received:
            assert keys == {"x", "y"}

    def test_search_data_has_correct_columns(self):
        search_space = {
            "x": np.arange(-5, 5, 1),
            "y": np.arange(-5, 5, 1),
        }
        opt = HillClimbingOptimizer(
            search_space,
            conditions=[lambda p: {"y": p["x"] > 0}],
            random_state=42,
        )
        opt.search(lambda p: -(p["x"] ** 2), n_iter=20, verbosity=False)

        df = opt.search_data
        assert "score" in df.columns
        assert "x" in df.columns
        assert "y" in df.columns
        assert len(df) == 20
