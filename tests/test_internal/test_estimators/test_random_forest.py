"""Tests for the native RandomForestRegressor."""

import pytest

from gradient_free_optimizers._estimators import RandomForestRegressor
from gradient_free_optimizers._estimators._decision_tree_regressor import (
    DecisionTreeRegressor,
)


class TestFitPredict:
    def test_fit_returns_self(self, regression_data):
        X, y = regression_data
        rf = RandomForestRegressor(n_estimators=5, random_state=42)
        assert rf.fit(X, y) is rf

    def test_predict_length(self, regression_data):
        X, y = regression_data
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(X, y)
        preds = rf.predict(X)
        assert len(preds) == len(y)

    def test_predict_on_unseen_data(self, regression_data):
        X, y = regression_data
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(X, y)
        preds = rf.predict([[0.5, 0.5], [1.5, 1.5]])
        assert len(preds) == 2

    def test_n_estimators_count(self, regression_data):
        X, y = regression_data
        rf = RandomForestRegressor(n_estimators=7, random_state=42)
        rf.fit(X, y)
        assert len(rf.estimators_) == 7

    def test_trees_are_decision_tree_type(self, regression_data):
        X, y = regression_data
        rf = RandomForestRegressor(n_estimators=3, random_state=42)
        rf.fit(X, y)
        for tree in rf.estimators_:
            assert isinstance(tree, DecisionTreeRegressor)


class TestBootstrap:
    def test_bootstrap_true_is_default(self):
        rf = RandomForestRegressor()
        assert rf.bootstrap is True

    def test_bootstrap_false(self, regression_data):
        X, y = regression_data
        rf = RandomForestRegressor(n_estimators=5, bootstrap=False, random_state=42)
        rf.fit(X, y)
        preds = rf.predict(X)
        assert len(preds) == len(y)


class TestMaxFeatures:
    def test_sqrt_is_default(self):
        rf = RandomForestRegressor()
        assert rf.max_features == "sqrt"

    def test_sqrt(self):
        rf = RandomForestRegressor(max_features="sqrt")
        assert rf._get_max_features(16) == 4
        assert rf._get_max_features(1) == 1

    def test_log2(self):
        rf = RandomForestRegressor(max_features="log2")
        assert rf._get_max_features(16) == 4
        assert rf._get_max_features(1) == 1

    def test_none(self):
        rf = RandomForestRegressor(max_features=None)
        assert rf._get_max_features(10) == 10

    def test_float(self):
        rf = RandomForestRegressor(max_features=0.5)
        assert rf._get_max_features(10) == 5
        assert rf._get_max_features(3) == 1

    def test_int_capped(self):
        rf = RandomForestRegressor(max_features=5)
        assert rf._get_max_features(10) == 5
        assert rf._get_max_features(3) == 3


class TestPredictionContract:
    def test_prediction_is_tree_average(self, regression_data):
        X, y = regression_data
        n_trees = 5
        rf = RandomForestRegressor(n_estimators=n_trees, random_state=42)
        rf.fit(X, y)

        individual_preds = [tree.predict(X) for tree in rf.estimators_]
        ensemble_pred = rf.predict(X)

        for i in range(len(X)):
            expected = sum(float(p[i]) for p in individual_preds) / n_trees
            assert float(ensemble_pred[i]) == pytest.approx(expected, abs=1e-10)

    def test_reproducibility(self, regression_data):
        X, y = regression_data
        rf1 = RandomForestRegressor(n_estimators=5, random_state=42)
        rf1.fit(X, y)
        rf2 = RandomForestRegressor(n_estimators=5, random_state=42)
        rf2.fit(X, y)
        preds1 = rf1.predict(X)
        preds2 = rf2.predict(X)
        for p1, p2 in zip(preds1, preds2):
            assert float(p1) == pytest.approx(float(p2), abs=1e-10)
