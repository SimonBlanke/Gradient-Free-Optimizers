"""Tests for ExtraTreeRegressor and ExtraTreesRegressor."""

import pytest

from gradient_free_optimizers._estimators import ExtraTreesRegressor
from gradient_free_optimizers._estimators._decision_tree_regressor import (
    DecisionTreeRegressor,
)
from gradient_free_optimizers._estimators._extra_trees_regressor import (
    ExtraTreeRegressor,
)


class TestExtraTreeRegressor:
    def test_inherits_from_decision_tree(self):
        assert issubclass(ExtraTreeRegressor, DecisionTreeRegressor)

    def test_fit_predict_basic(self, regression_data):
        X, y = regression_data
        et = ExtraTreeRegressor(random_state=42)
        et.fit(X, y)
        preds = et.predict(X)
        assert len(preds) == len(y)

    def test_single_sample_returns_no_split(self):
        """_find_best_split returns (None, None) when n_samples < 2."""
        et = ExtraTreeRegressor(random_state=42)
        et.fit([[0.0], [1.0]], [0.0, 1.0])
        from gradient_free_optimizers._array_backend import asarray

        X_one = asarray([[5.0]], dtype=float)
        y_one = asarray([3.0], dtype=float)
        feat, thresh = et._find_best_split(X_one, y_one)
        assert feat is None
        assert thresh is None


class TestExtraTreesEnsemble:
    def test_fit_returns_self(self, regression_data):
        X, y = regression_data
        etr = ExtraTreesRegressor(n_estimators=5, random_state=42)
        assert etr.fit(X, y) is etr

    def test_n_estimators_count(self, regression_data):
        X, y = regression_data
        etr = ExtraTreesRegressor(n_estimators=7, random_state=42)
        etr.fit(X, y)
        assert len(etr.estimators_) == 7

    def test_trees_are_extra_tree_type(self, regression_data):
        X, y = regression_data
        etr = ExtraTreesRegressor(n_estimators=3, random_state=42)
        etr.fit(X, y)
        for tree in etr.estimators_:
            assert isinstance(tree, ExtraTreeRegressor)

    def test_bootstrap_false_is_default(self):
        etr = ExtraTreesRegressor()
        assert etr.bootstrap is False

    def test_bootstrap_true(self, regression_data):
        X, y = regression_data
        etr = ExtraTreesRegressor(n_estimators=5, bootstrap=True, random_state=42)
        etr.fit(X, y)
        preds = etr.predict(X)
        assert len(preds) == len(y)

    def test_prediction_is_tree_average(self, regression_data):
        X, y = regression_data
        n_trees = 5
        etr = ExtraTreesRegressor(n_estimators=n_trees, random_state=42)
        etr.fit(X, y)

        individual_preds = [tree.predict(X) for tree in etr.estimators_]
        ensemble_pred = etr.predict(X)

        for i in range(len(X)):
            expected = sum(float(p[i]) for p in individual_preds) / n_trees
            assert float(ensemble_pred[i]) == pytest.approx(expected, abs=1e-10)

    def test_reproducibility(self, regression_data):
        X, y = regression_data
        etr1 = ExtraTreesRegressor(n_estimators=5, random_state=42)
        etr1.fit(X, y)
        etr2 = ExtraTreesRegressor(n_estimators=5, random_state=42)
        etr2.fit(X, y)
        preds1 = etr1.predict(X)
        preds2 = etr2.predict(X)
        for p1, p2 in zip(preds1, preds2):
            assert float(p1) == pytest.approx(float(p2), abs=1e-10)


class TestMaxFeatures:
    def test_sqrt(self):
        etr = ExtraTreesRegressor(max_features="sqrt")
        assert etr._get_max_features(16) == 4
        assert etr._get_max_features(1) == 1

    def test_log2(self):
        etr = ExtraTreesRegressor(max_features="log2")
        assert etr._get_max_features(16) == 4
        assert etr._get_max_features(1) == 1

    def test_none(self):
        etr = ExtraTreesRegressor(max_features=None)
        assert etr._get_max_features(10) == 10

    def test_float(self):
        etr = ExtraTreesRegressor(max_features=0.5)
        assert etr._get_max_features(10) == 5
        assert etr._get_max_features(3) == 1

    def test_int(self):
        etr = ExtraTreesRegressor(max_features=3)
        assert etr._get_max_features(10) == 3
        assert etr._get_max_features(2) == 2
