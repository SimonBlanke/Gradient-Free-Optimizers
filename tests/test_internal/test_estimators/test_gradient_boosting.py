"""Tests for the native GradientBoostingRegressor."""

import pytest

from gradient_free_optimizers._estimators import GradientBoostingRegressor


class TestFitPredict:
    def test_fit_returns_self(self, regression_data):
        X, y = regression_data
        gbr = GradientBoostingRegressor(n_estimators=5, random_state=42)
        assert gbr.fit(X, y) is gbr

    def test_predict_length(self, regression_data):
        X, y = regression_data
        gbr = GradientBoostingRegressor(n_estimators=5, random_state=42)
        gbr.fit(X, y)
        preds = gbr.predict(X)
        assert len(preds) == len(y)

    def test_init_prediction_is_target_mean(self, regression_data):
        X, y = regression_data
        gbr = GradientBoostingRegressor(n_estimators=5, random_state=42)
        gbr.fit(X, y)
        expected_mean = sum(y) / len(y)
        assert gbr._init_prediction == pytest.approx(expected_mean, abs=1e-10)

    def test_n_estimators_count(self, regression_data):
        X, y = regression_data
        gbr = GradientBoostingRegressor(n_estimators=7, random_state=42)
        gbr.fit(X, y)
        assert len(gbr.estimators_) == 7

    def test_estimators_stored_as_single_element_lists(self, regression_data):
        """Each stage stores the tree as [tree] for sklearn compatibility."""
        X, y = regression_data
        gbr = GradientBoostingRegressor(n_estimators=3, random_state=42)
        gbr.fit(X, y)
        for stage in gbr.estimators_:
            assert isinstance(stage, list)
            assert len(stage) == 1


class TestLearning:
    def test_more_stages_reduce_training_error(self, regression_data):
        X, y = regression_data
        errors = []
        for n_est in [1, 5, 20]:
            gbr = GradientBoostingRegressor(n_estimators=n_est, random_state=42)
            gbr.fit(X, y)
            preds = gbr.predict(X)
            mse = sum((float(p) - t) ** 2 for p, t in zip(preds, y)) / len(y)
            errors.append(mse)

        assert errors[1] <= errors[0] + 1e-10
        assert errors[2] <= errors[1] + 1e-10

    def test_prediction_matches_manual_computation(self, regression_data):
        X, y = regression_data
        gbr = GradientBoostingRegressor(n_estimators=5, random_state=42)
        gbr.fit(X, y)

        n = len(X)
        expected = [gbr._init_prediction] * n
        for tree_list in gbr.estimators_:
            tree = tree_list[0]
            tree_pred = tree.predict(X)
            for i in range(n):
                expected[i] += gbr.learning_rate * float(tree_pred[i])

        actual = gbr.predict(X)
        for a, e in zip(actual, expected):
            assert float(a) == pytest.approx(e, abs=1e-10)

    def test_subsample_less_than_one(self, regression_data):
        X, y = regression_data
        gbr = GradientBoostingRegressor(n_estimators=10, subsample=0.5, random_state=42)
        gbr.fit(X, y)
        preds = gbr.predict(X)
        assert len(preds) == len(y)


class TestReproducibility:
    def test_same_seed_same_predictions(self, regression_data):
        X, y = regression_data
        gbr1 = GradientBoostingRegressor(n_estimators=5, random_state=42)
        gbr1.fit(X, y)
        gbr2 = GradientBoostingRegressor(n_estimators=5, random_state=42)
        gbr2.fit(X, y)
        preds1 = gbr1.predict(X)
        preds2 = gbr2.predict(X)
        for p1, p2 in zip(preds1, preds2):
            assert float(p1) == pytest.approx(float(p2), abs=1e-10)
