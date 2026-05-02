"""Tests for the native DecisionTreeRegressor."""

import pytest

from gradient_free_optimizers._estimators import DecisionTreeRegressor


class TestFitPredict:
    def test_fit_returns_self(self, regression_data):
        X, y = regression_data
        dt = DecisionTreeRegressor(random_state=42)
        assert dt.fit(X, y) is dt

    def test_perfect_memorization(self, regression_data):
        """An unconstrained tree should memorize training data exactly."""
        X, y = regression_data
        dt = DecisionTreeRegressor(random_state=42)
        dt.fit(X, y)
        preds = dt.predict(X)
        for pred, target in zip(preds, y):
            assert float(pred) == pytest.approx(target, abs=1e-10)

    def test_step_function_recovery(self, step_data):
        X, y = step_data
        dt = DecisionTreeRegressor(random_state=42)
        dt.fit(X, y)
        preds = dt.predict(X)
        for pred, target in zip(preds, y):
            assert float(pred) == pytest.approx(target, abs=1e-10)

    def test_constant_target(self, constant_data):
        X, y = constant_data
        dt = DecisionTreeRegressor(random_state=42)
        dt.fit(X, y)
        preds = dt.predict(X)
        for pred in preds:
            assert float(pred) == pytest.approx(5.0, abs=1e-10)

    def test_predict_on_unseen_data(self, regression_data):
        """Predictions on unseen data must lie within training target range."""
        X, y = regression_data
        dt = DecisionTreeRegressor(random_state=42)
        dt.fit(X, y)
        preds = dt.predict([[0.5, 1.0], [1.5, 0.5]])
        assert len(preds) == 2
        for p in preds:
            assert min(y) <= float(p) <= max(y)


class TestStoppingCriteria:
    def test_max_depth_one_yields_at_most_two_leaves(self, regression_data):
        X, y = regression_data
        dt = DecisionTreeRegressor(max_depth=1, random_state=42)
        dt.fit(X, y)
        preds = dt.predict(X)
        unique_preds = {float(p) for p in preds}
        assert len(unique_preds) <= 2

    def test_min_samples_split_prevents_splitting(self):
        X = [[0.0], [1.0], [2.0], [3.0]]
        y = [0.0, 1.0, 2.0, 3.0]
        dt = DecisionTreeRegressor(min_samples_split=5, random_state=42)
        dt.fit(X, y)
        preds = dt.predict(X)
        unique_preds = {float(p) for p in preds}
        assert len(unique_preds) == 1
        assert float(preds[0]) == pytest.approx(sum(y) / len(y), abs=1e-10)

    def test_min_samples_leaf_limits_leaves(self, regression_data):
        X, y = regression_data
        dt = DecisionTreeRegressor(min_samples_leaf=5, random_state=42)
        dt.fit(X, y)
        preds = dt.predict(X)
        unique_preds = {float(p) for p in preds}
        assert len(unique_preds) <= 2


class TestFeatureSelection:
    def test_max_features_none_uses_all(self, regression_data):
        X, y = regression_data
        dt = DecisionTreeRegressor(max_features=None, random_state=42)
        dt.fit(X, y)
        assert dt._max_features == 2

    def test_max_features_int(self, regression_data):
        X, y = regression_data
        dt = DecisionTreeRegressor(max_features=1, random_state=42)
        dt.fit(X, y)
        assert dt._max_features == 1

    def test_max_features_float(self, regression_data):
        X, y = regression_data
        dt = DecisionTreeRegressor(max_features=0.5, random_state=42)
        dt.fit(X, y)
        assert dt._max_features == 1

    def test_max_features_capped_at_n_features(self, regression_data):
        X, y = regression_data
        dt = DecisionTreeRegressor(max_features=10, random_state=42)
        dt.fit(X, y)
        assert dt._max_features == 2


class TestApply:
    def test_returns_valid_indices(self, regression_data):
        X, y = regression_data
        dt = DecisionTreeRegressor(random_state=42)
        dt.fit(X, y)
        indices = dt.apply(X)
        n_nodes = len(dt._nodes)
        for idx in indices:
            assert 0 <= int(idx) < n_nodes

    def test_consistent_across_calls(self, regression_data):
        X, y = regression_data
        dt = DecisionTreeRegressor(random_state=42)
        dt.fit(X, y)
        indices1 = dt.apply(X)
        indices2 = dt.apply(X)
        for i1, i2 in zip(indices1, indices2):
            assert int(i1) == int(i2)

    def test_same_leaf_yields_same_prediction(self, regression_data):
        """Samples landing in the same leaf must get identical predictions."""
        X, y = regression_data
        dt = DecisionTreeRegressor(random_state=42)
        dt.fit(X, y)
        preds = dt.predict(X)
        indices = dt.apply(X)
        leaf_to_pred = {}
        for pred, idx in zip(preds, indices):
            leaf = int(idx)
            if leaf in leaf_to_pred:
                assert float(pred) == leaf_to_pred[leaf]
            else:
                leaf_to_pred[leaf] = float(pred)


class TestTreeStructure:
    def test_impurity_array_matches_node_count(self, regression_data):
        X, y = regression_data
        dt = DecisionTreeRegressor(random_state=42)
        dt.fit(X, y)
        assert len(dt.tree_.impurity) == len(dt._nodes)

    def test_node_index_mapping_complete(self, regression_data):
        X, y = regression_data
        dt = DecisionTreeRegressor(random_state=42)
        dt.fit(X, y)
        assert len(dt.tree_._node_to_index) == len(dt._nodes)

    def test_single_sample_leaf_has_zero_impurity(self):
        X = [[0.0], [1.0]]
        y = [0.0, 1.0]
        dt = DecisionTreeRegressor(random_state=42)
        dt.fit(X, y)
        for node in dt._nodes:
            if node.feature is None and node.n_samples == 1:
                assert node.impurity == pytest.approx(0.0, abs=1e-10)


class TestReproducibility:
    def test_same_seed_same_predictions(self, regression_data):
        X, y = regression_data
        dt1 = DecisionTreeRegressor(max_features=1, random_state=42)
        dt1.fit(X, y)
        dt2 = DecisionTreeRegressor(max_features=1, random_state=42)
        dt2.fit(X, y)
        preds1 = dt1.predict(X)
        preds2 = dt2.predict(X)
        for p1, p2 in zip(preds1, preds2):
            assert float(p1) == float(p2)
