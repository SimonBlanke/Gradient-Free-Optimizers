"""Tests for the native GaussianProcessRegressor."""

import math

import pytest

from gradient_free_optimizers._estimators import GaussianProcessRegressor


class TestFitPredict:
    def test_fit_returns_self(self):
        X = [[0.0], [1.0], [2.0], [3.0], [4.0]]
        y = [0.0, 1.0, 2.0, 3.0, 4.0]
        gp = GaussianProcessRegressor(optimize=False)
        assert gp.fit(X, y) is gp

    def test_predict_at_training_points(self):
        """GP posterior mean should closely recover training targets."""
        X = [[0.0], [1.0], [2.0], [3.0], [4.0]]
        y = [0.0, 1.0, 2.0, 3.0, 4.0]
        theta = [math.log(1.5), math.log(1.0), math.log(0.1)]
        gp = GaussianProcessRegressor(theta=theta, optimize=False)
        gp.fit(X, y)
        preds = gp.predict(X)
        for pred, target in zip(preds, y):
            assert float(pred) == pytest.approx(target, abs=0.1)

    def test_predict_returns_correct_length(self):
        X = [[0.0], [1.0], [2.0]]
        y = [0.0, 1.0, 4.0]
        gp = GaussianProcessRegressor(optimize=False)
        gp.fit(X, y)
        preds = gp.predict([[0.5], [1.5], [2.5]])
        assert len(preds) == 3

    def test_multidimensional_input(self):
        X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        y = [0.0, 1.0, 1.0, 2.0]
        gp = GaussianProcessRegressor(optimize=False)
        gp.fit(X, y)
        preds = gp.predict(X)
        assert len(preds) == 4
        for p in preds:
            assert math.isfinite(float(p))


class TestUncertainty:
    def test_return_std_shape(self):
        X = [[0.0], [1.0], [2.0]]
        y = [0.0, 1.0, 4.0]
        gp = GaussianProcessRegressor(optimize=False)
        gp.fit(X, y)
        mu, std = gp.predict([[0.5], [1.5]], return_std=True)
        assert len(mu) == 2
        assert len(std) == 2

    def test_std_nonnegative(self):
        X = [[0.0], [1.0], [2.0]]
        y = [0.0, 1.0, 4.0]
        gp = GaussianProcessRegressor(optimize=False)
        gp.fit(X, y)
        test_points = [[0.0], [0.5], [1.0], [1.5], [2.0], [5.0]]
        _, std = gp.predict(test_points, return_std=True)
        for s in std:
            assert float(s) >= -1e-10

    def test_uncertainty_grows_away_from_data(self):
        X = [[0.0], [1.0], [2.0]]
        y = [0.0, 1.0, 4.0]
        gp = GaussianProcessRegressor(optimize=False)
        gp.fit(X, y)
        _, std_near = gp.predict([[1.0]], return_std=True)
        _, std_far = gp.predict([[10.0]], return_std=True)
        assert float(std_far[0]) > float(std_near[0])


class TestRBFKernel:
    def test_diagonal_returns_sigma_f_squared(self):
        X = [[0.0], [1.0], [2.0]]
        log_sigma_f = 0.5
        theta = [0.0, log_sigma_f, -2.0]
        K_diag = GaussianProcessRegressor._rbf_kernel(X, X, theta, diag=True)
        expected = math.exp(2 * log_sigma_f)
        for val in K_diag:
            assert float(val) == pytest.approx(expected, abs=1e-10)

    def test_kernel_matrix_is_symmetric(self):
        X = [[0.0], [1.0], [2.0], [3.0]]
        theta = [0.0, 0.0, -2.0]
        K = GaussianProcessRegressor._rbf_kernel(X, X, theta)
        n = len(X)
        for i in range(n):
            for j in range(i + 1, n):
                assert float(K[i][j]) == pytest.approx(float(K[j][i]), abs=1e-10)

    def test_diagonal_of_full_kernel_equals_sigma_f_squared(self):
        """K(x, x) on the full matrix should equal sigma_f^2."""
        X = [[0.0], [1.0], [2.0]]
        log_sigma_f = 0.3
        theta = [0.0, log_sigma_f, -2.0]
        K = GaussianProcessRegressor._rbf_kernel(X, X, theta)
        expected = math.exp(2 * log_sigma_f)
        for i in range(len(X)):
            assert float(K[i][i]) == pytest.approx(expected, abs=1e-10)

    def test_kernel_decays_with_distance(self):
        """Kernel value should decrease as distance between points grows."""
        theta = [0.0, 0.0, -2.0]
        K_close = GaussianProcessRegressor._rbf_kernel([[0.0]], [[0.1]], theta)
        K_far = GaussianProcessRegressor._rbf_kernel([[0.0]], [[5.0]], theta)
        assert float(K_close[0][0]) > float(K_far[0][0])


class TestOptimization:
    def test_optimize_false_preserves_theta(self):
        X = [[0.0], [1.0], [2.0]]
        y = [0.0, 1.0, 2.0]
        theta0 = [math.log(1.0), math.log(1.0), math.log(0.1)]
        gp = GaussianProcessRegressor(theta=list(theta0), optimize=False)
        gp.fit(X, y)
        for t_fit, t_init in zip(gp.theta, theta0):
            assert float(t_fit) == pytest.approx(t_init, abs=1e-10)

    def test_optimize_true_produces_finite_predictions(self):
        X = [[0.0], [1.0], [2.0], [3.0], [4.0]]
        y = [0.0, 1.0, 4.0, 9.0, 16.0]
        gp = GaussianProcessRegressor(optimize=True)
        gp.fit(X, y)
        preds = gp.predict(X)
        for p in preds:
            assert math.isfinite(float(p))
