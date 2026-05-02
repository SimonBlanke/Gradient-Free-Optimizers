"""Tests for the native KernelDensityEstimator."""

import math

import pytest

from gradient_free_optimizers._array_backend import array as arr
from gradient_free_optimizers._estimators import KernelDensityEstimator


class TestFit:
    def test_fit_returns_self(self):
        X = [[0.0], [1.0], [2.0]]
        kde = KernelDensityEstimator(bandwidth=1.0)
        assert kde.fit(X) is kde

    def test_silverman_bandwidth(self):
        X_train = [[0.0], [1.0], [2.0], [3.0], [4.0]]
        kde = KernelDensityEstimator()
        kde.fit(X_train)

        vals = [0.0, 1.0, 2.0, 3.0, 4.0]
        n = len(vals)
        d = 1
        mean_val = sum(vals) / n
        variance = sum((x - mean_val) ** 2 for x in vals) / (n - 1)
        sigma = math.sqrt(variance)
        factor = (4 / (d + 2)) ** (1 / (d + 4))
        expected = factor * sigma * n ** (-1 / (d + 4))

        assert kde.bandwidth == pytest.approx(expected, abs=1e-10)

    def test_custom_bandwidth_used(self):
        kde = KernelDensityEstimator(bandwidth=0.5)
        kde.fit([[0.0], [1.0], [2.0]])
        assert kde.bandwidth == 0.5

    def test_constant_data_raises_due_to_zero_bandwidth(self):
        """Silverman's rule yields bandwidth=0 for constant data."""
        kde = KernelDensityEstimator()
        with pytest.raises(ValueError, match="bandwidth must be positive"):
            kde.fit([[1.0], [1.0], [1.0]])

    def test_negative_bandwidth_raises(self):
        kde = KernelDensityEstimator(bandwidth=-0.5)
        with pytest.raises(ValueError, match="bandwidth must be positive"):
            kde.fit([[0.0], [1.0]])

    def test_1d_array_input(self):
        X = arr([0.0, 1.0, 2.0])
        kde = KernelDensityEstimator(bandwidth=1.0)
        kde.fit(X)
        assert kde.n_features == 1
        assert kde.n_samples == 3

    def test_multidimensional_input(self):
        X = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        kde = KernelDensityEstimator(bandwidth=1.0)
        kde.fit(X)
        assert kde.n_features == 2
        assert kde.n_samples == 3


class TestScoreSamples:
    def test_not_fitted_raises(self):
        kde = KernelDensityEstimator()
        with pytest.raises(RuntimeError, match="must be fitted"):
            kde.score_samples([[0.0]])

    def test_log_density_returns_correct_length(self):
        kde = KernelDensityEstimator(bandwidth=1.0)
        kde.fit([[0.0], [1.0], [2.0]])
        scores = kde.score_samples([[0.5], [1.5]])
        assert len(scores) == 2

    def test_log_density_values_are_finite(self):
        kde = KernelDensityEstimator(bandwidth=1.0)
        kde.fit([[0.0], [1.0], [2.0]])
        scores = kde.score_samples([[0.0], [1.0], [2.0]], log=True)
        for s in scores:
            assert math.isfinite(float(s))

    def test_log_vs_density_consistency(self):
        """exp(log_density) should equal density (above the atol floor)."""
        kde = KernelDensityEstimator(bandwidth=1.0)
        kde.fit([[0.0], [1.0], [2.0]])
        X_test = [[0.5], [1.0], [1.5]]
        log_d = kde.score_samples(X_test, log=True)
        d = kde.score_samples(X_test, log=False)
        for ld, dd in zip(log_d, d):
            expected = math.exp(float(ld))
            if expected < 1e-12:
                assert float(dd) == 0.0
            else:
                assert float(dd) == pytest.approx(expected, abs=1e-10)

    def test_density_peaks_at_training_cluster(self):
        X_train = [[0.0], [0.0], [0.0]]
        kde = KernelDensityEstimator(bandwidth=0.5)
        kde.fit(X_train)
        d_at_cluster = kde.score_samples([[0.0]], log=False)
        d_far_away = kde.score_samples([[5.0]], log=False)
        assert float(d_at_cluster[0]) > float(d_far_away[0])

    def test_density_is_nonnegative(self):
        kde = KernelDensityEstimator(bandwidth=1.0)
        kde.fit([[0.0], [1.0], [2.0]])
        test_points = [[-5.0], [0.0], [1.0], [2.0], [7.0]]
        densities = kde.score_samples(test_points, log=False)
        for d in densities:
            assert float(d) >= 0.0

    def test_log_density_is_finite(self):
        """Log-density should be finite for all query points."""
        kde = KernelDensityEstimator(bandwidth=5.0)
        kde.fit([[0.0], [1.0], [2.0]])
        scores = kde.score_samples([[0.0], [1.0], [10.0]], log=True)
        for s in scores:
            assert math.isfinite(float(s))

    def test_density_integrates_to_approximately_one(self):
        """Numerical integration of the density should yield approximately 1."""
        kde = KernelDensityEstimator(bandwidth=1.0)
        kde.fit([[0.0], [2.0]])

        dx = 0.02
        points = [[-6.0 + i * dx] for i in range(600)]
        densities = kde.score_samples(points, log=False)
        integral = sum(float(d) for d in densities) * dx
        assert integral == pytest.approx(1.0, abs=0.05)
