"""
Core tests for the math backend.

Tests both the SciPy backend and the pure Python fallback to ensure
they produce equivalent results.
"""

import math

import pytest

from gradient_free_optimizers._math_backend import HAS_SCIPY
from gradient_free_optimizers._math_backend import _pure as pure_math

# Conditionally import scipy backend
if HAS_SCIPY:
    from gradient_free_optimizers._math_backend import _scipy as scipy_backend
else:
    scipy_backend = None

# Skip all tests in this module if scipy is not available
pytestmark = pytest.mark.skipif(
    not HAS_SCIPY, reason="SciPy not available for comparison tests"
)


# =============================================================================
# Normal Distribution Tests
# =============================================================================


class TestNormDistribution:
    """Test normal distribution functions."""

    @pytest.mark.parametrize(
        ("x", "expected_approx"),
        [
            (0, 0.5),
            (-3, 0.00135),
            (3, 0.99865),
            (1, 0.8413),
            (-1, 0.1587),
        ],
    )
    def test_norm_cdf(self, x, expected_approx):
        scipy_result = scipy_backend.norm_cdf(x)
        pure_result = pure_math.norm_cdf(x)
        # Check against expected value
        assert abs(scipy_result - expected_approx) < 0.001
        # Check pure matches scipy
        assert abs(scipy_result - pure_result) < 0.001

    @pytest.mark.parametrize(
        "x",
        [0, 1, -1, 2, -2, 0.5, -0.5],
    )
    def test_norm_pdf(self, x):
        scipy_result = scipy_backend.norm_pdf(x)
        pure_result = pure_math.norm_pdf(x)
        assert abs(scipy_result - pure_result) < 1e-6

    def test_norm_cdf_with_loc_scale(self):
        """Test normal CDF with non-standard parameters."""
        x = 10
        loc = 5
        scale = 2

        scipy_result = scipy_backend.norm_cdf(x, loc=loc, scale=scale)
        pure_result = pure_math.norm_cdf(x, loc=loc, scale=scale)

        # (10 - 5) / 2 = 2.5 standard deviations
        assert abs(scipy_result - pure_result) < 0.001
        assert scipy_result > 0.99  # Should be very high

    def test_norm_pdf_with_loc_scale(self):
        """Test normal PDF with non-standard parameters."""
        x = 5
        loc = 5
        scale = 2

        scipy_result = scipy_backend.norm_pdf(x, loc=loc, scale=scale)
        pure_result = pure_math.norm_pdf(x, loc=loc, scale=scale)

        # At the mean, PDF should be 1/(scale * sqrt(2*pi))
        expected = 1 / (scale * math.sqrt(2 * math.pi))
        assert abs(scipy_result - expected) < 1e-10
        assert abs(pure_result - expected) < 1e-10


# =============================================================================
# Distance Function Tests
# =============================================================================


class TestDistanceFunctions:
    """Test distance calculation functions."""

    def test_cdist_euclidean_1d(self):
        xa = [[0], [1], [2]]
        xb = [[0], [3]]
        scipy_result = scipy_backend.cdist(xa, xb)
        pure_result = pure_math.cdist(xa, xb)

        # Expected distances:
        # (0,0)=0, (0,3)=3, (1,0)=1, (1,3)=2, (2,0)=2, (2,3)=1
        expected = [[0, 3], [1, 2], [2, 1]]

        for i in range(3):
            for j in range(2):
                assert abs(scipy_result[i, j] - expected[i][j]) < 1e-10
                assert abs(pure_result[i][j] - expected[i][j]) < 1e-10

    def test_cdist_euclidean_2d(self):
        xa = [[0, 0], [1, 1]]
        xb = [[1, 0], [0, 1]]
        scipy_result = scipy_backend.cdist(xa, xb)
        pure_result = pure_math.cdist(xa, xb)

        # (0,0) to (1,0) = 1, (0,0) to (0,1) = 1
        # (1,1) to (1,0) = 1, (1,1) to (0,1) = 1
        for i in range(2):
            for j in range(2):
                assert abs(scipy_result[i, j] - 1.0) < 1e-10
                assert abs(pure_result[i][j] - 1.0) < 1e-10

    def test_cdist_larger_matrices(self):
        """Test cdist with larger matrices."""
        xa = [[0, 0], [1, 0], [2, 0], [3, 0]]
        xb = [[0, 0], [0, 1], [0, 2]]

        scipy_result = scipy_backend.cdist(xa, xb)
        pure_result = pure_math.cdist(xa, xb)

        assert scipy_result.shape == (4, 3)
        assert len(pure_result) == 4
        assert len(pure_result[0]) == 3

        # Check some values
        # (0,0) to (0,0) = 0
        assert abs(scipy_result[0, 0] - 0) < 1e-10
        assert abs(pure_result[0][0] - 0) < 1e-10

        # (3,0) to (0,2) = sqrt(9 + 4) = sqrt(13)
        expected = math.sqrt(13)
        assert abs(scipy_result[3, 2] - expected) < 1e-10
        assert abs(pure_result[3][2] - expected) < 1e-10


# =============================================================================
# Special Function Tests
# =============================================================================


class TestLogsumexp:
    """Test numerically stable log-sum-exp."""

    def test_logsumexp_basic(self):
        values = [1, 2, 3]
        scipy_result = scipy_backend.logsumexp(values)
        pure_result = pure_math.logsumexp(values)
        # log(e^1 + e^2 + e^3) = log(e + e^2 + e^3) ~ 3.41
        assert abs(scipy_result - pure_result) < 1e-6

    def test_logsumexp_large_values(self):
        """Test numerical stability with large values."""
        values = [1000, 1001, 1002]
        scipy_result = scipy_backend.logsumexp(values)
        pure_result = pure_math.logsumexp(values)
        # Should not overflow
        assert abs(scipy_result - pure_result) < 1e-6
        assert scipy_result > 1002  # Should be slightly > max

    def test_logsumexp_negative_values(self):
        """Test with negative values."""
        values = [-100, -99, -98]
        scipy_result = scipy_backend.logsumexp(values)
        pure_result = pure_math.logsumexp(values)
        assert abs(scipy_result - pure_result) < 1e-6
        assert scipy_result > -98  # Should be slightly > max

    def test_logsumexp_single_value(self):
        """Test with single value."""
        values = [5]
        scipy_result = scipy_backend.logsumexp(values)
        pure_result = pure_math.logsumexp(values)
        # logsumexp([x]) = x
        assert abs(scipy_result - 5) < 1e-10
        assert abs(pure_result - 5) < 1e-10


# =============================================================================
# Backend Selection Tests
# =============================================================================


class TestCholesky:
    """Test Cholesky decomposition."""

    def test_3x3_positive_definite(self):
        mat = [[4, 2, 1], [2, 5, 3], [1, 3, 6]]
        scipy_L = scipy_backend.cholesky(mat, lower=True)
        pure_L = pure_math.cholesky(mat, lower=True)

        for i in range(3):
            for j in range(3):
                assert abs(scipy_L[i, j] - pure_L[i][j]) < 1e-8

    def test_5x5_positive_definite(self):
        mat = [
            [10, 1, 0, 0, 0],
            [1, 10, 1, 0, 0],
            [0, 1, 10, 1, 0],
            [0, 0, 1, 10, 1],
            [0, 0, 0, 1, 10],
        ]
        scipy_L = scipy_backend.cholesky(mat, lower=True)
        pure_L = pure_math.cholesky(mat, lower=True)

        for i in range(5):
            for j in range(5):
                assert abs(scipy_L[i, j] - pure_L[i][j]) < 1e-8


class TestChoSolve:
    """Test solving via Cholesky factorization."""

    def test_solution_accuracy(self):
        mat = [[4, 2], [2, 5]]
        b = [1.0, 2.0]

        scipy_L = scipy_backend.cholesky(mat, lower=True)
        scipy_x = scipy_backend.cho_solve((scipy_L, True), b)

        pure_L = pure_math.cholesky(mat, lower=True)
        pure_x = pure_math.cho_solve((pure_L, True), b)

        for i in range(2):
            assert abs(float(scipy_x[i]) - float(pure_x[i])) < 1e-8

    def test_3x3_system(self):
        mat = [[4, 2, 1], [2, 5, 3], [1, 3, 6]]
        b = [1.0, 0.0, -1.0]

        scipy_L = scipy_backend.cholesky(mat, lower=True)
        scipy_x = scipy_backend.cho_solve((scipy_L, True), b)

        pure_L = pure_math.cholesky(mat, lower=True)
        pure_x = pure_math.cho_solve((pure_L, True), b)

        for i in range(3):
            assert abs(float(scipy_x[i]) - float(pure_x[i])) < 1e-8


class TestSolve:
    """Test Gaussian elimination solver."""

    def test_2x2_system(self):
        a = [[2.0, 1.0], [1.0, 3.0]]
        b = [5.0, 7.0]
        scipy_x = scipy_backend.solve(a, b)
        pure_x = pure_math.solve(a, b)
        for i in range(2):
            assert abs(float(scipy_x[i]) - float(pure_x[i])) < 1e-8

    def test_3x3_system(self):
        a = [[3.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 3.0]]
        b = [4.0, 6.0, 4.0]
        scipy_x = scipy_backend.solve(a, b)
        pure_x = pure_math.solve(a, b)
        for i in range(3):
            assert abs(float(scipy_x[i]) - float(pure_x[i])) < 1e-8


class TestSolveTriangular:
    """Test triangular system solver."""

    def test_lower_triangular_1d(self):
        L = [[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.5, 1.0, 4.0]]
        b = [2.0, 5.0, 10.5]
        scipy_x = scipy_backend.solve_triangular(L, b, lower=True)
        pure_x = pure_math.solve_triangular(L, b, lower=True)
        for i in range(3):
            assert abs(float(scipy_x[i]) - float(pure_x[i])) < 1e-8

    def test_lower_triangular_2d(self):
        L = [[2.0, 0.0], [1.0, 3.0]]
        b = [[4.0, 6.0], [7.0, 9.0]]
        scipy_x = scipy_backend.solve_triangular(L, b, lower=True)
        pure_x = pure_math.solve_triangular(L, b, lower=True)
        for i in range(2):
            for j in range(2):
                assert abs(float(scipy_x[i, j]) - float(pure_x[i][j])) < 1e-8

    def test_upper_triangular(self):
        U = [[3.0, 1.0, 2.0], [0.0, 4.0, 1.0], [0.0, 0.0, 2.0]]
        b = [9.0, 9.0, 4.0]
        scipy_x = scipy_backend.solve_triangular(U, b, lower=False)
        pure_x = pure_math.solve_triangular(U, b, lower=False)
        for i in range(3):
            assert abs(float(scipy_x[i]) - float(pure_x[i])) < 1e-8


class TestCdistPurePath:
    """Test the math.dist based cdist implementation."""

    def test_3d_points(self):
        xa = [[0, 0, 0], [1, 1, 1]]
        xb = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        scipy_result = scipy_backend.cdist(xa, xb)
        pure_result = pure_math.cdist(xa, xb)

        for i in range(2):
            for j in range(3):
                assert abs(scipy_result[i, j] - pure_result[i][j]) < 1e-10

    def test_single_dimension(self):
        xa = [[0], [5], [10]]
        xb = [[3], [7]]
        scipy_result = scipy_backend.cdist(xa, xb)
        pure_result = pure_math.cdist(xa, xb)

        expected = [[3, 7], [2, 2], [7, 3]]
        for i in range(3):
            for j in range(2):
                assert abs(scipy_result[i, j] - expected[i][j]) < 1e-10
                assert abs(pure_result[i][j] - expected[i][j]) < 1e-10


class TestBackendSelection:
    """Test that backend selection works correctly."""

    def test_has_scipy_flag(self):
        assert HAS_SCIPY is True  # We know SciPy is installed in test env

    def test_math_backend_uses_scipy(self):
        import gradient_free_optimizers._math_backend as math_backend

        if HAS_SCIPY:
            assert math_backend._backend_name == "scipy"


# =============================================================================
# Integration Tests
# =============================================================================


class TestMathBackendIntegration:
    """Integration tests simulating typical GFO workflows."""

    def test_expected_improvement_calculation(self):
        """Simulate Expected Improvement acquisition function."""
        # In EI: improvement = (mu - best) * CDF(z) + sigma * PDF(z)
        # where z = (mu - best) / sigma

        mu = 0.8  # Predicted mean
        sigma = 0.1  # Predicted uncertainty
        best = 0.7  # Best score so far

        z = (mu - best) / sigma  # = 1.0

        cdf_scipy = scipy_backend.norm_cdf(z)
        pdf_scipy = scipy_backend.norm_pdf(z)

        cdf_pure = pure_math.norm_cdf(z)
        pdf_pure = pure_math.norm_pdf(z)

        # Calculate EI with both backends
        ei_scipy = (mu - best) * cdf_scipy + sigma * pdf_scipy
        ei_pure = (mu - best) * cdf_pure + sigma * pdf_pure

        assert abs(ei_scipy - ei_pure) < 1e-6
        assert ei_scipy > 0  # Should have positive expected improvement

    def test_distance_based_exploration(self):
        """Simulate distance-based position validation."""
        # Current positions that have been evaluated
        evaluated = [[0, 0], [1, 1], [2, 2]]
        # New candidate position
        candidate = [[1.5, 1.5]]

        distances_scipy = scipy_backend.cdist(candidate, evaluated)
        distances_pure = pure_math.cdist(candidate, evaluated)

        # Find minimum distance
        min_dist_scipy = min(distances_scipy[0])
        min_dist_pure = min(distances_pure[0])

        assert abs(min_dist_scipy - min_dist_pure) < 1e-10
        # Distance to (1,1) should be sqrt(0.5^2 + 0.5^2) = sqrt(0.5) ~ 0.707
        expected_min = math.sqrt(0.5)
        assert abs(min_dist_scipy - expected_min) < 1e-10
