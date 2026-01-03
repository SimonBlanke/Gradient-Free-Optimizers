"""
Tests for array backend and math backend functions.

These tests verify that the pure Python implementations produce results
consistent with NumPy/SciPy implementations.
"""

import pytest
import math

# Import both backends for comparison testing
from gradient_free_optimizers._array_backend import _pure as pure_array
from gradient_free_optimizers._array_backend import _numpy as numpy_array
from gradient_free_optimizers._math_backend import _pure as pure_math
from gradient_free_optimizers._math_backend import _scipy as scipy_math

import numpy as np


# =============================================================================
# Test: __matmul__ operator (@)
# =============================================================================

class TestMatmul:
    """Tests for matrix multiplication operator."""

    def test_matmul_1d_1d_dot_product(self):
        """1D @ 1D should return scalar dot product."""
        a = pure_array.array([1.0, 2.0, 3.0])
        b = pure_array.array([4.0, 5.0, 6.0])
        result = a @ b
        expected = 1*4 + 2*5 + 3*6  # = 32
        assert result == expected

    def test_matmul_2d_1d_matrix_vector(self):
        """2D @ 1D should return 1D vector."""
        A = pure_array.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        v = pure_array.array([1.0, 2.0])
        result = A @ v
        # [[1, 2], [3, 4], [5, 6]] @ [1, 2] = [5, 11, 17]
        expected = [1*1 + 2*2, 3*1 + 4*2, 5*1 + 6*2]
        assert list(result) == expected

    def test_matmul_1d_2d_vector_matrix(self):
        """1D @ 2D should return 1D vector."""
        v = pure_array.array([1.0, 2.0, 3.0])
        A = pure_array.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = v @ A
        # [1, 2, 3] @ [[1, 2], [3, 4], [5, 6]] = [22, 28]
        expected = [1*1 + 2*3 + 3*5, 1*2 + 2*4 + 3*6]
        assert list(result) == expected

    def test_matmul_2d_2d_matrix_matrix(self):
        """2D @ 2D should return 2D matrix."""
        A = pure_array.array([[1.0, 2.0], [3.0, 4.0]])
        B = pure_array.array([[5.0, 6.0], [7.0, 8.0]])
        result = A @ B
        # [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        expected = [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]
        assert result.shape == (2, 2)
        for i in range(2):
            for j in range(2):
                assert result[i, j] == expected[i][j]

    def test_matmul_matches_numpy(self):
        """Pure Python matmul should match NumPy results."""
        # Create test data
        A_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        B_data = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]

        A_pure = pure_array.array(A_data)
        B_pure = pure_array.array(B_data)
        A_np = np.array(A_data)
        B_np = np.array(B_data)

        result_pure = A_pure @ B_pure
        result_np = A_np @ B_np

        for i in range(2):
            for j in range(2):
                assert abs(result_pure[i, j] - result_np[i, j]) < 1e-10

    def test_matmul_incompatible_dimensions_raises(self):
        """Incompatible dimensions should raise ValueError."""
        A = pure_array.array([[1.0, 2.0], [3.0, 4.0]])
        B = pure_array.array([[1.0, 2.0, 3.0]])  # 1x3, incompatible with 2x2
        with pytest.raises(ValueError):
            A @ B

    def test_rmatmul(self):
        """Test right matrix multiplication."""
        A = pure_array.array([[1.0, 2.0], [3.0, 4.0]])
        # Test with list (should trigger __rmatmul__)
        result = [[5.0, 6.0], [7.0, 8.0]] @ A
        # This should work via __rmatmul__
        assert result is not None


# =============================================================================
# Test: diag function
# =============================================================================

class TestDiag:
    """Tests for diagonal extraction/construction."""

    def test_diag_extract_from_2d(self):
        """Extract main diagonal from 2D matrix."""
        A = pure_array.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = pure_array.diag(A)
        expected = [1.0, 5.0, 9.0]
        assert list(result) == expected

    def test_diag_extract_rectangular(self):
        """Extract diagonal from rectangular matrix."""
        A = pure_array.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        result = pure_array.diag(A)
        expected = [1.0, 6.0]  # Only 2 diagonal elements
        assert list(result) == expected

    def test_diag_extract_with_offset(self):
        """Extract diagonal with positive offset (superdiagonal)."""
        A = pure_array.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = pure_array.diag(A, k=1)
        expected = [2.0, 6.0]  # Superdiagonal
        assert list(result) == expected

    def test_diag_extract_with_negative_offset(self):
        """Extract diagonal with negative offset (subdiagonal)."""
        A = pure_array.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = pure_array.diag(A, k=-1)
        expected = [4.0, 8.0]  # Subdiagonal
        assert list(result) == expected

    def test_diag_construct_from_1d(self):
        """Construct diagonal matrix from 1D vector."""
        v = pure_array.array([1.0, 2.0, 3.0])
        result = pure_array.diag(v)
        expected = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        assert result.shape == (3, 3)
        for i in range(3):
            for j in range(3):
                assert result[i, j] == expected[i][j]

    def test_diag_construct_with_offset(self):
        """Construct diagonal matrix with offset."""
        v = pure_array.array([1.0, 2.0])
        result = pure_array.diag(v, k=1)
        expected = [[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]
        assert result.shape == (3, 3)
        for i in range(3):
            for j in range(3):
                assert result[i, j] == expected[i][j]

    def test_diag_matches_numpy(self):
        """Pure Python diag should match NumPy results."""
        A_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]

        # Extract diagonal
        A_pure = pure_array.array(A_data)
        A_np = np.array(A_data)

        result_pure = pure_array.diag(A_pure)
        result_np = np.diag(A_np)

        assert list(result_pure) == list(result_np)

        # Construct diagonal
        v_data = [1.0, 2.0, 3.0]
        v_pure = pure_array.array(v_data)
        v_np = np.array(v_data)

        result_pure = pure_array.diag(v_pure)
        result_np = np.diag(v_np)

        for i in range(3):
            for j in range(3):
                assert result_pure[i, j] == result_np[i, j]


# =============================================================================
# Test: empty_like function
# =============================================================================

class TestEmptyLike:
    """Tests for empty_like function."""

    def test_empty_like_1d(self):
        """empty_like should create array with same shape as 1D input."""
        a = pure_array.array([1.0, 2.0, 3.0, 4.0])
        result = pure_array.empty_like(a)
        assert result.shape == a.shape
        assert len(result) == 4

    def test_empty_like_2d(self):
        """empty_like should create array with same shape as 2D input."""
        a = pure_array.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = pure_array.empty_like(a)
        assert result.shape == a.shape
        assert result.shape == (2, 3)

    def test_empty_like_with_dtype(self):
        """empty_like should respect dtype parameter."""
        a = pure_array.array([1.0, 2.0, 3.0])
        result = pure_array.empty_like(a, dtype=int)
        # All values should be 0 (integers)
        assert all(v == 0 for v in result)


# =============================================================================
# Test: Shape-preserving elementwise functions
# =============================================================================

class TestElementwiseShapePreservation:
    """Tests for elementwise functions preserving 2D shape."""

    def test_exp_preserves_2d_shape(self):
        """exp should preserve 2D array shape."""
        A = pure_array.array([[1.0, 2.0], [3.0, 4.0]])
        result = pure_array.exp(A)
        assert result.shape == (2, 2)
        assert abs(result[0, 0] - math.exp(1.0)) < 1e-10
        assert abs(result[1, 1] - math.exp(4.0)) < 1e-10

    def test_log_preserves_2d_shape(self):
        """log should preserve 2D array shape."""
        A = pure_array.array([[1.0, 2.0], [3.0, 4.0]])
        result = pure_array.log(A)
        assert result.shape == (2, 2)
        assert abs(result[0, 0] - math.log(1.0)) < 1e-10
        assert abs(result[1, 1] - math.log(4.0)) < 1e-10

    def test_sqrt_preserves_2d_shape(self):
        """sqrt should preserve 2D array shape."""
        A = pure_array.array([[1.0, 4.0], [9.0, 16.0]])
        result = pure_array.sqrt(A)
        assert result.shape == (2, 2)
        assert abs(result[0, 0] - 1.0) < 1e-10
        assert abs(result[0, 1] - 2.0) < 1e-10
        assert abs(result[1, 0] - 3.0) < 1e-10
        assert abs(result[1, 1] - 4.0) < 1e-10

    def test_abs_preserves_2d_shape(self):
        """abs should preserve 2D array shape."""
        A = pure_array.array([[-1.0, 2.0], [-3.0, 4.0]])
        result = pure_array.abs(A)
        assert result.shape == (2, 2)
        assert result[0, 0] == 1.0
        assert result[1, 0] == 3.0

    def test_sin_preserves_2d_shape(self):
        """sin should preserve 2D array shape."""
        A = pure_array.array([[0.0, math.pi/2], [math.pi, 3*math.pi/2]])
        result = pure_array.sin(A)
        assert result.shape == (2, 2)
        assert abs(result[0, 0] - 0.0) < 1e-10
        assert abs(result[0, 1] - 1.0) < 1e-10

    def test_cos_preserves_2d_shape(self):
        """cos should preserve 2D array shape."""
        A = pure_array.array([[0.0, math.pi/2], [math.pi, 3*math.pi/2]])
        result = pure_array.cos(A)
        assert result.shape == (2, 2)
        assert abs(result[0, 0] - 1.0) < 1e-10
        assert abs(result[0, 1] - 0.0) < 1e-10

    def test_power_preserves_2d_shape(self):
        """power should preserve 2D array shape."""
        A = pure_array.array([[1.0, 2.0], [3.0, 4.0]])
        result = pure_array.power(A, 2)
        assert result.shape == (2, 2)
        assert result[0, 0] == 1.0
        assert result[0, 1] == 4.0
        assert result[1, 0] == 9.0
        assert result[1, 1] == 16.0

    def test_elementwise_matches_numpy_2d(self):
        """Elementwise functions should match NumPy for 2D arrays."""
        data = [[1.0, 2.0], [3.0, 4.0]]

        A_pure = pure_array.array(data)
        A_np = np.array(data)

        # Test exp
        result_pure = pure_array.exp(A_pure)
        result_np = np.exp(A_np)
        for i in range(2):
            for j in range(2):
                assert abs(result_pure[i, j] - result_np[i, j]) < 1e-10

        # Test sqrt
        result_pure = pure_array.sqrt(A_pure)
        result_np = np.sqrt(A_np)
        for i in range(2):
            for j in range(2):
                assert abs(result_pure[i, j] - result_np[i, j]) < 1e-10


# =============================================================================
# Test: solve_triangular function
# =============================================================================

class TestSolveTriangular:
    """Tests for triangular system solver."""

    def test_solve_triangular_lower_1d(self):
        """Solve lower triangular system with 1D right-hand side."""
        # L @ x = b where L is lower triangular
        L = np.array([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [4.0, 2.0, 1.0]])
        b = np.array([4.0, 7.0, 12.0])

        result = pure_math.solve_triangular(L, b, lower=True)

        # Verify: L @ result should equal b
        check = L @ result
        for i in range(3):
            assert abs(check[i] - b[i]) < 1e-10

    def test_solve_triangular_upper_1d(self):
        """Solve upper triangular system with 1D right-hand side."""
        # U @ x = b where U is upper triangular
        U = np.array([[2.0, 1.0, 4.0], [0.0, 3.0, 2.0], [0.0, 0.0, 1.0]])
        b = np.array([13.0, 8.0, 2.0])

        result = pure_math.solve_triangular(U, b, lower=False)

        # Verify: U @ result should equal b
        check = U @ result
        for i in range(3):
            assert abs(check[i] - b[i]) < 1e-10

    def test_solve_triangular_2d_rhs(self):
        """Solve triangular system with 2D right-hand side (multiple vectors)."""
        L = np.array([[2.0, 0.0], [3.0, 4.0]])
        B = np.array([[4.0, 6.0], [14.0, 20.0]])  # Two column vectors

        result = pure_math.solve_triangular(L, B, lower=True)

        # Verify: L @ result should equal B
        check = L @ result
        for i in range(2):
            for j in range(2):
                assert abs(check[i, j] - B[i, j]) < 1e-10

    def test_solve_triangular_matches_scipy(self):
        """Pure Python solve_triangular should match SciPy results."""
        L = np.array([[3.0, 0.0, 0.0], [2.0, 4.0, 0.0], [1.0, 5.0, 6.0]])
        b = np.array([9.0, 16.0, 33.0])

        result_pure = pure_math.solve_triangular(L, b, lower=True)
        result_scipy = scipy_math.solve_triangular(L, b, lower=True)

        for i in range(3):
            assert abs(result_pure[i] - result_scipy[i]) < 1e-10

    def test_solve_triangular_identity(self):
        """Solve with identity matrix should return b."""
        I = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        b = np.array([1.0, 2.0, 3.0])

        result = pure_math.solve_triangular(I, b, lower=True)

        for i in range(3):
            assert abs(result[i] - b[i]) < 1e-10


# =============================================================================
# Test: solve function with assume_a parameter
# =============================================================================

class TestSolveAssumeA:
    """Tests for solve function with assume_a parameter."""

    def test_solve_with_assume_a_pos(self):
        """solve with assume_a='pos' should work for positive definite matrix."""
        # Create a positive definite matrix
        A = np.array([[4.0, 2.0], [2.0, 3.0]])
        b = np.array([8.0, 7.0])

        result = pure_math.solve(A, b, assume_a='pos')

        # Verify: A @ result should equal b
        check = A @ result
        for i in range(2):
            assert abs(check[i] - b[i]) < 1e-8

    def test_solve_matches_scipy_with_assume_a(self):
        """solve with assume_a should match SciPy."""
        A = np.array([[5.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 3.0]])
        b = np.array([10.0, 12.0, 10.0])

        result_pure = pure_math.solve(A, b, assume_a='pos')
        result_scipy = scipy_math.solve(A, b, assume_a='pos')

        for i in range(3):
            assert abs(result_pure[i] - result_scipy[i]) < 1e-8


# =============================================================================
# Test: Cholesky-based operations work with pure backend
# =============================================================================

class TestCholeskyIntegration:
    """Integration tests for Cholesky-based operations."""

    def test_cholesky_cho_solve_roundtrip(self):
        """Test cholesky + cho_solve produces correct solution."""
        # Create positive definite matrix
        A = np.array([[4.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 6.0]])
        b = np.array([7.0, 9.0, 9.0])

        L = pure_math.cholesky(A, lower=True)
        x = pure_math.cho_solve((L, True), b)

        # Verify: A @ x should equal b
        check = A @ x
        for i in range(3):
            assert abs(check[i] - b[i]) < 1e-10

    def test_cholesky_decomposition_property(self):
        """L @ L.T should equal original matrix."""
        A = np.array([[4.0, 2.0], [2.0, 5.0]])
        L = pure_math.cholesky(A, lower=True)

        # Reconstruct A from L @ L.T
        reconstructed = L @ L.T

        for i in range(2):
            for j in range(2):
                assert abs(reconstructed[i, j] - A[i, j]) < 1e-10


# =============================================================================
# Test: Eye and identity operations
# =============================================================================

class TestEye:
    """Tests for eye (identity matrix) function."""

    def test_eye_square(self):
        """eye(n) should create n x n identity matrix."""
        result = pure_array.eye(3)
        expected = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        assert result.shape == (3, 3)
        for i in range(3):
            for j in range(3):
                assert result[i, j] == expected[i][j]

    def test_eye_rectangular(self):
        """eye(n, m) should create n x m identity matrix."""
        result = pure_array.eye(2, 4)
        expected = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        assert result.shape == (2, 4)
        for i in range(2):
            for j in range(4):
                assert result[i, j] == expected[i][j]

    def test_eye_matches_numpy(self):
        """Pure Python eye should match NumPy."""
        for n in [2, 3, 5]:
            result_pure = pure_array.eye(n)
            result_np = np.eye(n)
            for i in range(n):
                for j in range(n):
                    assert result_pure[i, j] == result_np[i, j]
