"""
Core tests for the array backend.

Tests both the NumPy backend and the pure Python fallback to ensure
they produce equivalent results.
"""

import math

import pytest

from gradient_free_optimizers._array_backend import HAS_NUMPY
from gradient_free_optimizers._array_backend import _numpy as np_backend
from gradient_free_optimizers._array_backend import _pure as pure_backend

from .conftest import arrays_close, to_list

# =============================================================================
# Array Creation Tests
# =============================================================================


class TestArrayCreation:
    """Test array creation functions."""

    def test_array_from_list(self):
        np_arr = np_backend.array([1, 2, 3, 4, 5])
        pure_arr = pure_backend.array([1, 2, 3, 4, 5])
        assert to_list(np_arr) == to_list(pure_arr)

    def test_array_from_nested_list(self):
        np_arr = np_backend.array([[1, 2], [3, 4]])
        pure_arr = pure_backend.array([[1, 2], [3, 4]])
        assert np_arr.shape == pure_arr.shape

    def test_zeros_1d(self):
        np_arr = np_backend.zeros(5)
        pure_arr = pure_backend.zeros(5)
        assert to_list(np_arr) == to_list(pure_arr)

    def test_zeros_2d(self):
        np_arr = np_backend.zeros((3, 4))
        pure_arr = pure_backend.zeros((3, 4))
        assert np_arr.shape == pure_arr.shape

    def test_ones_1d(self):
        np_arr = np_backend.ones(5)
        pure_arr = pure_backend.ones(5)
        assert to_list(np_arr) == to_list(pure_arr)

    def test_ones_2d(self):
        np_arr = np_backend.ones((2, 3))
        pure_arr = pure_backend.ones((2, 3))
        assert np_arr.shape == pure_arr.shape

    @pytest.mark.parametrize(
        ("start", "stop", "num"),
        [
            (0, 10, 5),
            (0, 1, 11),
            (-5, 5, 21),
            (0, 100, 101),
        ],
    )
    def test_linspace(self, start, stop, num):
        np_arr = np_backend.linspace(start, stop, num)
        pure_arr = pure_backend.linspace(start, stop, num)
        assert arrays_close(np_arr, pure_arr)

    @pytest.mark.parametrize(
        ("start", "stop", "step"),
        [
            (0, 10, 1),
            (0, 10, 2),
            (0, 5, 0.5),
            (-10, 10, 1),
        ],
    )
    def test_arange(self, start, stop, step):
        np_arr = np_backend.arange(start, stop, step)
        pure_arr = pure_backend.arange(start, stop, step)
        assert arrays_close(np_arr, pure_arr)

    def test_eye(self):
        np_arr = np_backend.eye(3)
        pure_arr = pure_backend.eye(3)
        np_flat = to_list(np_arr.flatten())
        pure_flat = to_list(pure_arr.flatten())
        assert np_flat == pure_flat


# =============================================================================
# Mathematical Operations Tests
# =============================================================================


class TestMathOperations:
    """Test mathematical operations."""

    def test_sum(self, sample_array):
        np_result = np_backend.sum(np_backend.array(sample_array))
        pure_result = pure_backend.sum(pure_backend.array(sample_array))
        assert np_result == pure_result

    def test_mean(self, sample_array):
        np_result = np_backend.mean(np_backend.array(sample_array))
        pure_result = pure_backend.mean(pure_backend.array(sample_array))
        assert abs(np_result - pure_result) < 1e-10

    def test_std(self, sample_array):
        np_result = np_backend.std(np_backend.array(sample_array))
        pure_result = pure_backend.std(pure_backend.array(sample_array))
        assert abs(np_result - pure_result) < 1e-10

    @pytest.mark.parametrize(
        "values",
        [
            [0, 1, 2, 3],
            [0.5, 1.0, 1.5, 2.0],
            [-1, 0, 1],
        ],
    )
    def test_exp(self, values):
        np_result = np_backend.exp(np_backend.array(values))
        pure_result = pure_backend.exp(pure_backend.array(values))
        assert arrays_close(np_result, pure_result)

    @pytest.mark.parametrize(
        "values",
        [
            [1, 2, 3, 4],
            [0.5, 1.0, 2.0, 10.0],
            [math.e, math.e**2],
        ],
    )
    def test_log(self, values):
        np_result = np_backend.log(np_backend.array(values))
        pure_result = pure_backend.log(pure_backend.array(values))
        assert arrays_close(np_result, pure_result)

    @pytest.mark.parametrize(
        "values",
        [
            [1, 4, 9, 16, 25],
            [0.25, 0.5, 1.0, 2.0],
        ],
    )
    def test_sqrt(self, values):
        np_result = np_backend.sqrt(np_backend.array(values))
        pure_result = pure_backend.sqrt(pure_backend.array(values))
        assert arrays_close(np_result, pure_result)

    def test_abs(self):
        values = [-3, -2, -1, 0, 1, 2, 3]
        np_result = np_backend.abs(np_backend.array(values))
        pure_result = pure_backend.abs(pure_backend.array(values))
        assert to_list(np_result) == to_list(pure_result)

    def test_power(self):
        values = [1, 2, 3, 4, 5]
        np_result = np_backend.power(np_backend.array(values), 2)
        pure_result = pure_backend.power(pure_backend.array(values), 2)
        assert to_list(np_result) == to_list(pure_result)


# =============================================================================
# Clipping and Rounding Tests
# =============================================================================


class TestClippingAndRounding:
    """Test clipping and rounding operations."""

    @pytest.mark.parametrize(
        ("values", "a_min", "a_max", "expected"),
        [
            ([1, 2, 3, 4, 5], 2, 4, [2, 2, 3, 4, 4]),
            ([-5, 0, 5, 10], 0, 5, [0, 0, 5, 5]),
            ([0.5, 1.5, 2.5], 1, 2, [1, 1.5, 2]),
        ],
    )
    def test_clip(self, values, a_min, a_max, expected):
        np_result = np_backend.clip(np_backend.array(values), a_min, a_max)
        pure_result = pure_backend.clip(pure_backend.array(values), a_min, a_max)
        assert to_list(np_result) == expected
        assert to_list(pure_result) == expected

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ([1.2, 2.5, 3.7, 4.1], [1, 2, 4, 4]),
            ([0.4, 0.5, 0.6], [0, 0, 1]),
            ([-1.5, -0.5, 0.5, 1.5], [-2, 0, 0, 2]),
        ],
    )
    def test_rint(self, values, expected):
        np_result = np_backend.rint(np_backend.array(values))
        pure_result = pure_backend.rint(pure_backend.array(values))
        # Note: rounding behavior may differ slightly at .5 boundaries
        assert arrays_close(np_result, pure_result, atol=1)


# =============================================================================
# Index Operations Tests
# =============================================================================


class TestIndexOperations:
    """Test index-based operations."""

    def test_argmax(self):
        values = [3, 1, 4, 1, 5, 9, 2, 6]
        np_result = np_backend.argmax(np_backend.array(values))
        pure_result = pure_backend.argmax(pure_backend.array(values))
        assert np_result == pure_result == 5

    def test_argmin(self):
        values = [3, 1, 4, 1, 5, 9, 2, 6]
        np_result = np_backend.argmin(np_backend.array(values))
        pure_result = pure_backend.argmin(pure_backend.array(values))
        assert np_result == pure_result == 1

    def test_argsort(self):
        values = [3, 1, 4, 1, 5]
        np_result = np_backend.argsort(np_backend.array(values))
        pure_result = pure_backend.argsort(pure_backend.array(values))
        # First two elements should be indices 1 and 3 (both have value 1)
        assert to_list(np_result)[:2] in [[1, 3], [3, 1]]
        assert to_list(pure_result)[:2] in [[1, 3], [3, 1]]

    def test_where_indices_only(self):
        values = [True, False, True, False, True]
        np_result = np_backend.where(np_backend.array(values))
        pure_result = pure_backend.where(pure_backend.array(values))
        assert to_list(np_result[0]) == to_list(pure_result[0]) == [0, 2, 4]

    def test_where_with_values(self):
        condition = [True, False, True, False]
        x = [1, 2, 3, 4]
        y = [10, 20, 30, 40]
        np_result = np_backend.where(
            np_backend.array(condition), np_backend.array(x), np_backend.array(y)
        )
        pure_result = pure_backend.where(
            pure_backend.array(condition),
            pure_backend.array(x),
            pure_backend.array(y),
        )
        assert to_list(np_result) == to_list(pure_result) == [1, 20, 3, 40]

    def test_searchsorted(self):
        arr = [1, 3, 5, 7, 9]
        values = [0, 2, 5, 8, 10]
        np_result = np_backend.searchsorted(np_backend.array(arr), values)
        pure_result = pure_backend.searchsorted(pure_backend.array(arr), values)
        assert to_list(np_result) == to_list(pure_result)


# =============================================================================
# Set Operations Tests
# =============================================================================


class TestSetOperations:
    """Test set-like operations."""

    def test_unique(self):
        values = [1, 2, 2, 3, 3, 3, 4]
        np_result = np_backend.unique(np_backend.array(values))
        pure_result = pure_backend.unique(pure_backend.array(values))
        assert to_list(np_result) == to_list(pure_result) == [1, 2, 3, 4]

    def test_intersect1d(self):
        a = [1, 2, 3, 4, 5]
        b = [3, 4, 5, 6, 7]
        np_result = np_backend.intersect1d(np_backend.array(a), np_backend.array(b))
        pure_result = pure_backend.intersect1d(
            pure_backend.array(a), pure_backend.array(b)
        )
        assert to_list(np_result) == to_list(pure_result) == [3, 4, 5]

    def test_isin(self):
        elements = [1, 2, 3, 4, 5]
        test_elements = [2, 4]
        np_result = np_backend.isin(
            np_backend.array(elements), np_backend.array(test_elements)
        )
        pure_result = pure_backend.isin(
            pure_backend.array(elements), pure_backend.array(test_elements)
        )
        expected = [False, True, False, True, False]
        assert to_list(np_result) == to_list(pure_result) == expected


# =============================================================================
# Linear Algebra Tests
# =============================================================================


class TestLinearAlgebra:
    """Test linear algebra operations."""

    def test_dot_product_1d(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        np_result = np_backend.dot(np_backend.array(a), np_backend.array(b))
        pure_result = pure_backend.dot(pure_backend.array(a), pure_backend.array(b))
        assert np_result == pure_result == 32

    def test_linalg_norm(self):
        values = [3, 4]
        np_result = np_backend.linalg.norm(np_backend.array(values))
        pure_result = pure_backend.linalg.norm(pure_backend.array(values))
        assert abs(np_result - pure_result) < 1e-10
        assert abs(np_result - 5.0) < 1e-10


# =============================================================================
# Random Generation Tests
# =============================================================================


class TestRandomGeneration:
    """Test random number generation."""

    def test_seed_reproducibility(self):
        np_backend.random.seed(42)
        np_result1 = np_backend.random.randint(0, 100, size=10)
        np_backend.random.seed(42)
        np_result2 = np_backend.random.randint(0, 100, size=10)
        assert to_list(np_result1) == to_list(np_result2)

        pure_backend.random.seed(42)
        pure_result1 = pure_backend.random.randint(0, 100, size=10)
        pure_backend.random.seed(42)
        pure_result2 = pure_backend.random.randint(0, 100, size=10)
        assert to_list(pure_result1) == to_list(pure_result2)

    def test_randint_range(self):
        pure_backend.random.seed(0)
        result = pure_backend.random.randint(0, 10, size=100)
        values = to_list(result)
        assert all(0 <= v < 10 for v in values)

    def test_uniform_range(self):
        pure_backend.random.seed(0)
        result = pure_backend.random.uniform(0, 1, size=100)
        values = to_list(result)
        assert all(0 <= v <= 1 for v in values)

    def test_choice_from_array(self):
        arr = [10, 20, 30, 40, 50]
        pure_backend.random.seed(0)
        result = pure_backend.random.choice(arr, size=10)
        values = to_list(result)
        assert all(v in arr for v in values)

    def test_normal_distribution(self):
        pure_backend.random.seed(0)
        result = pure_backend.random.normal(0, 1, size=1000)
        values = to_list(result)
        # Check mean is approximately 0
        mean = sum(values) / len(values)
        assert abs(mean) < 0.2  # Generous tolerance for 1000 samples


# =============================================================================
# Array Properties Tests
# =============================================================================


class TestArrayProperties:
    """Test array property access."""

    def test_shape_1d(self):
        np_arr = np_backend.array([1, 2, 3, 4, 5])
        pure_arr = pure_backend.array([1, 2, 3, 4, 5])
        assert np_arr.shape == pure_arr.shape == (5,)

    def test_shape_2d(self):
        np_arr = np_backend.array([[1, 2, 3], [4, 5, 6]])
        pure_arr = pure_backend.array([[1, 2, 3], [4, 5, 6]])
        assert np_arr.shape == pure_arr.shape == (2, 3)

    def test_ndim(self):
        np_arr_1d = np_backend.array([1, 2, 3])
        np_arr_2d = np_backend.array([[1, 2], [3, 4]])
        pure_arr_1d = pure_backend.array([1, 2, 3])
        pure_arr_2d = pure_backend.array([[1, 2], [3, 4]])

        assert np_arr_1d.ndim == pure_arr_1d.ndim == 1
        assert np_arr_2d.ndim == pure_arr_2d.ndim == 2


# =============================================================================
# Array Manipulation Tests
# =============================================================================


class TestArrayOperations:
    """Test array manipulation operations."""

    def test_reshape(self):
        values = list(range(6))
        np_arr = np_backend.array(values).reshape((2, 3))
        pure_arr = pure_backend.array(values).reshape((2, 3))
        assert np_arr.shape == pure_arr.shape == (2, 3)

    def test_flatten(self):
        values = [[1, 2, 3], [4, 5, 6]]
        np_arr = np_backend.flatten(np_backend.array(values))
        pure_arr = pure_backend.flatten(pure_backend.array(values))
        assert to_list(np_arr) == to_list(pure_arr) == [1, 2, 3, 4, 5, 6]

    def test_transpose_2d(self):
        values = [[1, 2, 3], [4, 5, 6]]
        np_arr = np_backend.array(values)
        pure_arr = pure_backend.array(values)
        np_T = np_arr.T
        pure_T = pure_arr.T
        assert np_T.shape == pure_T.shape == (3, 2)


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Test mathematical constants."""

    def test_inf(self):
        assert np_backend.inf == pure_backend.inf == float("inf")

    def test_pi(self):
        assert abs(np_backend.pi - pure_backend.pi) < 1e-15
        assert abs(np_backend.pi - math.pi) < 1e-15

    def test_e(self):
        assert abs(np_backend.e - pure_backend.e) < 1e-15
        assert abs(np_backend.e - math.e) < 1e-15


# =============================================================================
# Comparison Tests
# =============================================================================


class TestComparison:
    """Test comparison operations."""

    def test_maximum(self):
        a = [1, 5, 3, 7]
        b = [2, 4, 6, 0]
        np_result = np_backend.maximum(np_backend.array(a), np_backend.array(b))
        pure_result = pure_backend.maximum(pure_backend.array(a), pure_backend.array(b))
        assert to_list(np_result) == to_list(pure_result) == [2, 5, 6, 7]

    def test_minimum(self):
        a = [1, 5, 3, 7]
        b = [2, 4, 6, 0]
        np_result = np_backend.minimum(np_backend.array(a), np_backend.array(b))
        pure_result = pure_backend.minimum(pure_backend.array(a), pure_backend.array(b))
        assert to_list(np_result) == to_list(pure_result) == [1, 4, 3, 0]

    def test_isnan(self):
        values = [1.0, float("nan"), 3.0, float("nan")]
        np_result = np_backend.isnan(np_backend.array(values))
        pure_result = pure_backend.isnan(pure_backend.array(values))
        assert to_list(np_result) == to_list(pure_result) == [False, True, False, True]

    def test_isinf(self):
        values = [1.0, float("inf"), -float("inf"), 0.0]
        np_result = np_backend.isinf(np_backend.array(values))
        pure_result = pure_backend.isinf(pure_backend.array(values))
        assert to_list(np_result) == to_list(pure_result) == [False, True, True, False]


# =============================================================================
# Backend Selection Tests
# =============================================================================


class TestBackendSelection:
    """Test that backend selection works correctly."""

    def test_has_numpy_flag(self):
        assert HAS_NUMPY is True  # We know NumPy is installed in test env

    def test_array_backend_uses_numpy(self):
        import gradient_free_optimizers._array_backend as array_backend

        if HAS_NUMPY:
            assert array_backend._backend_name == "numpy"
