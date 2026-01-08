"""
Tests for the GFOArray class (pure Python array implementation).

These tests ensure the GFOArray class works correctly as a NumPy replacement.
"""

import pytest

from gradient_free_optimizers._array_backend import _pure as pure_backend

from .conftest import to_list


class TestGFOArrayArithmetic:
    """Test GFOArray arithmetic operations."""

    def test_add(self):
        a = pure_backend.array([1, 2, 3])
        b = pure_backend.array([4, 5, 6])
        result = a + b
        assert to_list(result) == [5, 7, 9]

    def test_sub(self):
        a = pure_backend.array([4, 5, 6])
        b = pure_backend.array([1, 2, 3])
        result = a - b
        assert to_list(result) == [3, 3, 3]

    def test_mul(self):
        a = pure_backend.array([1, 2, 3])
        b = pure_backend.array([4, 5, 6])
        result = a * b
        assert to_list(result) == [4, 10, 18]

    def test_div(self):
        a = pure_backend.array([4, 6, 8])
        b = pure_backend.array([2, 2, 2])
        result = a / b
        assert to_list(result) == [2, 3, 4]

    def test_scalar_add(self):
        a = pure_backend.array([1, 2, 3])
        assert to_list(a + 10) == [11, 12, 13]

    def test_scalar_mul(self):
        a = pure_backend.array([1, 2, 3])
        assert to_list(a * 2) == [2, 4, 6]

    def test_scalar_sub(self):
        a = pure_backend.array([1, 2, 3])
        assert to_list(a - 1) == [0, 1, 2]

    def test_negation(self):
        a = pure_backend.array([1, -2, 3])
        result = -a
        assert to_list(result) == [-1, 2, -3]


class TestGFOArrayComparison:
    """Test GFOArray comparison operations."""

    def test_less_than(self):
        a = pure_backend.array([1, 2, 3])
        b = pure_backend.array([2, 2, 2])
        assert to_list(a < b) == [True, False, False]

    def test_less_equal(self):
        a = pure_backend.array([1, 2, 3])
        b = pure_backend.array([2, 2, 2])
        assert to_list(a <= b) == [True, True, False]

    def test_greater_than(self):
        a = pure_backend.array([1, 2, 3])
        b = pure_backend.array([2, 2, 2])
        assert to_list(a > b) == [False, False, True]

    def test_greater_equal(self):
        a = pure_backend.array([1, 2, 3])
        b = pure_backend.array([2, 2, 2])
        assert to_list(a >= b) == [False, True, True]

    def test_equal(self):
        a = pure_backend.array([1, 2, 3])
        b = pure_backend.array([2, 2, 2])
        assert to_list(a == b) == [False, True, False]


class TestGFOArrayIndexing:
    """Test GFOArray indexing and slicing."""

    def test_getitem_single(self):
        a = pure_backend.array([10, 20, 30, 40, 50])
        assert a[0] == 10
        assert a[2] == 30
        assert a[-1] == 50

    def test_getitem_slice(self):
        a = pure_backend.array([10, 20, 30, 40, 50])
        result = a[1:4]
        assert to_list(result) == [20, 30, 40]

    def test_iteration(self):
        a = pure_backend.array([1, 2, 3])
        values = list(a)
        assert values == [1, 2, 3]

    def test_len(self):
        a = pure_backend.array([1, 2, 3, 4, 5])
        assert len(a) == 5


class TestGFOArrayMethods:
    """Test GFOArray instance methods."""

    def test_copy(self):
        a = pure_backend.array([1, 2, 3])
        b = a.copy()
        b[0] = 100
        assert a[0] == 1  # Original unchanged
        assert b[0] == 100

    def test_tolist(self):
        a = pure_backend.array([1, 2, 3])
        result = a.tolist()
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_astype_to_int(self):
        a = pure_backend.array([1.5, 2.7, 3.9])
        b = a.astype(int)
        assert to_list(b) == [1, 2, 3]

    def test_astype_to_float(self):
        a = pure_backend.array([1, 2, 3])
        b = a.astype(float)
        assert to_list(b) == [1.0, 2.0, 3.0]

    def test_flatten(self):
        a = pure_backend.array([[1, 2], [3, 4]])
        result = a.flatten()
        assert to_list(result) == [1, 2, 3, 4]

    def test_reshape(self):
        a = pure_backend.array([1, 2, 3, 4, 5, 6])
        b = a.reshape((2, 3))
        assert b.shape == (2, 3)

    def test_transpose(self):
        a = pure_backend.array([[1, 2, 3], [4, 5, 6]])
        b = a.T
        assert b.shape == (3, 2)


class TestGFOArrayAggregation:
    """Test GFOArray aggregation methods."""

    def test_sum(self):
        a = pure_backend.array([1, 2, 3, 4, 5])
        assert a.sum() == 15

    def test_mean(self):
        a = pure_backend.array([1, 2, 3, 4, 5])
        assert a.mean() == 3.0

    def test_min(self):
        a = pure_backend.array([3, 1, 4, 1, 5])
        assert a.min() == 1

    def test_max(self):
        a = pure_backend.array([3, 1, 4, 1, 5])
        assert a.max() == 5

    def test_argmax(self):
        a = pure_backend.array([3, 1, 5, 2, 4])
        assert a.argmax() == 2

    def test_argmin(self):
        a = pure_backend.array([3, 1, 5, 2, 4])
        assert a.argmin() == 1


class TestGFOArray2D:
    """Test GFOArray 2D operations."""

    def test_2d_creation(self):
        a = pure_backend.array([[1, 2, 3], [4, 5, 6]])
        assert a.shape == (2, 3)
        assert a.ndim == 2

    def test_2d_indexing(self):
        a = pure_backend.array([[1, 2, 3], [4, 5, 6]])
        assert a[0, 0] == 1
        assert a[1, 2] == 6

    def test_2d_sum(self):
        a = pure_backend.array([[1, 2, 3], [4, 5, 6]])
        assert a.sum() == 21


class TestGFOArrayIntegration:
    """Integration tests simulating typical GFO workflows."""

    def test_position_manipulation(self):
        """Simulate position clipping and rounding in optimization."""
        pos = pure_backend.array([5.7, 3.2, 7.9])
        max_pos = pure_backend.array([10, 10, 10])

        # Clip and round
        clipped = pure_backend.clip(pos, 0, max_pos)
        rounded = pure_backend.rint(clipped)

        assert to_list(rounded) == [6, 3, 8]

    def test_score_tracking(self):
        """Simulate finding best score in optimization."""
        scores = pure_backend.array([0.5, 0.8, 0.3, 0.9, 0.1])
        best_idx = pure_backend.argmax(scores)
        best_score = scores[best_idx]

        assert best_idx == 3
        assert best_score == 0.9

    def test_random_position_generation(self):
        """Simulate random position generation."""
        pure_backend.random.seed(42)
        max_positions = [10, 20, 30]

        positions = []
        for max_p in max_positions:
            pos = pure_backend.random.randint(0, max_p + 1)
            positions.append(pos)
            assert 0 <= pos <= max_p

        assert len(positions) == 3
