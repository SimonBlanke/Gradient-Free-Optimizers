"""Tests for the _dimension_types module.

Tests cover:
- DimensionType enum values
- classify_search_space_value() function
- DimensionInfo dataclass validation
- DimensionMasks properties and factory methods
- is_legacy_search_space() detection
"""

import numpy as np
import pytest

from gradient_free_optimizers._dimension_types import (
    DimensionInfo,
    DimensionMasks,
    DimensionType,
    classify_search_space_value,
    is_legacy_search_space,
)

# =============================================================================
# DimensionType enum tests
# =============================================================================


class TestDimensionType:
    """Tests for the DimensionType enum."""

    def test_enum_values_exist(self):
        """DimensionType should have all three expected values."""
        assert hasattr(DimensionType, "DISCRETE_NUMERICAL")
        assert hasattr(DimensionType, "CONTINUOUS")
        assert hasattr(DimensionType, "CATEGORICAL")

    def test_enum_values_are_distinct(self):
        """Each DimensionType value should be unique."""
        values = [
            DimensionType.DISCRETE_NUMERICAL,
            DimensionType.CONTINUOUS,
            DimensionType.CATEGORICAL,
        ]
        assert len(values) == len(set(values))

    def test_enum_string_values(self):
        """DimensionType values should have descriptive string values."""
        assert DimensionType.DISCRETE_NUMERICAL.value == "discrete_numerical"
        assert DimensionType.CONTINUOUS.value == "continuous"
        assert DimensionType.CATEGORICAL.value == "categorical"


# =============================================================================
# classify_search_space_value() tests
# =============================================================================


class TestClassifySearchSpaceValue:
    """Tests for the classify_search_space_value function."""

    def test_numpy_array_is_discrete_numerical(self):
        """NumPy arrays should be classified as DISCRETE_NUMERICAL."""
        value = np.array([1, 2, 3, 4, 5])
        assert classify_search_space_value(value) == DimensionType.DISCRETE_NUMERICAL

    def test_numpy_linspace_is_discrete_numerical(self):
        """np.linspace result should be classified as DISCRETE_NUMERICAL."""
        value = np.linspace(0, 1, 100)
        assert classify_search_space_value(value) == DimensionType.DISCRETE_NUMERICAL

    def test_numpy_arange_is_discrete_numerical(self):
        """np.arange result should be classified as DISCRETE_NUMERICAL."""
        value = np.arange(-5, 5, 0.1)
        assert classify_search_space_value(value) == DimensionType.DISCRETE_NUMERICAL

    def test_tuple_two_elements_is_continuous(self):
        """Tuple with 2 elements should be classified as CONTINUOUS."""
        value = (0.0, 1.0)
        assert classify_search_space_value(value) == DimensionType.CONTINUOUS

    def test_tuple_floats_is_continuous(self):
        """Tuple of floats (min, max) should be CONTINUOUS."""
        value = (-5.0, 5.0)
        assert classify_search_space_value(value) == DimensionType.CONTINUOUS

    def test_tuple_ints_is_continuous(self):
        """Tuple of ints (min, max) should also be CONTINUOUS."""
        value = (0, 100)
        assert classify_search_space_value(value) == DimensionType.CONTINUOUS

    def test_tuple_three_elements_is_categorical(self):
        """Tuple with 3+ elements should be classified as CATEGORICAL."""
        value = (1, 2, 3)
        assert classify_search_space_value(value) == DimensionType.CATEGORICAL

    def test_list_strings_is_categorical(self):
        """List of strings should be classified as CATEGORICAL."""
        value = ["adam", "sgd", "rmsprop"]
        assert classify_search_space_value(value) == DimensionType.CATEGORICAL

    def test_list_numbers_is_categorical(self):
        """List of numbers should be classified as CATEGORICAL."""
        value = [16, 32, 64, 128]
        assert classify_search_space_value(value) == DimensionType.CATEGORICAL

    def test_list_booleans_is_categorical(self):
        """List of booleans should be classified as CATEGORICAL."""
        value = [True, False]
        assert classify_search_space_value(value) == DimensionType.CATEGORICAL

    def test_list_mixed_types_is_categorical(self):
        """List of mixed types should be classified as CATEGORICAL."""
        value = ["option1", 42, None]
        assert classify_search_space_value(value) == DimensionType.CATEGORICAL

    def test_empty_list_is_categorical(self):
        """Empty list should be classified as CATEGORICAL."""
        value = []
        assert classify_search_space_value(value) == DimensionType.CATEGORICAL


# =============================================================================
# DimensionInfo dataclass tests
# =============================================================================


class TestDimensionInfo:
    """Tests for the DimensionInfo dataclass."""

    def test_discrete_numerical_valid(self):
        """Valid discrete numerical DimensionInfo should be created."""
        info = DimensionInfo(
            name="x",
            dim_type=DimensionType.DISCRETE_NUMERICAL,
            bounds=(0, 99),
            values=[0.0, 0.1, 0.2],
            size=100,
        )
        assert info.name == "x"
        assert info.dim_type == DimensionType.DISCRETE_NUMERICAL
        assert info.bounds == (0, 99)
        assert info.size == 100

    def test_continuous_valid(self):
        """Valid continuous DimensionInfo should be created."""
        info = DimensionInfo(
            name="lr",
            dim_type=DimensionType.CONTINUOUS,
            bounds=(0.0001, 0.1),
            values=None,
            size=None,
        )
        assert info.name == "lr"
        assert info.dim_type == DimensionType.CONTINUOUS
        assert info.bounds == (0.0001, 0.1)
        assert info.values is None
        assert info.size is None

    def test_categorical_valid(self):
        """Valid categorical DimensionInfo should be created."""
        info = DimensionInfo(
            name="optimizer",
            dim_type=DimensionType.CATEGORICAL,
            bounds=(0, 2),
            values=["adam", "sgd", "rmsprop"],
            size=3,
        )
        assert info.name == "optimizer"
        assert info.dim_type == DimensionType.CATEGORICAL
        assert info.values == ["adam", "sgd", "rmsprop"]
        assert info.size == 3

    def test_continuous_with_values_raises(self):
        """Continuous dimension with values should raise ValueError."""
        with pytest.raises(ValueError, match="should not have values"):
            DimensionInfo(
                name="lr",
                dim_type=DimensionType.CONTINUOUS,
                bounds=(0.0, 1.0),
                values=[0.0, 0.5, 1.0],  # Invalid for continuous
                size=None,
            )

    def test_continuous_with_size_raises(self):
        """Continuous dimension with size should raise ValueError."""
        with pytest.raises(ValueError, match="should not have size"):
            DimensionInfo(
                name="lr",
                dim_type=DimensionType.CONTINUOUS,
                bounds=(0.0, 1.0),
                values=None,
                size=100,  # Invalid for continuous
            )

    def test_discrete_without_values_raises(self):
        """Discrete dimension without values should raise ValueError."""
        with pytest.raises(ValueError, match="must have values"):
            DimensionInfo(
                name="x",
                dim_type=DimensionType.DISCRETE_NUMERICAL,
                bounds=(0, 99),
                values=None,  # Invalid for discrete
                size=100,
            )

    def test_discrete_without_size_raises(self):
        """Discrete dimension without size should raise ValueError."""
        with pytest.raises(ValueError, match="must have size"):
            DimensionInfo(
                name="x",
                dim_type=DimensionType.DISCRETE_NUMERICAL,
                bounds=(0, 99),
                values=[1, 2, 3],
                size=None,  # Invalid for discrete
            )

    def test_categorical_without_values_raises(self):
        """Categorical dimension without values should raise ValueError."""
        with pytest.raises(ValueError, match="must have values"):
            DimensionInfo(
                name="algo",
                dim_type=DimensionType.CATEGORICAL,
                bounds=(0, 2),
                values=None,  # Invalid for categorical
                size=3,
            )


# =============================================================================
# DimensionMasks tests
# =============================================================================


class TestDimensionMasks:
    """Tests for the DimensionMasks dataclass."""

    def test_from_dim_types_discrete_only(self):
        """DimensionMasks from all discrete types should work."""
        dim_types = [
            DimensionType.DISCRETE_NUMERICAL,
            DimensionType.DISCRETE_NUMERICAL,
            DimensionType.DISCRETE_NUMERICAL,
        ]
        masks = DimensionMasks.from_dim_types(dim_types)

        assert masks.discrete_numerical == [0, 1, 2]
        assert masks.continuous == []
        assert masks.categorical == []

    def test_from_dim_types_continuous_only(self):
        """DimensionMasks from all continuous types should work."""
        dim_types = [DimensionType.CONTINUOUS, DimensionType.CONTINUOUS]
        masks = DimensionMasks.from_dim_types(dim_types)

        assert masks.discrete_numerical == []
        assert masks.continuous == [0, 1]
        assert masks.categorical == []

    def test_from_dim_types_categorical_only(self):
        """DimensionMasks from all categorical types should work."""
        dim_types = [DimensionType.CATEGORICAL, DimensionType.CATEGORICAL]
        masks = DimensionMasks.from_dim_types(dim_types)

        assert masks.discrete_numerical == []
        assert masks.continuous == []
        assert masks.categorical == [0, 1]

    def test_from_dim_types_mixed(self):
        """DimensionMasks from mixed types should correctly classify indices."""
        dim_types = [
            DimensionType.DISCRETE_NUMERICAL,  # index 0
            DimensionType.CONTINUOUS,  # index 1
            DimensionType.CATEGORICAL,  # index 2
            DimensionType.DISCRETE_NUMERICAL,  # index 3
        ]
        masks = DimensionMasks.from_dim_types(dim_types)

        assert masks.discrete_numerical == [0, 3]
        assert masks.continuous == [1]
        assert masks.categorical == [2]

    def test_has_discrete_numerical_true(self):
        """has_discrete_numerical should be True when discrete dims exist."""
        masks = DimensionMasks(discrete_numerical=[0], continuous=[], categorical=[])
        assert masks.has_discrete_numerical is True

    def test_has_discrete_numerical_false(self):
        """has_discrete_numerical should be False when no discrete dims."""
        masks = DimensionMasks(discrete_numerical=[], continuous=[0], categorical=[])
        assert masks.has_discrete_numerical is False

    def test_has_continuous_true(self):
        """has_continuous should be True when continuous dims exist."""
        masks = DimensionMasks(discrete_numerical=[], continuous=[0], categorical=[])
        assert masks.has_continuous is True

    def test_has_continuous_false(self):
        """has_continuous should be False when no continuous dims."""
        masks = DimensionMasks(discrete_numerical=[0], continuous=[], categorical=[])
        assert masks.has_continuous is False

    def test_has_categorical_true(self):
        """has_categorical should be True when categorical dims exist."""
        masks = DimensionMasks(discrete_numerical=[], continuous=[], categorical=[0])
        assert masks.has_categorical is True

    def test_has_categorical_false(self):
        """has_categorical should be False when no categorical dims."""
        masks = DimensionMasks(discrete_numerical=[0], continuous=[], categorical=[])
        assert masks.has_categorical is False

    def test_is_homogeneous_discrete(self):
        """is_homogeneous_discrete should be True for all-discrete spaces."""
        masks = DimensionMasks(
            discrete_numerical=[0, 1, 2], continuous=[], categorical=[]
        )
        assert masks.is_homogeneous_discrete is True

    def test_is_homogeneous_discrete_false_when_mixed(self):
        """is_homogeneous_discrete should be False for mixed spaces."""
        masks = DimensionMasks(discrete_numerical=[0], continuous=[1], categorical=[])
        assert masks.is_homogeneous_discrete is False

    def test_is_homogeneous_continuous(self):
        """is_homogeneous_continuous should be True for all-continuous spaces."""
        masks = DimensionMasks(discrete_numerical=[], continuous=[0, 1], categorical=[])
        assert masks.is_homogeneous_continuous is True

    def test_is_homogeneous_categorical(self):
        """is_homogeneous_categorical should be True for all-categorical spaces."""
        masks = DimensionMasks(discrete_numerical=[], continuous=[], categorical=[0, 1])
        assert masks.is_homogeneous_categorical is True

    def test_is_mixed_two_types(self):
        """is_mixed should be True when 2 types are present."""
        masks = DimensionMasks(discrete_numerical=[0], continuous=[1], categorical=[])
        assert masks.is_mixed is True

    def test_is_mixed_three_types(self):
        """is_mixed should be True when all 3 types are present."""
        masks = DimensionMasks(discrete_numerical=[0], continuous=[1], categorical=[2])
        assert masks.is_mixed is True

    def test_is_mixed_false_single_type(self):
        """is_mixed should be False for single type."""
        masks = DimensionMasks(
            discrete_numerical=[0, 1, 2], continuous=[], categorical=[]
        )
        assert masks.is_mixed is False

    def test_total_dimensions(self):
        """total_dimensions should return sum of all dimension counts."""
        masks = DimensionMasks(
            discrete_numerical=[0, 1], continuous=[2], categorical=[3, 4]
        )
        assert masks.total_dimensions == 5

    def test_total_dimensions_empty(self):
        """total_dimensions should be 0 for empty masks."""
        masks = DimensionMasks(discrete_numerical=[], continuous=[], categorical=[])
        assert masks.total_dimensions == 0


# =============================================================================
# is_legacy_search_space() tests
# =============================================================================


class TestIsLegacySearchSpace:
    """Tests for the is_legacy_search_space function."""

    def test_all_numpy_arrays_is_legacy(self):
        """Search space with only NumPy arrays should be legacy."""
        search_space = {
            "x": np.linspace(-5, 5, 100),
            "y": np.arange(0, 10, 1),
        }
        assert is_legacy_search_space(search_space) is True

    def test_with_continuous_is_not_legacy(self):
        """Search space with continuous dimension is not legacy."""
        search_space = {
            "x": np.linspace(-5, 5, 100),
            "y": (0.0, 10.0),  # Continuous
        }
        assert is_legacy_search_space(search_space) is False

    def test_with_categorical_is_not_legacy(self):
        """Search space with categorical dimension is not legacy."""
        search_space = {
            "x": np.linspace(-5, 5, 100),
            "algo": ["adam", "sgd"],  # Categorical
        }
        assert is_legacy_search_space(search_space) is False

    def test_only_continuous_is_not_legacy(self):
        """Search space with only continuous dimensions is not legacy."""
        search_space = {
            "x": (0.0, 1.0),
            "y": (-5.0, 5.0),
        }
        assert is_legacy_search_space(search_space) is False

    def test_only_categorical_is_not_legacy(self):
        """Search space with only categorical dimensions is not legacy."""
        search_space = {
            "algo": ["adam", "sgd"],
            "use_bias": [True, False],
        }
        assert is_legacy_search_space(search_space) is False

    def test_mixed_all_types_is_not_legacy(self):
        """Search space with all three types is not legacy."""
        search_space = {
            "x": np.linspace(-5, 5, 100),
            "y": (0.0, 1.0),
            "algo": ["adam", "sgd"],
        }
        assert is_legacy_search_space(search_space) is False

    def test_empty_search_space_is_legacy(self):
        """Empty search space should be considered legacy."""
        search_space = {}
        assert is_legacy_search_space(search_space) is True
