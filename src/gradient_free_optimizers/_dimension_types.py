"""Dimension type definitions for extended search space support.

This module provides the foundation for supporting different dimension types
in the search space:
- DISCRETE_NUMERICAL: NumPy arrays or GFOArrays with discrete numeric values
- CONTINUOUS: Tuples (min, max) representing continuous ranges
- CATEGORICAL: Python lists representing categorical choices

These types enable gradient-free optimizers to handle mixed search spaces
while maintaining backward compatibility with existing discrete-only spaces.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class DimensionType(Enum):
    """Classification of dimension types in the search space.

    Attributes
    ----------
        DISCRETE_NUMERICAL: A dimension with discrete numeric values,
            typically provided as a NumPy array or GFOArray.
            Example: np.linspace(0, 1, 100)

        CONTINUOUS: A dimension with a continuous range of values,
            provided as a tuple (min, max).
            Example: (0.0, 1.0)

        CATEGORICAL: A dimension with categorical choices,
            provided as a Python list.
            Example: ["adam", "sgd", "rmsprop"]
    """

    DISCRETE_NUMERICAL = "discrete_numerical"
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


@dataclass
class DimensionInfo:
    """Information about a single dimension in the search space.

    Attributes
    ----------
        name: The parameter name (key in search space dict)
        dim_type: The type of this dimension
        bounds: (min, max) tuple - for discrete/categorical: (0, n-1),
                for continuous: (min_value, max_value)
        values: For discrete/categorical: list of possible values;
                for continuous: None
        size: Number of discrete values; for continuous: None
    """

    name: str
    dim_type: DimensionType
    bounds: tuple
    values: list[Any] | None
    size: int | None

    def __post_init__(self):
        """Validate the dimension info after initialization."""
        if self.dim_type == DimensionType.CONTINUOUS:
            if self.values is not None:
                raise ValueError(
                    f"Continuous dimension '{self.name}' should not have values"
                )
            if self.size is not None:
                raise ValueError(
                    f"Continuous dimension '{self.name}' should not have size"
                )
        else:
            if self.values is None:
                raise ValueError(
                    f"Non-continuous dimension '{self.name}' must have values"
                )
            if self.size is None:
                raise ValueError(
                    f"Non-continuous dimension '{self.name}' must have size"
                )


@dataclass
class DimensionMasks:
    """Index masks for fast access to dimensions by type.

    Enables vectorized operations on all dimensions of the same type
    simultaneously, which is critical for performance with large
    search spaces (e.g., 100M+ dimensions).

    Attributes
    ----------
        discrete_numerical: List of indices for discrete-numerical dimensions
        continuous: List of indices for continuous dimensions
        categorical: List of indices for categorical dimensions
    """

    discrete_numerical: list[int]
    continuous: list[int]
    categorical: list[int]

    @property
    def has_discrete_numerical(self) -> bool:
        """Check if there are any discrete-numerical dimensions."""
        return len(self.discrete_numerical) > 0

    @property
    def has_continuous(self) -> bool:
        """Check if there are any continuous dimensions."""
        return len(self.continuous) > 0

    @property
    def has_categorical(self) -> bool:
        """Check if there are any categorical dimensions."""
        return len(self.categorical) > 0

    @property
    def is_homogeneous_discrete(self) -> bool:
        """Check if all dimensions are discrete-numerical (legacy mode).

        When True, the optimizer can use the original code paths
        for full backward compatibility.
        """
        return (
            self.has_discrete_numerical
            and not self.has_continuous
            and not self.has_categorical
        )

    @property
    def is_homogeneous_continuous(self) -> bool:
        """Check if all dimensions are continuous."""
        return (
            self.has_continuous
            and not self.has_discrete_numerical
            and not self.has_categorical
        )

    @property
    def is_homogeneous_categorical(self) -> bool:
        """Check if all dimensions are categorical."""
        return (
            self.has_categorical
            and not self.has_discrete_numerical
            and not self.has_continuous
        )

    @property
    def is_mixed(self) -> bool:
        """Check if the search space has mixed dimension types."""
        type_count = sum(
            [self.has_discrete_numerical, self.has_continuous, self.has_categorical]
        )
        return type_count > 1

    @property
    def total_dimensions(self) -> int:
        """Total number of dimensions across all types."""
        return (
            len(self.discrete_numerical) + len(self.continuous) + len(self.categorical)
        )

    @classmethod
    def from_dim_types(cls, dim_types: list[DimensionType]) -> "DimensionMasks":
        """Create DimensionMasks from a list of dimension types.

        Args:
            dim_types: List of DimensionType for each dimension

        Returns
        -------
            DimensionMasks with indices grouped by type
        """
        return cls(
            discrete_numerical=[
                i
                for i, t in enumerate(dim_types)
                if t == DimensionType.DISCRETE_NUMERICAL
            ],
            continuous=[
                i for i, t in enumerate(dim_types) if t == DimensionType.CONTINUOUS
            ],
            categorical=[
                i for i, t in enumerate(dim_types) if t == DimensionType.CATEGORICAL
            ],
        )


def classify_search_space_value(value: Any) -> DimensionType:
    """Classify a single search space value by its Python type.

    This function determines the dimension type based on how the
    value is provided in the search space dictionary:

    - Tuple with 2 elements -> CONTINUOUS (min, max)
    - Python list -> CATEGORICAL
    - Everything else (array-like) -> DISCRETE_NUMERICAL

    Args:
        value: A value from the search space dictionary

    Returns
    -------
        The appropriate DimensionType

    Examples
    --------
        >>> classify_search_space_value((0.0, 1.0))
        DimensionType.CONTINUOUS

        >>> classify_search_space_value(["adam", "sgd"])
        DimensionType.CATEGORICAL

        >>> classify_search_space_value(np.linspace(0, 1, 100))
        DimensionType.DISCRETE_NUMERICAL
    """
    if isinstance(value, tuple):
        if len(value) == 2:
            # Tuple with exactly 2 elements -> continuous (min, max)
            return DimensionType.CONTINUOUS
        else:
            # Tuple with != 2 elements treated as categorical
            return DimensionType.CATEGORICAL
    elif isinstance(value, list):
        # Python list -> categorical
        return DimensionType.CATEGORICAL
    else:
        # Array-like (NumPy array, GFOArray, etc.) -> discrete numerical
        return DimensionType.DISCRETE_NUMERICAL


def is_legacy_search_space(search_space: dict) -> bool:
    """Check if a search space contains only discrete-numerical dimensions.

    This is used to determine if the optimizer should use legacy code paths
    for full backward compatibility.

    Args:
        search_space: Dictionary mapping parameter names to their domains

    Returns
    -------
        True if all dimensions are discrete-numerical
    """
    for value in search_space.values():
        dim_type = classify_search_space_value(value)
        if dim_type != DimensionType.DISCRETE_NUMERICAL:
            return False
    return True
