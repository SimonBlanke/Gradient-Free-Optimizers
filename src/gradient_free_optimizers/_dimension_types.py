"""Dimension type definitions for extended search space support.

This module provides the foundation for supporting different dimension types
in the search space:
- DISCRETE_NUMERICAL: NumPy arrays or GFOArrays with discrete numeric values
- CONTINUOUS: Tuples (min, max) representing continuous ranges
- CATEGORICAL: Python lists representing categorical choices
- DISTRIBUTION: SciPy stats continuous distributions, optimized internally in
  quantile space

These types enable gradient-free optimizers to handle mixed search spaces
while maintaining backward compatibility with existing discrete-only spaces.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

DEFAULT_DISTRIBUTION_QUANTILES = (0.001, 0.999)


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

        DISTRIBUTION: A dimension backed by a SciPy stats continuous
            distribution. The optimizer operates on quantiles and the
            converter exposes distribution values to the objective function.
            Example: scipy.stats.norm(loc=0, scale=1)
    """

    DISCRETE_NUMERICAL = "discrete_numerical"
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    DISTRIBUTION = "distribution"


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
        distribution: For distribution dimensions: the SciPy distribution
                object; otherwise None
    """

    name: str
    dim_type: DimensionType
    bounds: tuple
    values: list[Any] | None
    size: int | None
    distribution: Any | None = None

    def __post_init__(self):
        """Validate the dimension info after initialization."""
        if self.dim_type in (DimensionType.CONTINUOUS, DimensionType.DISTRIBUTION):
            if self.values is not None:
                raise ValueError(
                    f"{self.dim_type.value.title()} dimension '{self.name}' "
                    "should not have values"
                )
            if self.size is not None:
                raise ValueError(
                    f"{self.dim_type.value.title()} dimension '{self.name}' "
                    "should not have size"
                )
            if (
                self.dim_type == DimensionType.DISTRIBUTION
                and self.distribution is None
            ):
                raise ValueError(
                    f"Distribution dimension '{self.name}' must have distribution"
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
        distribution: List of indices for distribution dimensions
    """

    discrete_numerical: list[int]
    continuous: list[int]
    categorical: list[int]
    distribution: list[int] = field(default_factory=list)

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
    def has_distribution(self) -> bool:
        """Check if there are any distribution dimensions."""
        return len(self.distribution) > 0

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
            and not self.has_distribution
        )

    @property
    def is_homogeneous_continuous(self) -> bool:
        """Check if all dimensions are continuous."""
        return (
            self.has_continuous
            and not self.has_discrete_numerical
            and not self.has_categorical
            and not self.has_distribution
        )

    @property
    def is_homogeneous_categorical(self) -> bool:
        """Check if all dimensions are categorical."""
        return (
            self.has_categorical
            and not self.has_discrete_numerical
            and not self.has_continuous
            and not self.has_distribution
        )

    @property
    def is_homogeneous_distribution(self) -> bool:
        """Check if all dimensions are distribution-backed."""
        return (
            self.has_distribution
            and not self.has_discrete_numerical
            and not self.has_continuous
            and not self.has_categorical
        )

    @property
    def is_mixed(self) -> bool:
        """Check if the search space has mixed dimension types."""
        type_count = sum(
            [
                self.has_discrete_numerical,
                self.has_continuous,
                self.has_categorical,
                self.has_distribution,
            ]
        )
        return type_count > 1

    @property
    def total_dimensions(self) -> int:
        """Total number of dimensions across all types."""
        return (
            len(self.discrete_numerical)
            + len(self.continuous)
            + len(self.categorical)
            + len(self.distribution)
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
            distribution=[
                i for i, t in enumerate(dim_types) if t == DimensionType.DISTRIBUTION
            ],
        )


def _as_float(value: Any) -> float:
    """Convert NumPy/scalar-like values to a Python float."""
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def is_scipy_distribution(value: Any) -> bool:
    """Return True for supported SciPy stats continuous distributions.

    SciPy is optional. This helper avoids importing SciPy unless the object
    already looks distribution-like, and it returns False when SciPy is absent.
    """
    ppf = getattr(value, "ppf", None)
    cdf = getattr(value, "cdf", None)
    if not (callable(ppf) and callable(cdf)):
        return False

    try:
        from scipy.stats import rv_continuous
    except ImportError:
        return False

    if isinstance(value, rv_continuous):
        return True

    # Frozen distributions (e.g. stats.norm(loc=0, scale=1)) expose the
    # underlying rv_continuous via .dist; discrete frozen objects have
    # rv_discrete there instead, so the isinstance check excludes them.
    dist = getattr(value, "dist", None)
    return isinstance(dist, rv_continuous)


def distribution_ppf(distribution: Any, quantile: float) -> float:
    """Evaluate a SciPy distribution ppf and return a Python float."""
    return _as_float(distribution.ppf(float(quantile)))


def distribution_cdf(distribution: Any, value: float) -> float:
    """Evaluate a SciPy distribution cdf and return a Python float."""
    return _as_float(distribution.cdf(float(value)))


def distribution_quantile_bounds(distribution: Any) -> tuple[float, float]:
    """Return internal quantile bounds for a SciPy distribution dimension.

    Finite-support distributions can safely use the full [0, 1] quantile
    range. Infinite-support distributions use effective tail cutoffs to keep
    all optimizer position bounds finite.
    """
    try:
        lower = distribution_ppf(distribution, 0.0)
        upper = distribution_ppf(distribution, 1.0)
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_DISTRIBUTION_QUANTILES

    if math.isfinite(lower) and math.isfinite(upper) and lower < upper:
        return (0.0, 1.0)

    q_low, q_high = DEFAULT_DISTRIBUTION_QUANTILES
    low_value = distribution_ppf(distribution, q_low)
    high_value = distribution_ppf(distribution, q_high)

    if not (
        math.isfinite(low_value)
        and math.isfinite(high_value)
        and low_value < high_value
    ):
        raise ValueError(
            "SciPy distribution must produce finite values at default "
            f"quantiles {DEFAULT_DISTRIBUTION_QUANTILES}."
        )

    return DEFAULT_DISTRIBUTION_QUANTILES


def classify_search_space_value(value: Any) -> DimensionType:
    """Classify a single search space value by its Python type.

    This function determines the dimension type based on how the
    value is provided in the search space dictionary:

    - SciPy stats distribution -> DISTRIBUTION
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
    if is_scipy_distribution(value):
        return DimensionType.DISTRIBUTION
    elif isinstance(value, tuple):
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
