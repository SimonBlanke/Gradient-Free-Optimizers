from __future__ import annotations

from dataclasses import MISSING, Field, fields, is_dataclass
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats._distn_infrastructure import rv_frozen

from .dimensions import (
    BaseDimension,
    CategoricalDimension,
    DistributionDimension,
    FixedDimension,
    IntegerDimension,
    RealDimension,
)


class BaseSearchSpace:
    """
    Base class to define v2 search spaces as dataclasses.

    Users define a dataclass inheriting from this class, where the type/values of
    each field specify the dimension kind according to the following rules:

    - list[T] -> CategoricalDimension(values=list)
    - numpy.ndarray (numeric) -> Discrete grid (treated as categorical values)
    - (low: number, high: number) tuple -> RealDimension(low, high)
    - scipy.stats rv_frozen (bounded) -> DistributionDimension(rv)
    - scalar -> FixedDimension(value)
    """

    def _build_dimensions(self) -> Tuple[List[str], List[BaseDimension]]:
        if not is_dataclass(self):
            raise TypeError("Search space must be a dataclass inheriting BaseSearchSpace")

        names: List[str] = []
        dims: List[BaseDimension] = []
        for f in fields(self):
            name = f.name
            value = getattr(self, name)
            dim = self._infer_dimension(value)
            names.append(name)
            dims.append(dim)
        return names, dims

    @staticmethod
    def _infer_dimension(value: Any) -> BaseDimension:
        # list -> categorical
        if isinstance(value, list):
            return CategoricalDimension(values=value)

        # numpy array -> discrete numeric grid (categorical over values)
        if isinstance(value, np.ndarray):
            return CategoricalDimension(values=value.tolist())

        # tuple of (low, high) numbers -> real
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and all(isinstance(v, (int, float, np.floating, np.integer)) for v in value)
        ):
            low, high = float(value[0]), float(value[1])
            return RealDimension(low=low, high=high)

        # scipy rv_frozen -> distribution (bounded by support)
        if isinstance(value, rv_frozen):
            return DistributionDimension(rv=value)

        # scalar -> fixed
        return FixedDimension(value=value)


@dataclass
class CombinedSpace:
    variants: Sequence[BaseSearchSpace]

    def __post_init__(self) -> None:
        if len(self.variants) == 0:
            raise ValueError("CombinedSpace requires at least one variant")

    def build(self) -> Tuple[List[str], List[BaseDimension], List[int]]:
        """Builds a flat list of dimensions by introducing an implicit selector.

        Returns:
            names: list of parameter names (selector first as "__variant__")
            dims: list of BaseDimension (selector first as categorical over variants)
            offsets: cumulative starting index per variant (for external use)
        """
        variant_dims: List[Tuple[List[str], List[BaseDimension]]] = []
        for v in self.variants:
            names, dims = v._build_dimensions()
            variant_dims.append((names, dims))

        selector = CategoricalDimension(values=list(range(len(self.variants))))
        names: List[str] = ["__variant__"]
        dims: List[BaseDimension] = [selector]
        offsets: List[int] = [1]
        for n, d in variant_dims:
            names.extend(n)
            dims.extend(d)
            offsets.append(len(names))
        return names, dims, offsets


def combine_spaces(*spaces: BaseSearchSpace) -> CombinedSpace:
    return CombinedSpace(variants=list(spaces))


# Syntactic sugar: allow `space1 + space2` to create CombinedSpace
def _space_add(self: BaseSearchSpace, other: BaseSearchSpace) -> CombinedSpace:
    return combine_spaces(self, other)


setattr(BaseSearchSpace, "__add__", _space_add)

