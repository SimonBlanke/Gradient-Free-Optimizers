from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.random import Generator, default_rng

from .dimensions import BaseDimension, CategoricalDimension


def _ensure_rng(rng: Optional[Generator]) -> Generator:
    return rng if isinstance(rng, Generator) else default_rng()


@dataclass
class ConverterV2:
    names: Sequence[str]
    dims: Sequence[BaseDimension]

    # Helpers for optimizers
    @property
    def n_dims(self) -> int:
        return len(self.dims)

    @property
    def kinds(self) -> List[str]:
        return [d.kind for d in self.dims]

    def mask_categorical(self) -> np.ndarray:
        return np.array([d.kind == "categorical" for d in self.dims], dtype=bool)

    # Core conversions
    def z_to_values(self, z: Sequence[float]) -> List[Any]:
        if len(z) != len(self.dims):
            raise ValueError("z length mismatch")
        return [dim.z_to_value(float(zz)) for dim, zz in zip(self.dims, z)]

    def values_to_z(self, values: Sequence[Any]) -> List[float]:
        if len(values) != len(self.dims):
            raise ValueError("values length mismatch")
        return [float(dim.value_to_z(v)) for dim, v in zip(self.dims, values)]

    def sample_z(self, rng: Optional[Generator] = None) -> List[float]:
        r = _ensure_rng(rng)
        zz: List[float] = []
        for dim in self.dims:
            # Sample native and convert to z to avoid bias
            v = dim.sample(r)
            zz.append(float(dim.value_to_z(v)))
        return zz

    def grid_z(self, max_points: int) -> List[List[float]]:
        # Per-dimension grids in z-space
        out: List[List[float]] = []
        for dim in self.dims:
            grid_vals = dim.grid(max_points)
            out.append([float(dim.value_to_z(v)) for v in grid_vals])
        return out

    # Params mapping
    def values_to_params(self, values: Sequence[Any]) -> Dict[str, Any]:
        return {name: val for name, val in zip(self.names, values)}

    def params_to_values(self, params: Dict[str, Any]) -> List[Any]:
        return [params[name] for name in self.names]


def converter_from_search_space(names: Sequence[str], dims: Sequence[BaseDimension]) -> ConverterV2:
    return ConverterV2(names=list(names), dims=list(dims))


@dataclass
class ObjectiveAdapter:
    converter: ConverterV2
    objective: Any  # Callable[[Dict[str, Any]], Any]

    def evaluate_z(self, z: Sequence[float]) -> Any:
        values = self.converter.z_to_values(z)
        params = self.converter.values_to_params(values)
        return self.objective(params)


def _canonical_value(v: Any) -> Any:
    # JSON-safe canonicalization for caching keys
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, np.ndarray):
        return ("np", tuple(map(_canonical_value, v.tolist())))
    # Functions/callables: use qualified name if available
    if callable(v):
        qual = getattr(v, "__qualname__", None) or getattr(v, "__name__", None)
        mod = getattr(v, "__module__", "")
        return ("callable", f"{mod}:{qual}")
    try:
        return ("repr", repr(v))
    except Exception:
        return ("id", id(v))


@dataclass
class CachedObjectiveAdapter(ObjectiveAdapter):
    cache: Optional[Dict[Tuple[Tuple[str, Any], ...], Any]] = None

    def __post_init__(self) -> None:
        if self.cache is None:
            self.cache = {}

    def _key_from_params(self, params: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
        items = tuple((k, _canonical_value(v)) for k, v in sorted(params.items()))
        return items

    def evaluate_z(self, z: Sequence[float]) -> Any:
        values = self.converter.z_to_values(z)
        params = self.converter.values_to_params(values)
        key = self._key_from_params(params)
        if key in self.cache:  # type: ignore[operator]
            return self.cache[key]  # type: ignore[index]
        res = self.objective(params)
        self.cache[key] = res  # type: ignore[index]
        return res

