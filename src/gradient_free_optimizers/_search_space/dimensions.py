from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.random import Generator, default_rng
from scipy.stats._distn_infrastructure import rv_frozen


def _ensure_rng(rng: Optional[Generator]) -> Generator:
    return rng if isinstance(rng, Generator) else default_rng()


class BaseDimension:
    kind: str = "base"

    def value_to_z(self, value: Any) -> float:
        raise NotImplementedError

    def z_to_value(self, z: float) -> Any:
        raise NotImplementedError

    def sample(self, rng: Optional[Generator] = None) -> Any:
        raise NotImplementedError

    def perturb(self, value: Any, scale: float, rng: Optional[Generator] = None) -> Any:
        raise NotImplementedError

    def grid(self, max_points: int) -> List[Any]:
        raise NotImplementedError

    @property
    def size(self) -> Optional[int]:
        return None


@dataclass
class FixedDimension(BaseDimension):
    value: Any

    kind: str = "fixed"

    def value_to_z(self, value: Any) -> float:
        return 0.0

    def z_to_value(self, z: float) -> Any:
        return self.value

    def sample(self, rng: Optional[Generator] = None) -> Any:
        return self.value

    def perturb(self, value: Any, scale: float, rng: Optional[Generator] = None) -> Any:
        return self.value

    def grid(self, max_points: int) -> List[Any]:
        return [self.value]

    @property
    def size(self) -> Optional[int]:
        return 1


@dataclass
class CategoricalDimension(BaseDimension):
    values: Sequence[Any]

    kind: str = "categorical"

    def __post_init__(self) -> None:
        if len(self.values) == 0:
            raise ValueError("CategoricalDimension requires non-empty values")

    def _clamp_z_index(self, z: float) -> int:
        n = len(self.values)
        if n == 1:
            return 0
        zz = min(max(z, 0.0), 1.0 - np.finfo(float).eps)
        i = int(np.floor(zz * n))
        return max(0, min(n - 1, i))

    def value_to_z(self, value: Any) -> float:
        n = len(self.values)
        try:
            idx = self.values.index(value)  # type: ignore[attr-defined]
        except Exception:
            # Fallback for non-list sequences
            idx = list(self.values).index(value)
        if n == 1:
            return 0.0
        # Map index center to z bucket center
        return (idx + 0.5) / n

    def z_to_value(self, z: float) -> Any:
        return self.values[self._clamp_z_index(z)]

    def sample(self, rng: Optional[Generator] = None) -> Any:
        r = _ensure_rng(rng)
        idx = r.integers(0, len(self.values))
        return self.values[int(idx)]

    def perturb(self, value: Any, scale: float, rng: Optional[Generator] = None) -> Any:
        r = _ensure_rng(rng)
        try:
            idx = self.values.index(value)  # type: ignore[attr-defined]
        except Exception:
            idx = list(self.values).index(value)

        n = len(self.values)
        if n <= 1:
            return value

        # Neighbor-based move; scale controls maximum hop length
        max_hops = max(1, int(np.ceil(float(max(0.0, min(1.0, scale))) * (n - 1))))
        hop = int(r.integers(-max_hops, max_hops + 1))
        hop = -1 if hop == 0 else hop  # ensure movement
        new_idx = int(np.clip(idx + hop, 0, n - 1))
        return self.values[new_idx]

    def grid(self, max_points: int) -> List[Any]:
        n = len(self.values)
        if n <= max_points:
            return list(self.values)
        # Evenly spaced subsample of indices
        idxs = np.linspace(0, n - 1, num=max_points)
        idxs = np.unique(np.round(idxs).astype(int))
        return [self.values[i] for i in idxs]

    @property
    def size(self) -> Optional[int]:
        return len(self.values)


@dataclass
class RealDimension(BaseDimension):
    low: float
    high: float
    brackets: str = "[]"  # one of: [], [), (], ()

    kind: str = "real"

    def __post_init__(self) -> None:
        if not np.isfinite(self.low) or not np.isfinite(self.high) or self.high <= self.low:
            raise ValueError("RealDimension requires finite low < high")
        if self.brackets not in ("[]", "[)", "(]", "()"):
            raise ValueError("Invalid brackets for RealDimension; use one of [], [), (], ()")
        self._include_low = self.brackets[0] == "["
        self._include_high = self.brackets[1] == ")" and False or True  # keep clarity
        self._include_high = self.brackets[1] == "]"

    def value_to_z(self, value: float) -> float:
        # Permissive mapping: accept endpoints regardless of brackets
        return float((value - self.low) / (self.high - self.low))

    def z_to_value(self, z: float) -> float:
        # Permissive mapping: z in [0,1] maps linearly; brackets handled in sampling/grid/perturb
        zz = float(np.clip(z, 0.0, 1.0))
        return float(self.low + zz * (self.high - self.low))

    def _nudge_inside(self, z: float) -> float:
        # Push exactly-at-boundary z into interior if corresponding side is open
        if z <= 0.0 and not self._include_low:
            return float(np.nextafter(0.0, 1.0))
        if z >= 1.0 and not self._include_high:
            return float(np.nextafter(1.0, 0.0))
        return z

    def sample(self, rng: Optional[Generator] = None) -> float:
        r = _ensure_rng(rng)
        z = float(r.random())  # [0,1)
        z = self._nudge_inside(z)
        return self.z_to_value(z)

    def perturb(self, value: float, scale: float, rng: Optional[Generator] = None) -> float:
        r = _ensure_rng(rng)
        z = self.value_to_z(value)
        step = float(r.normal(0.0, max(1e-12, float(scale))))
        z_new = float(np.clip(z + step, 0.0, 1.0))
        z_new = self._nudge_inside(z_new)
        return self.z_to_value(z_new)

    def grid(self, max_points: int) -> List[float]:
        n = max(2, int(max_points))
        if self.brackets == "[]":
            zz = np.linspace(0.0, 1.0, num=n)
        elif self.brackets == "[)":
            zz = np.linspace(0.0, 1.0, num=n + 1)[:-1]
        elif self.brackets == "(]":
            zz = np.linspace(0.0, 1.0, num=n + 1)[1:]
        else:  # ()
            zz = np.linspace(0.0, 1.0, num=n + 2)[1:-1]
        return [self.z_to_value(float(z)) for z in zz]


@dataclass
class IntegerDimension(BaseDimension):
    low: int
    high: int  # inclusive
    brackets: str = "[]"  # one of: [], [), (], ()

    kind: str = "integer"

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError("IntegerDimension requires low <= high")
        if self.brackets not in ("[]", "[)", "(]", "()"):
            raise ValueError("Invalid brackets for IntegerDimension; use one of [], [), (], ()")
        include_low = self.brackets[0] == "["
        include_high = self.brackets[1] == "]"
        self._lo_eff = int(self.low + (0 if include_low else 1))
        self._hi_eff = int(self.high - (0 if include_high else 1))
        if self._hi_eff < self._lo_eff:
            raise ValueError("IntegerDimension has empty set under given brackets")

    def value_to_z(self, value: int) -> float:
        if self._lo_eff == self._hi_eff:
            return 0.0
        return float((int(value) - self._lo_eff) / (self._hi_eff - self._lo_eff))

    def z_to_value(self, z: float) -> int:
        if self._lo_eff == self._hi_eff:
            return int(self._lo_eff)
        zz = float(np.clip(z, 0.0, 1.0))
        x = self._lo_eff + zz * (self._hi_eff - self._lo_eff)
        return int(np.clip(int(np.round(x)), self._lo_eff, self._hi_eff))

    def sample(self, rng: Optional[Generator] = None) -> int:
        r = _ensure_rng(rng)
        return int(r.integers(self._lo_eff, self._hi_eff + 1))

    def perturb(self, value: int, scale: float, rng: Optional[Generator] = None) -> int:
        r = _ensure_rng(rng)
        z = self.value_to_z(value)
        step = float(r.normal(0.0, max(1e-12, float(scale))))
        return self.z_to_value(float(np.clip(z + step, 0.0, 1.0)))

    def grid(self, max_points: int) -> List[int]:
        n = self.size or 0
        if n <= max_points:
            return list(range(self._lo_eff, self._hi_eff + 1))
        zz = np.linspace(0.0, 1.0, num=max_points)
        vals = [self.z_to_value(z) for z in zz]
        # Ensure uniqueness and sorting
        return sorted(list(dict.fromkeys(vals)))

    @property
    def size(self) -> Optional[int]:
        return self._hi_eff - self._lo_eff + 1


@dataclass
class DistributionDimension(BaseDimension):
    rv: rv_frozen
    low: Optional[float] = None
    high: Optional[float] = None

    kind: str = "distribution"

    def __post_init__(self) -> None:
        # Determine support if not explicitly given
        a, b = self.rv.support()
        lo = self.low if self.low is not None else a
        hi = self.high if self.high is not None else b
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError(
                "DistributionDimension requires finite bounds; provide low/high or use bounded rv"
            )
        self.low = float(lo)
        self.high = float(hi)
        self._F_low = float(self.rv.cdf(self.low))
        self._F_high = float(self.rv.cdf(self.high))
        self._den = max(1e-16, self._F_high - self._F_low)

    def _to_base_p(self, z: float) -> float:
        zz = float(np.clip(z, 0.0, 1.0))
        return float(self._F_low + zz * self._den)

    def value_to_z(self, value: float) -> float:
        Fv = float(self.rv.cdf(float(value)))
        return float(np.clip((Fv - self._F_low) / self._den, 0.0, 1.0))

    def z_to_value(self, z: float) -> float:
        p = self._to_base_p(z)
        x = float(self.rv.ppf(p))
        return float(np.clip(x, self.low, self.high))  # numerical safety

    def sample(self, rng: Optional[Generator] = None) -> float:
        # sample by inverse CDF over truncated mass
        r = _ensure_rng(rng)
        z = float(r.random())
        return self.z_to_value(z)

    def perturb(self, value: float, scale: float, rng: Optional[Generator] = None) -> float:
        r = _ensure_rng(rng)
        z = self.value_to_z(value)
        step = float(r.normal(0.0, max(1e-12, float(scale))))
        return self.z_to_value(float(np.clip(z + step, 0.0, 1.0)))

    def grid(self, max_points: int) -> List[float]:
        zz = np.linspace(0.0, 1.0, num=max(2, max_points))
        return [self.z_to_value(z) for z in zz]
