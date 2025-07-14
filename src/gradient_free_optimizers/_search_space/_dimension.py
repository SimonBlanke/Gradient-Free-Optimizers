from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import numbers
import math
import random

import numpy as np

__all__ = [
    "Dimension",
    "FixedDimension",
    "CategoricalDimension",
    "RealDimension",
    "IntegerDimension",
    "DistributionDimension",
    "make_dimension",
    "dict_to_dimensions",
]

###############################################################################
# Base class
###############################################################################


class Dimension:
    """Abstract representation of *one* search‑space dimension.

    Sub‑classes must provide:
    - ``size``: *int | float* – ``np.inf`` for uncountable domains
    - ``low`` & ``high`` (where meaningful) – numeric domain bounds
    - the methods defined below.
    """

    # Public protocol ---------------------------------------------------------
    def sample(self, rng: random.Random | np.random.Generator = np.random) -> Any:
        """Draw *one* random value from the dimension."""
        raise NotImplementedError

    def grid(self, max_points: int = 256) -> np.ndarray:
        """Return a representative *finite* grid inside the domain.

        Optimizers that rely on Cartesian enumeration can fall back to this.
        The default implementation raises ``NotImplementedError`` so subclasses
        *have* to think about what an appropriate grid means for them.
        """
        raise NotImplementedError

    def perturb(
        self,
        value: Any,
        scale: float,
        rng: random.Random | np.random.Generator = np.random,
    ) -> Any:
        """Return a neighbour of *value* with local step‐size ``scale`` (≈σ).

        ``scale`` is interpreted as *fraction of the domain* (≈ 0…1).  Concrete
        implementations can translate it to natural units.
        """
        raise NotImplementedError

    # Legacy helpers (needed until all algorithms are ported) ------------------
    def value_to_pos(self, value: Any, res: int = 256) -> int:  # pragma: no cover
        raise NotImplementedError

    def pos_to_value(self, pos: int, res: int = 256) -> Any:  # pragma: no cover
        raise NotImplementedError

    # New unified helpers ------------------------------------------------------
    @property
    def span(self) -> float:
        """Characteristic linear size of the domain.

        * Continuous / integer ranges → ``high - low``
        * Discrete sets → ``len(values) - 1`` (distance in index space)
        * Fixed scalars → 0
        * Distributions → ``high - low`` (≈ quantile range)
        """
        raise NotImplementedError

    def clip(self, value: Any) -> Any:
        """Clamp *value* into the valid domain (idempotent)."""
        return value  # sensible default for discrete domains

    # Convenience --------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}({self.span})"


###############################################################################
# Discrete / categorical dimensions
###############################################################################


class FixedDimension(Dimension):
    """A scalar constant – no variability."""

    def __init__(self, constant: Any):
        self.constant = constant
        self.size: int = 1

    # –– API ------------------------------------------------------------------
    def __len__(self):
        return 1

    @property
    def span(self):
        return 0.0

    def sample(self, rng=np.random):
        return self.constant

    def grid(self, max_points: int = 1):
        return np.asarray([self.constant])

    def perturb(self, value, scale, rng=np.random):
        return self.constant

    def value_to_pos(self, value, res: int = 256):
        return 0

    def pos_to_value(self, pos: int, res: int = 256):
        return self.constant

    def clip(self, value):
        return self.constant


class CategoricalDimension(Dimension):
    """An unordered set of discrete choices."""

    def __init__(self, values: Sequence):
        if len(values) == 0:
            raise ValueError("CategoricalDimension needs at least one element.")
        self.values = list(values)
        self.size: int = len(self.values)

    def __len__(self):
        return self.size

    # –– helpers --------------------------------------------------------------
    def _index(self, value) -> int:
        try:
            return self.values.index(value)
        except ValueError as err:  # pragma: no cover
            raise ValueError(f"{value!r} not in {self.values}") from err

    # –– API ------------------------------------------------------------------
    @property
    def span(self):
        return max(self.size - 1, 0)

    def sample(self, rng=np.random):
        return rng.choice(self.values)

    def grid(self, max_points: int = 256):
        return np.asarray(self.values)

    def perturb(self, value, scale, rng=np.random):
        idx = self._index(value)
        step = rng.choice([-1, 1])  # ignore scale for now (could weight prob.)
        new_idx = (idx + step) % self.size
        return self.values[new_idx]

    # legacy
    def value_to_pos(self, value, res: int = 256):
        return self._index(value)

    def pos_to_value(self, pos: int, res: int = 256):
        return self.values[pos % self.size]

    def clip(self, value):
        # easiest: return value if valid else closest valid (fallback to sample)
        return value if value in self.values else self.sample()


###############################################################################
# Uniform continuous / integer ranges
###############################################################################


class RealDimension(Dimension):
    """A continuous interval (optionally log‑scaled)."""

    def __init__(self, low: float, high: float, log: bool = False):
        if high <= low:
            raise ValueError("'high' must be larger than 'low'.")
        self.low: float = float(low)
        self.high: float = float(high)
        self.log: bool = bool(log)
        self.size: float = math.inf

    def __len__(self):
        return self.size  # type: ignore[return-value]

    # –– helpers --------------------------------------------------------------
    def _transform(self, x: float, rev: bool = False) -> float:
        """Log‑transform helper used when *log*==True."""
        return 10 ** x if rev else math.log10(x) if self.log else x

    # –– API ------------------------------------------------------------------
    @property
    def span(self):
        return self.high - self.low

    def sample(self, rng=np.random):
        if self.log:
            return 10 ** rng.uniform(math.log10(self.low), math.log10(self.high))
        return rng.uniform(self.low, self.high)

    def grid(self, max_points: int = 256):
        if max_points <= 1:
            return np.asarray([(self.low + self.high) / 2.0])
        if self.log:
            return np.logspace(math.log10(self.low), math.log10(self.high), max_points)
        return np.linspace(self.low, self.high, max_points)

    def perturb(self, value, scale, rng=np.random):
        width = self.high - self.low
        step = rng.normal(0, scale) * width
        return self.clip(value + step)

    # legacy (coarse quantisation)
    def value_to_pos(self, value, res: int = 256):
        grid = self.grid(res)
        return int(np.abs(grid - value).argmin())

    def pos_to_value(self, pos: int, res: int = 256):
        return self.grid(res)[pos % res]

    def clip(self, value):
        return float(np.clip(value, self.low, self.high))


class IntegerDimension(RealDimension):
    """Continuous integer range."""

    def __init__(self, low: int, high: int, log: bool = False):
        super().__init__(int(low), int(high), log)

    @property
    def span(self):
        return int(self.high - self.low)

    # –– API overrides --------------------------------------------------------
    def sample(self, rng=np.random):
        return int(super().sample(rng))

    def perturb(self, value, scale, rng=np.random):
        return int(round(super().perturb(value, scale, rng)))

    def grid(self, max_points: int = 256):
        return np.arange(self.low, self.high + 1)

    def clip(self, value):
        return int(super().clip(int(value)))


###############################################################################
# Arbitrary scipy.stats distribution
###############################################################################


class DistributionDimension(Dimension):
    """Wrap a frozen *scipy.stats* distribution."""

    def __init__(self, frozen_dist):
        # Requires the frozen distribution to implement .rvs/.cdf/.ppf
        required = ("rvs", "cdf", "ppf")
        if not all(hasattr(frozen_dist, attr) for attr in required):
            raise TypeError("Expected frozen scipy.stats distribution.")

        self.dist = frozen_dist
        self.size: float = math.inf
        self.low, self.high = self.dist.ppf(0.0), self.dist.ppf(1.0)

    def __len__(self):
        return math.inf  # type: ignore[return-value]

    # –– API ------------------------------------------------------------------
    @property
    def span(self):
        return float(self.high - self.low)

    def sample(self, rng=random):
        # Let scipy handle random_state argument
        return self.dist.rvs(random_state=np.random.RandomState())

    def grid(self, max_points: int = 256):
        q = np.linspace(0.01, 0.99, max_points)
        return self.dist.ppf(q)

    def perturb(self, value, scale, rng=random):
        q = self.dist.cdf(value)
        q_new = np.clip(q + rng.normal(0, scale), 0.0, 1.0)
        return self.dist.ppf(q_new)

    # legacy
    def value_to_pos(self, value, res: int = 256):
        grid = self.grid(res)
        return int(np.abs(grid - value).argmin())

    def pos_to_value(self, pos: int, res: int = 256):
        return self.grid(res)[pos % res]

    def clip(self, value):
        print("\n self.low", self.low, "\n")
        # Naively clamp via quantiles (could also reflect / wrap)
        return float(np.clip(value, self.low, self.high))


###############################################################################
# Heuristic factories
###############################################################################


def make_dimension(raw) -> Dimension:
    """Infer the correct *Dimension* subclass from a *raw* specification."""

    # 0) already a Dimension ---------------------------------------------------
    if isinstance(raw, Dimension):
        return raw

    # 1) frozen scipy.stats distribution?  (has .rvs & .cdf & .ppf)
    if all(hasattr(raw, attr) for attr in ("rvs", "cdf", "ppf")):
        return DistributionDimension(raw)

    # 2) tuple of two numbers  → Real or Integer uniform
    if (
        isinstance(raw, tuple)
        and len(raw) == 2
        and all(isinstance(x, numbers.Number) for x in raw)
    ):
        low, high = raw
        if all(float(x).is_integer() for x in raw):
            return IntegerDimension(int(low), int(high))
        return RealDimension(float(low), float(high))

    # 3) range object  → integer grid
    if isinstance(raw, range):
        return CategoricalDimension(list(raw))

    # 4) NumPy array / list / other sequence → categorical
    if isinstance(raw, np.ndarray):
        return CategoricalDimension(list(raw))
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return CategoricalDimension(list(raw))

    # 5) scalar constant -------------------------------------------------------
    return FixedDimension(raw)


def dict_to_dimensions(search_space: dict[str, Any]) -> dict[str, Dimension]:
    """Convert a *raw* search‑space dict to a dict of Dimension instances."""

    return {k: make_dimension(v) for k, v in search_space.items()}
