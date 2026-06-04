# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Trust Region Bayesian Optimization strategy."""

from __future__ import annotations

from dataclasses import dataclass, field

from gradient_free_optimizers._array_backend import array, ndarray, zeros


@dataclass
class _TrustRegionState:
    """Mutable state for one trust region."""

    length: float
    success_count: int = 0
    failure_count: int = 0
    center: ndarray | None = None


@dataclass
class TuRBO:
    """Trust Region Bayesian Optimization candidate strategy.

    TuRBO restricts Bayesian optimization candidates to a local hyperrectangle
    around the best observed point. The trust region expands after consecutive
    improvements and shrinks after consecutive non-improvements.

    Parameters
    ----------
    n_trust_regions : int, default=1
        Number of trust regions. Only TuRBO-1 is currently supported.
    base_length : float, default=0.8
        Initial trust region side length in normalized ``[0, 1]`` space.
    min_length : float, default=0.01
        Restart threshold. The region resets to ``base_length`` below this
        length.
    max_length : float, default=1.0
        Maximum trust region side length.
    success_tolerance : int, default=3
        Number of consecutive improvements before expanding the region.
    failure_tolerance : int, default=3
        Number of consecutive non-improvements before shrinking the region.
    """

    n_trust_regions: int = 1
    base_length: float = 0.8
    min_length: float = 0.01
    max_length: float = 1.0
    success_tolerance: int = 3
    failure_tolerance: int = 3

    _regions: list[_TrustRegionState] = field(
        default_factory=list, init=False, repr=False
    )
    _active_region_idx: int = field(default=0, init=False, repr=False)
    _best_score: float = field(default=float("-inf"), init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate trust-region configuration."""
        if self.n_trust_regions != 1:
            raise ValueError("TuRBO currently supports n_trust_regions=1 only.")
        if self.min_length <= 0:
            raise ValueError("min_length must be greater than 0.")
        if self.base_length <= 0:
            raise ValueError("base_length must be greater than 0.")
        if self.max_length <= 0:
            raise ValueError("max_length must be greater than 0.")
        if not self.min_length <= self.base_length <= self.max_length:
            raise ValueError("Expected min_length <= base_length <= max_length.")
        if self.success_tolerance < 1:
            raise ValueError("success_tolerance must be at least 1.")
        if self.failure_tolerance < 1:
            raise ValueError("failure_tolerance must be at least 1.")

    @property
    def initialized(self) -> bool:
        """Return whether the trust-region state has been initialized."""
        return len(self._regions) > 0

    def initialize(self, center: ndarray) -> None:
        """Initialize the trust region around ``center`` in normalized space."""
        center = array(center, dtype=float)
        self._regions = [
            _TrustRegionState(length=self.base_length, center=center.copy())
        ]
        self._active_region_idx = 0
        self._best_score = float("-inf")

    def bounds(
        self,
        center: ndarray,
        bounds_low: ndarray,
        bounds_high: ndarray,
    ) -> tuple[ndarray, ndarray]:
        """Return active trust-region bounds clipped to search-space bounds."""
        if not self.initialized:
            self.initialize(center)

        region = self._regions[self._active_region_idx]
        center = array(center, dtype=float)
        region.center = center.copy()

        half_length = region.length / 2.0
        lower = []
        upper = []
        for value, low, high in zip(center, bounds_low, bounds_high):
            lower.append(max(float(value) - half_length, float(low)))
            upper.append(min(float(value) + half_length, float(high)))

        return array(lower), array(upper)

    def filter_candidates(
        self,
        candidates: ndarray,
        center: ndarray,
        bounds_low: ndarray,
        bounds_high: ndarray,
    ) -> ndarray:
        """Return a boolean mask for candidates inside the trust region."""
        lower, upper = self.bounds(center, bounds_low, bounds_high)
        mask = zeros(len(candidates), dtype=bool)

        for i, candidate in enumerate(candidates):
            in_region = True
            for dim_idx in range(len(center)):
                value = candidate[dim_idx]
                if value < lower[dim_idx] or value > upper[dim_idx]:
                    in_region = False
                    break
            mask[i] = in_region

        return mask

    def on_evaluation(self, score_new: float, improved: bool) -> None:
        """Update trust-region length after a completed evaluation."""
        if not self.initialized:
            return

        region = self._regions[self._active_region_idx]

        if improved:
            region.success_count += 1
            region.failure_count = 0
            self._best_score = max(self._best_score, score_new)
        else:
            region.failure_count += 1
            region.success_count = 0

        if region.success_count >= self.success_tolerance:
            region.length = min(region.length * 2.0, self.max_length)
            region.success_count = 0

        if region.failure_count >= self.failure_tolerance:
            region.length /= 2.0
            region.failure_count = 0

        if region.length < self.min_length:
            region.length = self.base_length
            region.success_count = 0
            region.failure_count = 0
