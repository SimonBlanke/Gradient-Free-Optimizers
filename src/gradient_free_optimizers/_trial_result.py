from dataclasses import dataclass, asdict
from typing import Any, Mapping


@dataclass(slots=True)
class TrialResult:
    iteration: int
    pos: list[int]  # index vector (for debugging)
    params: Mapping[str, Any]  # decoded user space
    score: float
    metrics: Mapping[str, Any]  # extra outputs (may be {})

    def asdict(self) -> dict[str, Any]:
        d = asdict(self)
        # flatten metrics to top level for a nicer DataFrame
        d.update(self.metrics)
        return d
