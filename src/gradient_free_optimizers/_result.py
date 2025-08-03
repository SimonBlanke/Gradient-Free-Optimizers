from dataclasses import dataclass, field
from array import array
from typing import Tuple


@dataclass(slots=True)
class Result:
    score: float
    metrics: dict
