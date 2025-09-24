from dataclasses import dataclass, field
from array import array
from typing import Tuple


@dataclass
class Result:
    score: float
    metrics: dict
