from dataclasses import dataclass


@dataclass
class Result:
    score: float
    metrics: dict
