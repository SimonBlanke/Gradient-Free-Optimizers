import pandas as pd
from collections.abc import Sequence

from ._trial_result import TrialResult


class ResultsManager:
    def __init__(self):
        self._results: list[TrialResult] = []

    def add(self, result: TrialResult) -> None:
        self._results.append(result)

    @property
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(r.asdict() for r in self._results)

    def best(self) -> TrialResult | None:
        return max(self._results, key=lambda r: r.score, default=None)
