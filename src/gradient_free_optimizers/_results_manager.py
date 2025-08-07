import pandas as pd
from collections.abc import Sequence


class ResultsManager:
    def __init__(self):
        self._results_l = []

    def add(self, result, params) -> None:
        score_d = {"score": result.score}
        results_dict = {**score_d, **result.metrics}

        self._results_l.append({**results_dict, **params})

    @property
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._results_l)

    def best(self):
        return max(self._results, key=lambda r: r.score, default=None)
