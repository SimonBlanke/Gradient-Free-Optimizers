from ._trial_result import TrialResult


class ObjectiveAdapter:
    """Callable that maps *pos* ➝ (*score*, *metrics*, *params*)."""

    def __init__(self, conv, objective):
        self._conv = conv  # type: Converter
        self._objective = objective  # user-supplied callable

    def __call__(self, pos: list[int]) -> TrialResult:
        params = self._conv.value2para(self._conv.position2value(pos))
        out = self._objective(params)

        # normalise return type → (score, metrics)
        if isinstance(out, tuple):
            score, metrics = out
        else:
            score, metrics = float(out), {}

        return score, metrics, params
