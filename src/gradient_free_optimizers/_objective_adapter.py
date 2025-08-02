class ObjectiveAdapter:
    """Callable that maps *pos* ➝ (*score*, *metrics*, *params*)."""

    def __init__(self, conv, objective):
        self._conv = conv  # type: Converter
        self._objective = objective  # user-supplied callable

    def __call__(self, pos: list[int]):
        params = self._conv.value2para(self._conv.position2value(pos))
        out = self._objective(params)

        # normalise return type → (score, metrics)
        if isinstance(out, tuple):
            score, add_results_d = out
        else:
            score, add_results_d = float(out), {}

        return score, add_results_d, params
