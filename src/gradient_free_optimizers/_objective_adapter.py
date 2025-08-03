from ._result import Result


class ObjectiveAdapter:
    """Maps *pos* → (score, metrics, params)."""

    def __init__(self, conv, objective):
        self._conv = conv
        self._objective = objective  # user callable

    def _call_objective(self, pos):
        """Run the underlying objective and normalise outputs."""
        params = self._conv.value2para(self._conv.position2value(pos))
        out = self._objective(params)

        if isinstance(out, tuple):
            score, metrics = out
        else:
            score, metrics = float(out), {}

        result = Result(score, metrics)

        return result, params

    # keep one public entry point so subclasses can override cleanly
    def __call__(self, pos):
        return self._call_objective(pos)
