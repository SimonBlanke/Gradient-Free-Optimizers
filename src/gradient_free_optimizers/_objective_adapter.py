from ._result import Result, unpack_objective_result


class ObjectiveAdapter:
    """Maps *pos* → (Result, params)."""

    def __init__(self, conv, objective):
        self._conv = conv
        self._objective = objective  # user callable

    def _call_objective(self, pos):
        """Run the underlying objective and normalise outputs."""
        params = self._conv.value2para(self._conv.position2value(pos))
        out = self._objective(params)

        score, metrics = unpack_objective_result(out)
        result = Result(score, metrics)

        return result, params

    # keep one public entry point so subclasses can override cleanly
    def __call__(self, pos):
        return self._call_objective(pos)
