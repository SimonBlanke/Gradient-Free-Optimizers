from ._result import Result, unpack_objective_result
from ._search_params import SearchParams


class ObjectiveAdapter:
    """Maps *pos* → (Result, params)."""

    def __init__(self, conv, objective):
        self._conv = conv
        self._objective = objective  # user callable
        self._metadata = {}

    def _call_objective(self, pos):
        """Run the underlying objective and normalise outputs."""
        params = self._conv.value2para(self._conv.position2value(pos))

        search_params = SearchParams(params)
        for key, value in self._metadata.items():
            setattr(search_params, f"_{key}", value)

        out = self._objective(search_params)

        score, metrics = unpack_objective_result(out)
        result = Result(score, metrics)

        return result, params

    # keep one public entry point so subclasses can override cleanly
    def __call__(self, pos):
        return self._call_objective(pos)
