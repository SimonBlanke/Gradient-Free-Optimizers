from ._result import Result, unpack_objective_result
from ._search_params import SearchParams


class ObjectiveAdapter:
    """Maps *pos* -> (Result, params, active_mask).

    When conditions are configured on the Converter, the adapter:
    1. Builds the full params dict from the position
    2. Evaluates conditions to determine which params are active
    3. Wraps only the active params in a SearchParams object
    4. Passes SearchParams to the objective function
    5. Applies any deferred updates (set_conditions/set_constraints)
       after the objective returns
    """

    def __init__(self, conv, objective, optimizer_ref=None):
        self._conv = conv
        self._objective = objective
        self._optimizer_ref = optimizer_ref

    def _call_objective(self, pos):
        """Run the underlying objective and normalise outputs."""
        full_params = self._conv.value2para(self._conv.position2value(pos))

        if self._conv.conditions:
            active_mask = self._conv.evaluate_conditions(full_params)
            filtered = {k: v for k, v in full_params.items() if active_mask[k]}
        else:
            active_mask = None
            filtered = full_params

        params = SearchParams(filtered, optimizer_ref=self._optimizer_ref)
        out = self._objective(params)
        params._apply_deferred()

        score, metrics = unpack_objective_result(out)
        result = Result(score, metrics)

        return result, full_params, active_mask

    def __call__(self, pos):
        return self._call_objective(pos)
