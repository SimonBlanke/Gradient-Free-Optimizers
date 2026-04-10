from ._fitness_mapper import FitnessMapper, ScalarIdentity
from ._result import Result, objectives_as_list, unpack_objective_result


class ObjectiveAdapter:
    """Maps *pos* → (Result, params).

    When a FitnessMapper is provided, the raw objectives are passed
    through it to produce the scalar fitness stored in ``Result.score``.
    For single-objective (default), the identity mapper is used.
    """

    def __init__(self, conv, objective, fitness_mapper=None, n_objectives=1):
        self._conv = conv
        self._objective = objective
        self._fitness_mapper: FitnessMapper = fitness_mapper or ScalarIdentity()
        self._n_objectives = n_objectives

    def _call_objective(self, pos):
        """Run the underlying objective and normalise outputs."""
        params = self._conv.value2para(self._conv.position2value(pos))
        out = self._objective(params)

        objectives, metrics = unpack_objective_result(out)
        fitness = self._fitness_mapper(objectives)
        obj_list = objectives_as_list(objectives, self._n_objectives)
        result = Result(fitness, metrics, obj_list)

        return result, params

    def __call__(self, pos):
        return self._call_objective(pos)
