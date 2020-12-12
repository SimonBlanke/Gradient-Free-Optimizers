# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


class ResultsManager:
    def __init__(self, conv):
        super().__init__()
        self.conv = conv

        self.results_list = []

    def _obj_func_results(self, objective_function, para):
        results = objective_function(para)

        if isinstance(results, tuple):
            score = results[0]
            results_dict = results[1]
        else:
            score = results
            results_dict = {}

        results_dict["score"] = score

        return results_dict

    def score(self, objective_function):
        def _wrapper(pos):
            value = self.conv.position2value(pos)
            para = self.conv.value2para(value)
            results_dict = self._obj_func_results(objective_function, para)

            self.results_list.append({**results_dict, **para})

            return results_dict["score"]

        return _wrapper

