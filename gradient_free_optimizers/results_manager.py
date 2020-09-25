# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .conv import position2value


class ResultsManager:
    def __init__(self, objective_function, search_space):
        super().__init__()
        self.objective_function = objective_function
        self.search_space = search_space

        self.results_list = []

    def _value2para(self, pos):
        para = {}
        for key, p_ in zip(self.search_space.keys(), pos):
            para[key] = p_

        return para

    def pos2para(self, pos):
        value = position2value(self.search_space, pos)
        para = self._value2para(value)
        return para

    def _obj_func_results(self, para):
        results = self.objective_function(para)

        if isinstance(results, tuple):
            score = results[0]
            results_dict = results[1]
        else:
            score = results
            results_dict = {}

        results_dict["score"] = score

        return results_dict

    def score(self, pos_new):
        para_new = self.pos2para(pos_new)
        obj_func_results = self._obj_func_results(para_new)

        self.results_list.append({**obj_func_results, **para_new})

        return obj_func_results["score"]

