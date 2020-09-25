# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


class ResultsManager:
    def __init__(self, objective_function, search_space):
        super().__init__()
        self.objective_function = objective_function
        self.search_space = search_space

        self.results_list = []

    def _pos2para(self, pos):
        para = {}
        for key, p_ in zip(self.search_space.keys(), pos):
            para[key] = p_

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

    def score(self, pos):
        para = self._pos2para(pos)
        obj_func_results = self._obj_func_results(para)

        self.results_list.append({**obj_func_results, **para})

        return obj_func_results["score"]

