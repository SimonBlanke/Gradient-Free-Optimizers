# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import logging
import pandas as pd

from typing import Callable, Any, List
from multiprocessing.managers import DictProxy

from ._objective_adapter import ObjectiveAdapter
from ._result import Result


class CachedObjectiveAdapter(ObjectiveAdapter):
    def __init__(self, conv, objective):
        super().__init__(conv, objective)

        self.memory_dict = {}
        self.memory_dict_new = {}

    def memory(self, warm_start: pd.DataFrame, memory: Any = None):
        if isinstance(memory, DictProxy):
            self.memory_dict = memory

        if warm_start is None:
            return

        if not isinstance(warm_start, pd.DataFrame):
            logging.warning("Memory warm start must be of type pandas.DataFrame")
            logging.warning("Optimization will continue without memory warm start")
            return

        if warm_start.empty:
            logging.warning("Memory warm start has no values in current search space")
            logging.warning("Optimization will continue without memory warm start")
            return

        self.memory_dict.update(self._conv.dataframe2memory_dict(warm_start))

    def __call__(self, pos):
        pos_t = tuple(pos)

        if pos_t in self.memory_dict:
            params = self._conv.value2para(self._conv.position2value(pos))

            return self.memory_dict[pos_t], params
        else:
            result, params = self._call_objective(pos)
            self.memory_dict[pos_t] = result
            self.memory_dict_new[pos_t] = result
            return result, params
