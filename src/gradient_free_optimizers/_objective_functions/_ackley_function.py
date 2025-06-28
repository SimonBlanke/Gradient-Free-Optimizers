import numpy as np

from ._base_objective_function import BaseFunction


class AckleyFunction(BaseFunction):
    def __init__(self):
        self.A = 20
        self.angle = 2 * np.pi

    def objective_function(self, para):
        x = para["x0"]
        y = para["x1"]

        loss1 = -self.A * np.exp(-0.2 * np.sqrt(0.5 * (x * x + y * y)))
        loss2 = -np.exp(0.5 * (np.cos(self.angle * x) + np.cos(self.angle * y)))
        loss3 = np.exp(1)
        loss4 = self.A

        return loss1 + loss2 + loss3 + loss4

    @property
    def search_space(self):
        return {
            "x0": np.arange(-5, 5, 0.1),
            "x1": np.arange(-5, 5, 0.1),
        }
