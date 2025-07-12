import numpy as np

from ._base_function import BaseFunction


class AckleyFunction(BaseFunction):
    """
    Implements the Ackley function, a common benchmark function for optimization
    algorithms. Inherits from BaseFunction.

    Attributes:
        A (float): A constant used in the Ackley function.
        angle (float): The angle in radians used for cosine calculations.

    Methods:
        objective_function(para): Computes the Ackley function value for given
        parameters 'x0' and 'x1'.

        search_space: Defines the search space for the parameters 'x0' and 'x1',
        ranging from -5 to 5 with a step of 0.1.
    """

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
