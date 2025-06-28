import numpy as np

from ._base_function import BaseFunction


class SphereFunction(BaseFunction):
    """
    Implements the Sphere objective function for optimization tasks.

    Attributes:
        n_dim (int): Number of dimensions.
        A (float): Coefficient for the quadratic term.

    Methods:
        objective_function(para): Calculates the Sphere function value for a given parameter dictionary.
        search_space: Property that defines the search space for each dimension as a range from -8 to 8 with step 0.1.
    """

    def __init__(self, n_dim, A=1):
        self.n_dim = n_dim
        self.A = A

    def objective_function(self, para):
        loss = 0
        for dim in range(self.n_dim):
            dim_str = "x" + str(dim)
            x = para[dim_str]

            loss += self.A * x * x

        return loss

    @property
    def search_space(self):
        return {"x" + str(idx): np.arange(-8, 8, 0.1) for idx in range(self.n_dim)}
