import numpy as np

from ._base_function import BaseFunction


class SphereFunction(BaseFunction):
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
