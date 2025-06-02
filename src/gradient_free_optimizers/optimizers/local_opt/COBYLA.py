# copyright: hyperactive developers, MIT License (see LICENSE file)

# todo: write an informative docstring for the file or module, remove the above

__author__ = ["SimonBlanke", "yashnator"]

from ..base_optimizer import BaseOptimizer

import numpy as np

class COBYLA(BaseOptimizer):
    """TODO: write docstring

    COBYLA: Constrained Optimization BY Linear Approximation
    
    Reference:
    Powell, M.J.D. (1994). A Direct Search Optimization Method That Models the Objective and Constraint Functions by Linear Interpolation. 
    In: Gomez, S., Hennart, JP. (eds) Advances in Optimization and Numerical Analysis. Mathematics and Its Applications, vol 275. Springer, 
    Dordrecht. 
    Source: https://doi.org/10.1007/978-94-015-8330-5_4

    Parameters
    ----------
    rho_beg : int
        Initial tolerance (large enough for coarse exploration)
    rho_end : string, optional (default='default')
        Required tolerance
    x_0 : np.array
        Initial point for creating a simplex for the optimization

    Examples
    --------
    # TODO: Write examples
    
    >>> import numpy as np
    >>> from gradient_free_optimizers import COBYLA
    >>> 
    >>> def sphere_function(para: np.array):
    ...     x = para[0]
    ...     y = para[1]
    ...     return -(x * x + y * y)
    >>> 
    >>> def constraint_1(para):
    ...     return para[0] > -5
    >>> 
    >>> search_space = {
    ...     "x": np.arange(-10, 10, 0.1),
    ...     "y": np.arange(-10, 10, 0.1),
    ... }
    >>> 
    >>> opt = COBYLA(
    ...     search_space=search_space,
    ...     rho_beg=1.0,
    ...     rho_end=0.01,
    ...     x_0=np.array([0.0, 0.0]),
    ...     constraints=[constraint_1]
    ... )
    >>> opt.search(sphere_function, n_iter=10)
    """

    _tags = {
        "authors": ["SimonBlanke", "yashnator"],
    }
    
    def _generate_initial_simplex(self, x_0_initial, rho_beg):
        n = x_0_initial.shape[0]
        arr = np.ones((n + 1, 1)) * x_0_initial + rho_beg * np.eye(n + 1, n)
        print(arr)
        return arr
    
    def _vertices_to_oppsite_face_distances(self):
        """
        Compute the distances from each vertex of an n-dimensional-simplex
        to the opposite (n-1)-dimensional face.
        
        For each vertex, the opposite hyperplane is obtained after removing the current
        vertex and then finding the projection on the subspace spanned by the hyperplane.
        The distance is then the L2 norm between projection and the current vertex.
        
        Args:
            self: instance of current COBYLA class

        Returns:
            distances: (n+1,) array of distances from each vertex to its opposite face.
        """
        distances = np.zeros(self.dim + 1)
        for j in range(0, self.dim + 1):
            face_vertices = np.delete(self.simplex, j, axis=0) 
            start_vertex = face_vertices[0]
            A = np.stack([v - start_vertex for v in face_vertices[1:]], axis=1)
            b = self.simplex[j] - start_vertex

            AtA = A.T @ A
            proj_j = A @ np.linalg.solve(AtA, A.T @ b)
            distances[j] = np.linalg.norm(b - proj_j)
        return distances
    
    def _is_simplex_acceptable(self):
        eta = [np.linalg.norm(x - self.simplex[0]) for x in self.simplex]
        eta_constraint = self.beta * self._rho
        for eta_j in eta:
            if eta_j > eta_constraint:
                return False
        sigma = self._vertices_to_oppsite_face_distances()
        sigma_constraint = self.alpha * self._rho
        print(sigma)
        for sigma_j in sigma:
            if sigma_j < sigma_constraint:
                return False
        return True
    
    def _eval_constraints(self, pos):
        # TODO: evalute constraints in optimized way
        
        return None
    
    def _merit_value(self, pos):
        # TODO: write the merit function using the _eval_constraints
        return 0

    def __init__(
        self,
        search_space,
        x_0: np.array, 
        rho_beg: int, 
        rho_end: int, 
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=[],
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        alpha = 0.25,
        beta = 2.1
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )
        self.dim = np.shape(x_0)[0]
        self.simplex = self._generate_initial_simplex(x_0, rho_beg)
        self.rho_beg = rho_beg
        self.rho_end = rho_end
        self._rho = rho_beg
        self.state = 0
        self.FLAG = 0
        
        if alpha <= 0 or alpha >= 1:
            raise ValueError("Parameter alpha must belong to the range (0, 1)")
        
        if beta <= 1:
            raise ValueError("Parameter beta must belong to the range (1, ∞)")
        
        self.alpha = alpha
        self.beta = beta

    def iterate(self):
        # TODO: Impl
        return self.simplex[0]
    
    def _score(self, pos):
        # TODO: Impl
        return 0.0
    
    def evaluate(self, pos):
        # TODO: Impl
        return self._merit_value(pos)
        
    def finish_search(self):
        # TODO: Impl
        return None