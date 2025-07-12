class BaseFunction:
    """
    Abstract base class for defining optimization functions.

    Methods
    -------
    objective_function(para)
        Should be implemented to evaluate the objective function for given parameters.

    search_space()
        Should be implemented to define the parameter search space.
    """

    def objective_function(self, para):
        raise NotImplementedError

    def search_space(self):
        raise NotImplementedError
