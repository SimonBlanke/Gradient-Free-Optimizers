class BaseFunction:
    def objective_function(self, para):
        raise NotImplementedError

    def search_space(self):
        raise NotImplementedError
