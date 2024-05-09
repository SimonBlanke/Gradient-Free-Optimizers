# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


class ParameterTracker:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.n_considered_transitions = 0
        self.n_transitions = 0

    def transitions(function):
        def wrapper(self, *args, **kwargs):
            self.n_transitions += 1
            return function(self, *args, **kwargs)

        return wrapper

    def considered_transitions(function):
        def wrapper(self, *args, **kwargs):
            self.n_considered_transitions += 1
            return function(self, *args, **kwargs)

        return wrapper
