# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


class ExpectedImprovementBasedOptimization(SMBO):
    def __init__(
        self,
        *args,
        xi=0.01,
        warm_start_smbo=None,
        sampling={"random": 1000000},
        warnings=100000000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.new_positions = []
        self.xi = xi
        self.warm_start_smbo = warm_start_smbo
        self.sampling = sampling
        self.warnings = warnings
