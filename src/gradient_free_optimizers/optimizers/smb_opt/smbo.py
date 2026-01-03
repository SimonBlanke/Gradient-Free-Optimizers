# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..base_optimizer import BaseOptimizer
from .sampling import InitialSampler

import math
import logging

from gradient_free_optimizers._array_backend import (
    array,
    meshgrid,
    isnan,
    isinf,
    inf,
    nonzero,
    isin,
    intersect1d,
    random as np_random,
    HAS_NUMPY,
)

# Import numpy only for warm_start pandas operations
if HAS_NUMPY:
    import numpy as np


class SMBO(BaseOptimizer):
    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=[],
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        warm_start_smbo=None,
        max_sample_size=10000000,
        sampling={"random": 1000000},
        replacement=True,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )

        self.warm_start_smbo = warm_start_smbo
        self.max_sample_size = max_sample_size
        self.sampling = sampling
        self.replacement = replacement

        self.sampler = InitialSampler(self.conv, max_sample_size)

        self.init_warm_start_smbo(warm_start_smbo)

    def init_warm_start_smbo(self, search_data):
        if search_data is not None:
            # Note: warm_start uses pandas DataFrame, which requires numpy
            # This is expected since users providing warm_start data have dependencies
            import numpy as np

            # filter out nan and inf
            warm_start_smbo = search_data[
                ~search_data.isin([np.nan, np.inf, -np.inf]).any(axis=1)
            ]

            # filter out elements that are not in search space
            int_idx_list = []
            for para_name in self.conv.para_names:
                search_data_dim = warm_start_smbo[para_name].values
                search_space_dim = self.conv.search_space[para_name]

                int_idx = np.nonzero(
                    np.isin(search_data_dim, search_space_dim)
                )[0]
                int_idx_list.append(int_idx)

            intersec = int_idx_list[0]
            for int_idx in int_idx_list[1:]:
                intersec = np.intersect1d(intersec, int_idx)
            warm_start_smbo_f = warm_start_smbo.iloc[intersec]

            X_sample_values = warm_start_smbo_f[self.conv.para_names].values
            Y_sample = warm_start_smbo_f["score"].values

            self.X_sample = self.conv.values2positions(X_sample_values)
            self.Y_sample = list(Y_sample)

        else:
            self.X_sample = []
            self.Y_sample = []

    def track_X_sample(iterate):
        def wrapper(self, *args, **kwargs):
            pos = iterate(self, *args, **kwargs)
            self.X_sample.append(pos)
            return pos

        return wrapper

    def track_y_sample(evaluate):
        def wrapper(self, score):
            evaluate(self, score)

            if math.isnan(score) if isinstance(score, float) else isnan(score):
                del self.X_sample[-1]
            elif math.isinf(score) if isinstance(score, float) else isinf(score):
                del self.X_sample[-1]
            else:
                self.Y_sample.append(score)

        return wrapper

    def _sampling(self, all_pos_comb):
        if self.sampling is False:
            return all_pos_comb
        elif "random" in self.sampling:
            return self.random_sampling(all_pos_comb)

    def random_sampling(self, pos_comb):
        n_samples = self.sampling["random"]
        n_pos_comb = len(pos_comb) if hasattr(pos_comb, '__len__') else pos_comb.shape[0]

        if n_pos_comb <= n_samples:
            return pos_comb
        else:
            _idx_sample = np_random.choice(n_pos_comb, n_samples, replace=False)
            # Handle both numpy arrays and GFOArray
            if hasattr(pos_comb, 'shape'):
                pos_comb_sampled = pos_comb[_idx_sample, :]
            else:
                pos_comb_sampled = array([pos_comb[i] for i in _idx_sample])
            return pos_comb_sampled

    def _all_possible_pos(self):
        pos_space = self.sampler.get_pos_space()
        n_dim = len(pos_space)

        # Create meshgrid and reshape
        grids = meshgrid(*pos_space)
        # Transpose and reshape to get all combinations
        all_pos_comb = array(grids).T.reshape(-1, n_dim)

        all_pos_comb_constr = []
        for pos in all_pos_comb:
            if self.conv.not_in_constraint(pos):
                all_pos_comb_constr.append(pos)

        all_pos_comb_constr = array(all_pos_comb_constr)
        return all_pos_comb_constr

    def memory_warning(self, max_sample_size):
        if (
            self.conv.search_space_size > self.warnings
            and max_sample_size > self.warnings
        ):
            warning_message0 = "\n Warning:"
            warning_message1 = (
                "\n search space size of "
                + str(self.conv.search_space_size)
                + " exceeding recommended limit."
            )
            warning_message3 = (
                "\n Reduce search space size for better performance."
            )
            logging.warning(
                warning_message0 + warning_message1 + warning_message3
            )

    @track_X_sample
    def init_pos(self):
        return super().init_pos()

    @BaseOptimizer.track_new_pos
    @track_X_sample
    def iterate(self):
        return self._propose_location()

    def _remove_position(self, position):
        # Filter out the given position from all_pos_comb
        filtered = []
        for pos in self.all_pos_comb:
            # Check if positions match
            match = True
            for i in range(len(pos)):
                if pos[i] != position[i]:
                    match = False
                    break
            if not match:
                filtered.append(pos)
        self.all_pos_comb = array(filtered) if filtered else array([])

    @BaseOptimizer.track_new_score
    @track_y_sample
    def evaluate(self, score_new):
        self._evaluate_new2current(score_new)
        self._evaluate_current2best()

        if not self.replacement:
            self._remove_position(self.pos_new)

    @BaseOptimizer.track_new_score
    @track_y_sample
    def evaluate_init(self, score_new):
        self._evaluate_new2current(score_new)
        self._evaluate_current2best()

    def _propose_location(self):
        try:
            self._training()
        except ValueError:
            logging.warning(
                "Warning: training sequential model failed. Performing random iteration instead."
            )
            return self.move_random()

        exp_imp = self._expected_improvement()

        index_best = list(exp_imp.argsort()[::-1])
        all_pos_comb_sorted = self.pos_comb[index_best]
        pos_best = all_pos_comb_sorted[0]

        return pos_best
