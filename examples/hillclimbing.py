import numpy as np
from gradient_free_optimizers import HillClimbingOptimizer


n_iter = 10


def get_score(pos_new):
    x1 = pos_new[0]

    return -x1 * x1


space_dim = np.array([100])
init_positions = [np.array([10])]


opt = HillClimbingOptimizer(init_positions, space_dim, opt_para={})

for nth_init in range(len(init_positions)):
    pos_new = opt.init_pos(nth_init)
    score_new = get_score(pos_new)
    opt.evaluate(score_new)


for nth_iter in range(len(init_positions), n_iter):
    pos_new = opt.iterate(nth_iter)
    score_new = get_score(pos_new)
    opt.evaluate(score_new)
