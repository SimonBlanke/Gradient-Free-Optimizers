import numpy as np


def get_score(pos_new):
    return -(pos_new[0] * pos_new[0])


space_dim = np.array([10])
init_positions = [np.array([0]), np.array([1]), np.array([2]), np.array([3])]


def _base_test(
    opt_class,
    n_iter,
    get_score=get_score,
    space_dim=space_dim,
    init_positions=init_positions,
    opt_para={},
):
    opt = opt_class(space_dim, **opt_para)

    for init_position in init_positions:
        opt.init_pos(init_position)
        score_new = get_score(init_position)
        opt.evaluate(score_new)

    for _ in range(len(init_positions), n_iter):
        pos_new = opt.iterate()
        score_new = get_score(pos_new)
        opt.evaluate(score_new)
