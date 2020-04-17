# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .opt_args import Arguments


class BaseOptimizer:
    def __init__(self, space_dim, opt_para):
        self._opt_args_ = Arguments(**opt_para)
        self._opt_args_.set_opt_args()

        self.space_dim = space_dim
        self.opt_para = opt_para

        self.nth_iter = 0
        self.p_list = []

    def _base_init_pos(self, init_position, positioner):
        p = positioner(self.space_dim, self._opt_args_)
        self.p_current = p
        self.p_current.pos_new = init_position

        self.p_list.append(self.p_current)

    def _base_iterate(self, nth_iter):
        self.nth_iter = nth_iter
        self.p_current = self.p_list[self.nth_iter % len(self.p_list)]

    def _base_evaluate(self, score_new):
        if score_new >= self.p_current.score_best:
            self.p_current.score_best = score_new
            self.p_current.pos_best = self.p_current.pos_new

    def _update_pos(self, _cand_, _p_):
        if _p_.score_new > _p_.score_best:
            _p_.pos_best = _p_.pos_new
            _p_.score_best = _p_.score_new

        if _p_.score_new > _cand_.score_best:
            _p_.pos_current = _p_.pos_new
            _p_.score_current = _p_.score_new

            _cand_.pos_best = _p_.pos_new
            _cand_.score_best = _p_.score_new

            self._pbar_.best_since_iter = _cand_.i
