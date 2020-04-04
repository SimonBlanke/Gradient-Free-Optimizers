# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .base_positioner import BasePositioner
from .opt_args import Arguments


class BaseOptimizer:
    def __init__(self, n_iter, opt_para):
        self._opt_args_ = Arguments(**opt_para)
        self._opt_args_.set_opt_args(n_iter)

        self.p_list = []

    def iterate(self, i, _cand_):
        self.i = i

        if i < self.n_positioners:
            p = self._init_iteration(_cand_)
            self.p_list.append(p)

        else:
            self._iterate(i, _cand_)

        return _cand_

    def _finish_search(self):
        self._pbar_.close_p_bar()

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

    def _optimizer_eval(self, _cand_, _p_):
        _p_.score_new = _cand_.eval_pos(_p_.pos_new)
        self._pbar_.update_p_bar(1, _cand_)

    def _init_base_positioner(self, _cand_, positioner=None):
        if positioner:
            _p_ = positioner(self._opt_args_)
        else:
            _p_ = BasePositioner(self._opt_args_)

        _p_.pos_new = _cand_.pos_best
        _p_.score_new = _cand_.score_best

        return _p_
