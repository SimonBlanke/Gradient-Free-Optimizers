# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np


class SearchTracker:
    def __init__(self):
        super().__init__()

        self._pos_new = None
        self._score_new = -np.inf

        self._pos_current = None
        self._score_current = -np.inf

        self._pos_best = None
        self._score_best = -np.inf

        self.pos_new_list = []
        self.score_new_list = []

        self.pos_current_list = []
        self.score_current_list = []

        self.pos_best_list = []
        self.score_best_list = []

        # non inf and non nan
        self.positions_valid = []
        self.scores_valid = []

    ##################### evaluate #####################

    def _eval2current(self, pos, score):
        if score > self.score_current:
            self.score_current = score
            self.pos_current = pos

    def _eval2best(self, pos, score):
        if score > self.score_best:
            self.score_best = score
            self.pos_best = pos

    def _evaluate_new2current(self, score_new):
        if score_new > self.score_current:
            self.score_current = score_new
            self.pos_current = self.pos_new

    def _evaluate_current2best(self):
        if self.score_current > self.score_best:
            self.score_best = self.score_current
            self.pos_best = self.pos_current

    def _current2best(self):
        self.score_best = self.score_current
        self.pos_best = self.pos_current

    def _new2current(self):
        self.score_current = self.score_new
        self.pos_current = self.pos_new

    ##################### new #####################

    @property
    def pos_new(self):
        return self._pos_new

    @pos_new.setter
    def pos_new(self, pos):
        self.pos_new_list.append(pos)
        self._pos_new = pos

    @property
    def score_new(self):
        return self._score_new

    @score_new.setter
    def score_new(self, score):
        self.score_new_list.append(score)
        self._score_new = score

        if ~np.isinf(score) and ~np.isnan(score):
            self.positions_valid.append(self.pos_new)
            self.scores_valid.append(self.score_new)

        # print("self.scores_valid", self.scores_valid)
        # print("self.score_new_list", self.score_new_list)

    ##################### current #####################

    @property
    def pos_current(self):
        return self._pos_current

    @pos_current.setter
    def pos_current(self, pos):
        self.pos_current_list.append(pos)
        self._pos_current = pos

    @property
    def score_current(self):
        return self._score_current

    @score_current.setter
    def score_current(self, score):
        self.score_current_list.append(score)
        self._score_current = score

    ##################### best #####################

    @property
    def pos_best(self):
        return self._pos_best

    @pos_best.setter
    def pos_best(self, pos):
        self.pos_best_list.append(pos)
        self._pos_best = pos

    @property
    def score_best(self):
        return self._score_best

    @score_best.setter
    def score_best(self, score):
        self.score_best_list.append(score)
        self._score_best = score
