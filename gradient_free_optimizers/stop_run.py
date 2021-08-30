# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import numpy as np


def time_exceeded(start_time, max_time):
    run_time = time.time() - start_time
    return max_time and run_time > max_time


def score_exceeded(score_best, max_score):
    return max_score and score_best >= max_score


def no_change(score_new_list, early_stopping):
    if "n_iter_no_change" not in early_stopping:
        print(
            "Warning n_iter_no_change-parameter must be set in order for early stopping to work"
        )
        return False

    n_iter_no_change = early_stopping["n_iter_no_change"]
    if len(score_new_list) <= n_iter_no_change:
        return False

    scores_np = np.array(score_new_list)

    max_score = max(score_new_list)
    max_index = np.argmax(scores_np)
    length_pos = len(score_new_list)

    diff = length_pos - max_index

    if diff > n_iter_no_change:
        return True

    first_n = length_pos - n_iter_no_change
    scores_first_n = score_new_list[:first_n]

    max_first_n = max(scores_first_n)

    if "tol_abs" in early_stopping and early_stopping["tol_abs"] is not None:
        tol_abs = early_stopping["tol_abs"]

        if abs(max_first_n - max_score) < tol_abs:
            return True

    if "tol_rel" in early_stopping and early_stopping["tol_rel"] is not None:
        tol_rel = early_stopping["tol_rel"]

        percent_imp = ((max_score - max_first_n) / abs(max_first_n)) * 100
        if percent_imp < tol_rel:
            return True


class StopRun:
    def __init__(self, max_time, max_score, early_stopping):
        self.max_time = max_time
        self.max_score = max_score
        self.early_stopping = early_stopping

    def check(self, start_time, score_best, score_new_list):
        if self.max_time and time_exceeded(start_time, self.max_time):
            return True
        elif self.max_score and score_exceeded(score_best, self.max_score):
            return True
        elif self.early_stopping and no_change(score_new_list, self.early_stopping):
            return True
