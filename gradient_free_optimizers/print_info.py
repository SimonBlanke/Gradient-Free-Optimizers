# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

indent = "  "


def _print_times(eval_time, iter_time, n_iter):

    opt_time = iter_time - eval_time
    iterPerSec = n_iter / iter_time

    print(
        indent,
        "Evaluation time   :",
        eval_time,
        "sec",
        indent,
        "[{} %]".format(round(eval_time / iter_time * 100, 2)),
    )
    print(
        indent,
        "Optimization time :",
        opt_time,
        "sec",
        indent,
        "[{} %]".format(round(opt_time / iter_time * 100, 2)),
    )
    if iterPerSec >= 1:
        print(
            indent,
            "Iteration time    :",
            iter_time,
            "sec",
            indent,
            "[{} iter/sec]".format(round(iterPerSec, 2)),
        )
    else:
        secPerIter = iter_time / n_iter
        print(
            indent,
            "Iteration time    :",
            iter_time,
            "sec",
            indent,
            "[{} sec/iter]".format(round(secPerIter, 2)),
        )
    print(" ")


def align_para_names(para_names):
    str_lengths = [len(str_) for str_ in para_names]
    max_length = max(str_lengths)

    para_names_align = {}
    for para_name, str_length in zip(para_names, str_lengths):
        added_spaces = max_length - str_length
        para_names_align[para_name] = " " * added_spaces

    return para_names_align


def _print_results(objective_function, score_best, para_best):
    print("\nResults: '{}'".format(objective_function.__name__), " ")
    if para_best is None:
        print(indent, "Best score:", score_best, " ")
        print(indent, "Best parameter:", para_best, " ")
    else:
        para_names = list(para_best.keys())
        para_names_align = align_para_names(para_names)

        print(indent, "Best score:", score_best, " ")
        print(indent, "Best parameter:")
        for para_key in para_best.keys():
            added_spaces = para_names_align[para_key]
            print(
                indent,
                indent,
                "'{}'".format(para_key),
                "{}:".format(added_spaces),
                para_best[para_key],
                " ",
            )
    print(" ")


def print_info(
    verbosity,
    objective_function,
    score_best,
    para_best,
    eval_times,
    iter_times,
    n_iter,
):

    eval_time = np.array(eval_times).sum()
    iter_time = np.array(iter_times).sum()

    if "print_results" in verbosity:
        _print_results(objective_function, score_best, para_best)

    if "print_times" in verbosity:
        _print_times(eval_time, iter_time, n_iter)
