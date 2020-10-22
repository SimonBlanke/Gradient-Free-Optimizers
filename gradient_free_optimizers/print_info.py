# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


def print_info(
    verbosity,
    objective_function,
    score_best,
    para_best,
    eval_time,
    iter_time,
    n_iter,
):
    indent = "  "

    if verbosity["print_results"] is True:
        print("\nResults: '{}'".format(objective_function.__name__), " ")
        print(indent, "Best score:", score_best, " ")
        print(indent, "Best parameter:")
        for key in para_best.keys():
            print(indent, indent, "'{}'".format(key), para_best[key], " ")

    if verbosity["print_times"] is True:
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
