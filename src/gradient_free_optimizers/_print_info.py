# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._data.search_data import SearchData


indent = "  "


def _print_times(eval_time, iter_time, n_iter):
    # Guard against division by zero (can happen on Windows with low timer resolution)
    if iter_time == 0:
        print(indent, "Evaluation time   :", eval_time, "sec")
        print(indent, "Optimization time :", 0, "sec")
        print(
            indent,
            "Iteration time    :",
            iter_time,
            "sec",
            indent,
            "[too fast to measure]",
        )
        print(" ")
        return

    opt_time = iter_time - eval_time
    iterPerSec = n_iter / iter_time

    print(
        indent,
        "Evaluation time   :",
        eval_time,
        "sec",
        indent,
        f"[{round(eval_time / iter_time * 100, 2)} %]",
    )
    print(
        indent,
        "Optimization time :",
        opt_time,
        "sec",
        indent,
        f"[{round(opt_time / iter_time * 100, 2)} %]",
    )
    if iterPerSec >= 1:
        print(
            indent,
            "Iteration time    :",
            iter_time,
            "sec",
            indent,
            f"[{round(iterPerSec, 2)} iter/sec]",
        )
    else:
        secPerIter = iter_time / n_iter
        print(
            indent,
            "Iteration time    :",
            iter_time,
            "sec",
            indent,
            f"[{round(secPerIter, 2)} sec/iter]",
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


def _print_results(objective_function, score_best, para_best, random_seed):
    print(f"\nResults: '{objective_function.__name__}'", " ")
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
                f"'{para_key}'",
                f"{added_spaces}:",
                para_best[para_key],
                " ",
            )
    print(" ")
    print(indent, "Random seed:", random_seed, " ")
    print(" ")


def print_info(
    verbosity,
    objective_function,
    score_best,
    para_best,
    eval_times,
    iter_times,
    n_iter,
    random_seed,
):
    eval_time = sum(eval_times)
    iter_time = sum(iter_times)

    if "print_results" in verbosity:
        _print_results(objective_function, score_best, para_best, random_seed)

    if "print_times" in verbosity:
        _print_times(eval_time, iter_time, n_iter)


def _format_box(title: str, lines: list[str]) -> str:
    """Format content lines into a Unicode box with a title."""
    inner_width = max((len(line) for line in lines if line), default=0)
    inner_width = max(inner_width + 2, len(title) + 5)

    title_dashes = inner_width - len(title) - 3
    top = f"┌─ {title} " + "─" * title_dashes + "┐"
    bottom = "└" + "─" * inner_width + "┘"

    result = [top]
    for line in lines:
        result.append(f"│{line.ljust(inner_width)}│")
    result.append(bottom)
    return "\n".join(result)


def _format_para_lines(para: dict, prefix: str = "    ") -> list[str]:
    """Format parameters as aligned lines, one per parameter."""
    if not para:
        return [f"{prefix}None"]
    names = list(para.keys())
    max_len = max(len(n) for n in names)
    return [f"{prefix}{n}:{' ' * (max_len - len(n))}  {para[n]}" for n in names]


def _format_throughput(data: SearchData) -> str:
    """Format throughput as iter/sec or sec/iter depending on speed."""
    if data.total_time == 0:
        return "too fast to measure"
    if data.throughput >= 1:
        return f"{data.throughput:.2f} iter/sec"
    return f"{data.avg_iter_time:.2f} sec/iter"


def print_summary(data: SearchData) -> None:
    """Print a formatted summary box of the search results."""
    tracker = data._tracker

    lines = [
        "",
        f"  Objective:  {tracker.objective_name}",
        f"  Optimizer:  {tracker.optimizer_name}",
        f"  Seed:       {tracker.random_seed}",
        "",
        f"  Best score: {data.best_score}",
        f"  Best iter:  {data.best_iteration}",
        "  Best parameters:",
    ]
    lines.extend(_format_para_lines(data.best_para))

    n = data.n_iter
    init_pct = data.n_init / n * 100 if n else 0
    opt_pct = data.n_optimization / n * 100 if n else 0

    lines.append("")
    lines.append(f"  Iterations:      {n}")
    lines.append(f"    Initialization:  {data.n_init} ({init_pct:.1f}%)")
    lines.append(f"    Optimization:    {data.n_optimization} ({opt_pct:.1f}%)")
    lines.append(f"  Improvements:    {data.n_score_improvements}")

    plateau_len, plateau_start, plateau_end = data.longest_plateau
    if plateau_len > 1:
        lines.append(
            f"  Longest plateau: {plateau_len} iterations"
            f" (iter {plateau_start}-{plateau_end})"
        )

    if data.n_invalid > 0:
        pct = data.n_invalid / data.n_iter * 100
        lines.append(f"  Invalid evals:   {data.n_invalid}/{data.n_iter} ({pct:.1f}%)")

    lines.append("")
    lines.append("  Timing:")
    lines.append(
        f"    Evaluation time:    {data.eval_time:.3f}s ({data.eval_pct:.1f}%)"
    )
    lines.append(
        f"    Optimization time:  {data.overhead_time:.3f}s ({data.overhead_pct:.1f}%)"
    )
    lines.append(f"    Iteration time:     {data.total_time:.3f}s")
    lines.append(f"    Throughput:         {_format_throughput(data)}")

    lines.append("")

    print(_format_box("Search Summary", lines))
