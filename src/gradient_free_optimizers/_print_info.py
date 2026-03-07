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


def _format_box(title: str, lines: list[str], inner_width: int | None = None) -> str:
    """Format content lines into a Unicode box with a title."""
    if inner_width is None:
        inner_width = max((len(line) for line in lines if line), default=0)
        inner_width = max(inner_width + 2, len(title) + 5)

    title_dashes = inner_width - len(title) - 3
    top = f"┌─ {title} " + "─" * title_dashes + "┐"
    bottom = "└" + "─" * inner_width + "┘"

    result = [top]
    for line in lines:
        result.append(f"│{(line or '').ljust(inner_width)}│")
    result.append(bottom)
    return "\n".join(result)


def _format_throughput(data: SearchData) -> str:
    """Format throughput as iter/sec or sec/iter depending on speed."""
    if data.total_time == 0:
        return "too fast to measure"
    if data.throughput >= 1:
        return f"{data.throughput:.2f} iter/sec"
    return f"{data.avg_iter_time:.2f} sec/iter"


_SECTION_MARKER = "__section__"


def _section(name: str):
    """Create a section divider entry."""
    return (_SECTION_MARKER, name)


def _build_summary_entries(data: SearchData) -> list:
    """Build all summary entries as (prefix, label, value) triples.

    None = blank separator line.
    str = header line without value alignment.
    tuple with _SECTION_MARKER = inline section divider.
    tuple = (prefix, label, value) for aligned output.
    """
    import math

    tracker = data._tracker
    n = data.n_iter
    init_pct = data.n_init / n * 100 if n else 0
    opt_pct = data.n_optimization / n * 100 if n else 0

    entries = [
        None,
        ("  ", "Objective:", tracker.objective_name),
        ("  ", "Optimizer:", tracker.optimizer_name),
        ("  ", "Random state:", str(tracker.random_seed)),
        _section("Results"),
        ("  ", "Best score:", str(data.best_score)),
        ("  ", "Best iter:", str(data.best_iteration)),
    ]

    para = data.best_para
    if para:
        entries.append("  Best parameters:")
        for k, v in para.items():
            entries.append(("    ", f"{k}:", str(v)))

    acc_pct = data.acceptance_rate * 100
    _section_search = [
        _section("Search"),
        ("  ", "Iterations:", str(n)),
        ("    ", "Initialization:", f"{data.n_init} ({init_pct:.1f}%)"),
        ("    ", "Optimization:", f"{data.n_optimization} ({opt_pct:.1f}%)"),
        ("  ", "Improvements:", str(data.n_score_improvements)),
        (
            "  ",
            "Accepted:",
            f"{data.n_accepted}/{data.n_proposed} ({acc_pct:.1f}%)",
        ),
    ]

    if data.last_improvement >= 0:
        _section_search.append(
            ("  ", "Last improvement:", f"iter {data.last_improvement}")
        )

    plateau_len = data.longest_plateau[0]
    if plateau_len > 1:
        _section_search.append(("  ", "Longest plateau:", f"{plateau_len} iterations"))

    if data.n_invalid > 0:
        pct = data.n_invalid / data.n_iter * 100
        _section_search.append(
            ("  ", "Invalid evals:", f"{data.n_invalid}/{data.n_iter} ({pct:.1f}%)")
        )

    entries.extend(_section_search)

    _section_scores = [_section("Score Statistics")]
    if not math.isnan(data.score_mean):
        _section_scores.extend(
            [
                ("  ", "Min:", f"{data.score_min:.4g}"),
                ("  ", "Max:", f"{data.score_max:.4g}"),
                ("  ", "Mean:", f"{data.score_mean:.4g}"),
                ("  ", "Std:", f"{data.score_std:.4g}"),
            ]
        )

    entries.extend(_section_scores)

    entries.extend(
        [
            _section("Timing"),
            ("  ", "Evaluation time:", f"{data.eval_time:.3f}s ({data.eval_pct:.1f}%)"),
            (
                "  ",
                "Optimization time:",
                f"{data.overhead_time:.3f}s ({data.overhead_pct:.1f}%)",
            ),
            ("  ", "Iteration time:", f"{data.total_time:.3f}s"),
            ("  ", "Throughput:", _format_throughput(data)),
            None,
        ]
    )

    return entries


def print_summary(data: SearchData) -> None:
    """Print a formatted summary box of the search results."""
    entries = _build_summary_entries(data)

    col = max(
        len(prefix) + len(label)
        for e in entries
        if isinstance(e, tuple) and len(e) == 3
        for prefix, label, _ in [e]
    )

    lines = []
    for e in entries:
        if e is None:
            lines.append("")
        elif isinstance(e, tuple) and len(e) == 2 and e[0] == _SECTION_MARKER:
            lines.append("")  # blank line before section title
            lines.append(None)  # placeholder for divider
        elif isinstance(e, str):
            lines.append(e)
        else:
            prefix, label, value = e
            padding = col - len(prefix) - len(label)
            lines.append(f"{prefix}{label}{' ' * padding}  {value}")

    content_width = max((len(line) for line in lines if line), default=0)
    inner_width = max(content_width + 2, len("Search Summary") + 5)

    line_idx = 0
    for e in entries:
        if e is None:
            line_idx += 1
        elif isinstance(e, tuple) and len(e) == 2 and e[0] == _SECTION_MARKER:
            line_idx += 1  # skip the blank line
            name = e[1]
            prefix = f"  ── {name} "
            remaining = inner_width - len(prefix) - 2
            lines[line_idx] = prefix + "─" * max(remaining, 3)
            line_idx += 1
        else:
            line_idx += 1

    print(_format_box("Search Summary", lines, inner_width))
