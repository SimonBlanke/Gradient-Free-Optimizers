"""CLI help tool for Gradient-Free-Optimizers.

Provides terminal-friendly reference for summary metrics and data properties.
Invoked via ``gfo-help`` entry point or ``python -m gradient_free_optimizers help``.
"""

from __future__ import annotations

import sys

from ._print_info import _H, _format_box

SUMMARY_REFERENCE: list[tuple[str, str, str]] = [
    ("", "General", ""),
    ("Objective", "Name of the objective function", ""),
    ("Optimizer", "Optimizer class used", ""),
    ("Random state", "Seed for reproducibility", ""),
    ("", "Results", ""),
    ("Best score", "Highest score found", ".best_score"),
    ("Best iter", "Iteration of best score", ".best_iteration"),
    ("Best parameters", "Parameters at best score", ".best_para"),
    ("", "Search", ""),
    ("Iterations", "Total iterations (init + opt)", ".n_iter"),
    ("Initialization", "Initial exploration iterations", ".n_init"),
    ("Optimization", "Strategy-driven iterations", ".n_optimization"),
    ("Improvements", "Times best score improved", ".n_score_improvements"),
    ("Accepted", "Accepted / proposed positions", ".acceptance_rate"),
    ("Last improvement", "Iter of last score improvement", ".last_improvement"),
    ("Longest plateau", "Max iters without improvement", ".longest_plateau"),
    ("Invalid evals", "Evals returning inf/nan", ".n_invalid"),
    ("", "Score Statistics", ""),
    ("Min / Max", "Score range (excl. inf/nan)", ".score_min .score_max"),
    ("Mean", "Mean score (excl. inf/nan)", ".score_mean"),
    ("Std", "Score std dev (excl. inf/nan)", ".score_std"),
    ("", "Timing", ""),
    ("Evaluation time", "Time in objective function", ".eval_time"),
    ("Optimization time", "Time in optimizer logic", ".overhead_time"),
    ("Iteration time", "Total wall time", ".total_time"),
    ("Throughput", "Iterations per second", ".throughput"),
]


def _build_help_lines() -> tuple[list[str], int]:
    """Build formatted help lines and return them with inner_width."""
    label_col = max(len(label) for label, _, _ in SUMMARY_REFERENCE if label) + 4
    desc_col = max(len(desc) for _, desc, _ in SUMMARY_REFERENCE) + 4

    lines: list[str] = [""]

    for label, description, accessor in SUMMARY_REFERENCE:
        if not label:
            lines.append("")
            lines.append(f"  {_H}{_H} {description} ")
            continue

        pad_label = label_col - len(label)
        pad_desc = desc_col - len(description)

        if accessor:
            api = f"opt._data{accessor}"
        else:
            api = ""

        lines.append(f"  {label}{' ' * pad_label}{description}{' ' * pad_desc}{api}")

    lines.append("")
    lines.append("  Verbosity flags:")
    lines.append("    print_results, print_search_stats, print_statistics, print_times")
    lines.append("  API:  opt._data.<property>  |  opt._data.raw.<property>")
    lines.append("")

    content_width = max((len(line) for line in lines if line), default=0)
    inner_width = max(content_width + 2, len("gfo-help") + 5)

    for i, line in enumerate(lines):
        if line and line.lstrip().startswith(f"{_H}{_H}"):
            remaining = inner_width - len(line) - 2
            lines[i] = line + _H * max(remaining, 3)

    return lines, inner_width


def print_help() -> None:
    """Print the summary metrics reference to stdout."""
    lines, inner_width = _build_help_lines()
    print(_format_box("gfo-help", lines, inner_width))


def main() -> None:
    """Entry point for the gfo-help CLI tool."""
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: gfo-help")
        print("Show reference for search summary metrics and opt._data properties.")
        return
    print_help()


if __name__ == "__main__":
    main()
