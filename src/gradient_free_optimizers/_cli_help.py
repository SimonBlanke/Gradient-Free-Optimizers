"""CLI help tool for Gradient-Free-Optimizers.

Provides a terminal-friendly reference for the search summary metrics
shown when ``verbosity`` includes ``print_*`` flags.
"""

from __future__ import annotations

import sys

from ._print_info import _H, _format_box

SUMMARY_REFERENCE: list[tuple[str, str]] = [
    ("", "General"),
    ("Objective", "Name of the objective function"),
    ("Optimizer", "Optimizer class used"),
    ("Random state", "Seed for reproducibility"),
    ("", "Results"),
    ("Best score", "Highest score found"),
    ("Best iter", "Iteration of best score"),
    ("Best parameters", "Parameters at best score"),
    ("", "Search"),
    ("Iterations", "Total iterations (init + opt)"),
    ("Initialization", "Initial exploration iterations"),
    ("Optimization", "Strategy-driven iterations"),
    ("Improvements", "Times best score improved"),
    ("Accepted", "Accepted / proposed positions"),
    ("Last improvement", "Iter of last score improvement"),
    ("Longest plateau", "Max iters without improvement"),
    ("Invalid evals", "Evals returning inf/nan"),
    ("", "Score Statistics"),
    ("Min / Max", "Score range (excl. inf/nan)"),
    ("Mean", "Mean score (excl. inf/nan)"),
    ("Std", "Score std dev (excl. inf/nan)"),
    ("", "Timing"),
    ("Evaluation time", "Time in objective function"),
    ("Optimization time", "Time in optimizer logic"),
    ("Iteration time", "Total wall time"),
    ("Throughput", "Iterations per second"),
]


def _build_help_lines() -> tuple[list[str], int]:
    """Build formatted help lines and return them with inner_width."""
    label_col = max(len(label) for label, _ in SUMMARY_REFERENCE if label) + 4

    lines: list[str] = [""]

    for label, description in SUMMARY_REFERENCE:
        if not label:
            lines.append("")
            lines.append(f"  {_H}{_H} {description} ")
            continue

        pad_label = label_col - len(label)
        lines.append(f"  {label}{' ' * pad_label}{description}")

    lines.append("")
    lines.append("  Verbosity flags:")
    lines.append("    print_results, print_search_stats, print_statistics, print_times")
    lines.append("  Public results API: opt.best_score, opt.best_para, opt.search_data")
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
        print("Show reference for the search summary metrics.")
        return
    print_help()


if __name__ == "__main__":
    main()
