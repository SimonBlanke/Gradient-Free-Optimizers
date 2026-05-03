# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._data.data_accessor import DataAccessor


def _stdout_supports_unicode() -> bool:
    """Check whether stdout can encode Unicode box-drawing characters."""
    encoding = getattr(sys.stdout, "encoding", None) or ""
    try:
        "┌─┐│└┘".encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


_USE_UNICODE = _stdout_supports_unicode()

_TL = "┌" if _USE_UNICODE else "+"
_TR = "┐" if _USE_UNICODE else "+"
_BL = "└" if _USE_UNICODE else "+"
_BR = "┘" if _USE_UNICODE else "+"
_H = "─" if _USE_UNICODE else "-"
_V = "│" if _USE_UNICODE else "|"


def _format_box(title: str, lines: list[str], inner_width: int | None = None) -> str:
    """Format content lines into a box with a title."""
    if inner_width is None:
        inner_width = max((len(line) for line in lines if line), default=0)
        inner_width = max(inner_width + 2, len(title) + 5)

    title_dashes = inner_width - len(title) - 3
    top = f"{_TL}{_H} {title} " + _H * title_dashes + _TR
    bottom = _BL + _H * inner_width + _BR

    result = [top]
    for line in lines:
        result.append(f"{_V}{(line or '').ljust(inner_width)}{_V}")
    result.append(bottom)
    return "\n".join(result)


def _format_throughput(data: DataAccessor) -> str:
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


def _build_summary_entries(data: DataAccessor, sections: set[str]) -> list:
    """Build summary entries filtered by active verbosity sections.

    None = blank separator line.
    str = header line without value alignment.
    tuple with _SECTION_MARKER = inline section divider.
    tuple = (prefix, label, value) for aligned output.
    """
    import math

    tracker = data._tracker

    entries = [
        None,
        ("  ", "Objective:", tracker.objective_name),
        ("  ", "Optimizer:", tracker.optimizer_name),
        ("  ", "Random state:", str(tracker.random_seed)),
    ]

    if "print_results" in sections:
        entries.append(_section("Results"))
        entries.append(("  ", "Best score:", str(data._best_score)))
        entries.append(("  ", "Best iter:", str(data.best_iteration)))

        para = data._best_para
        if para:
            entries.append("  Best parameters:")
            for k, v in para.items():
                entries.append(("    ", f"{k}:", str(v)))

    if "print_search_stats" in sections:
        n = data.n_iter
        init_pct = data.n_init / n * 100 if n else 0
        opt_pct = data.n_optimization / n * 100 if n else 0
        acc_pct = data.acceptance_rate * 100

        entries.append(_section("Search"))
        entries.append(("  ", "Iterations:", str(n)))
        entries.append(("    ", "Initialization:", f"{data.n_init} ({init_pct:.1f}%)"))
        entries.append(
            ("    ", "Optimization:", f"{data.n_optimization} ({opt_pct:.1f}%)")
        )
        entries.append(("  ", "Improvements:", str(data.n_score_improvements)))
        entries.append(
            (
                "  ",
                "Accepted:",
                f"{data.n_accepted}/{data.n_proposed} ({acc_pct:.1f}%)",
            )
        )

        if data.last_improvement >= 0:
            entries.append(("  ", "Last improvement:", f"iter {data.last_improvement}"))

        plateau_len = data.longest_plateau[0]
        if plateau_len > 1:
            entries.append(("  ", "Longest plateau:", f"{plateau_len} iterations"))

        if data.n_invalid > 0:
            pct = data.n_invalid / data.n_iter * 100
            entries.append(
                ("  ", "Invalid evals:", f"{data.n_invalid}/{data.n_iter} ({pct:.1f}%)")
            )

    if "print_statistics" in sections:
        entries.append(_section("Score Statistics"))
        if not math.isnan(data.score_mean):
            entries.extend(
                [
                    ("  ", "Min:", f"{data.score_min:.4g}"),
                    ("  ", "Max:", f"{data.score_max:.4g}"),
                    ("  ", "Mean:", f"{data.score_mean:.4g}"),
                    ("  ", "Std:", f"{data.score_std:.4g}"),
                ]
            )

    if "print_times" in sections:
        entries.extend(
            [
                _section("Timing"),
                (
                    "  ",
                    "Evaluation time:",
                    f"{data.eval_time:.3f}s ({data.eval_pct:.1f}%)",
                ),
                (
                    "  ",
                    "Optimization time:",
                    f"{data.overhead_time:.3f}s ({data.overhead_pct:.1f}%)",
                ),
                ("  ", "Iteration time:", f"{data.total_time:.3f}s"),
                ("  ", "Throughput:", _format_throughput(data)),
            ]
        )

    entries.append(None)
    return entries


def print_summary(data: DataAccessor, sections: set[str]) -> None:
    """Print a formatted summary box of the search results."""
    entries = _build_summary_entries(data, sections)

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
            prefix = f"  {_H}{_H} {name} "
            remaining = inner_width - len(prefix) - 2
            lines[line_idx] = prefix + _H * max(remaining, 3)
            line_idx += 1
        else:
            line_idx += 1

    print(_format_box("Search Summary", lines, inner_width))
