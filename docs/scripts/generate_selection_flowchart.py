"""Generate an algorithm selection flowchart as SVG using matplotlib.

The flowchart is a decision tree that guides users toward the right
optimization algorithm based on their problem characteristics.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.use("svg")
matplotlib.rcParams["svg.fonttype"] = "none"

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "..", "source", "_static", "diagrams")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "algorithm_selection_flowchart.svg")

# -- Color palette ----------------------------------------------------------

COLORS = {
    "local": "#e53e3e",
    "global": "#38a169",
    "population": "#805ad5",
    "smbo": "#dd6b20",
    "text": "#2d3748",
    "background": "#f7fafc",
    "border": "#e2e8f0",
    "decision_fill": "#edf2f7",
    "yes": "#38a169",
    "no": "#e53e3e",
}

# Lighter fills derived from category colors (used for category boxes)
CATEGORY_FILLS = {
    "local": "#fed7d7",
    "global": "#c6f6d5",
    "population": "#e9d8fd",
    "smbo": "#feebc8",
}

FONT_FAMILY = "sans-serif"
FONT_SIZE_MAIN = 10
FONT_SIZE_DETAIL = 8
FONT_SIZE_LABEL = 8.5


def _rounded_box(
    ax,
    x,
    y,
    width,
    height,
    text,
    fill,
    edge_color,
    font_size=FONT_SIZE_MAIN,
    font_weight="normal",
    text_color=None,
    pad=0.15,
):
    """Draw a rounded rectangle with centered text."""
    text_color = text_color or COLORS["text"]
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle=f"round,pad={pad}",
        facecolor=fill,
        edgecolor=edge_color,
        linewidth=1.2,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=font_size,
        fontfamily=FONT_FAMILY,
        fontweight=font_weight,
        color=text_color,
        zorder=3,
    )
    return box


def _arrow(ax, start_xy, end_xy, color=COLORS["border"]):
    """Draw a connecting arrow between two points."""
    arrow = FancyArrowPatch(
        start_xy,
        end_xy,
        arrowstyle="-|>",
        color=color,
        linewidth=1.4,
        mutation_scale=12,
        zorder=1,
    )
    ax.add_patch(arrow)
    return arrow


def _edge_label(ax, x, y, text, color):
    """Draw a small Yes/No label next to an arrow."""
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=FONT_SIZE_LABEL,
        fontfamily=FONT_FAMILY,
        fontweight="bold",
        color=color,
        zorder=4,
    )


def _algorithm_row(ax, x, y, condition, algorithm, width=2.4):
    """Draw a single algorithm recommendation row (condition -> algorithm)."""
    cond_width = width * 0.55
    algo_width = width * 0.40
    gap = width * 0.05
    row_height = 0.32

    # Condition text (left-aligned)
    ax.text(
        x - width / 2 + 0.08,
        y,
        condition,
        ha="left",
        va="center",
        fontsize=FONT_SIZE_DETAIL,
        fontfamily=FONT_FAMILY,
        color=COLORS["text"],
        zorder=3,
    )

    # Arrow
    arrow_start_x = x - width / 2 + cond_width
    arrow_end_x = arrow_start_x + gap
    ax.annotate(
        "",
        xy=(arrow_end_x, y),
        xytext=(arrow_start_x, y),
        arrowprops=dict(arrowstyle="->", color=COLORS["border"], lw=0.8),
        zorder=3,
    )

    # Algorithm box (right side)
    algo_center_x = arrow_end_x + algo_width / 2
    algo_box = FancyBboxPatch(
        (algo_center_x - algo_width / 2, y - row_height / 2),
        algo_width,
        row_height,
        boxstyle="round,pad=0.06",
        facecolor="white",
        edgecolor=COLORS["border"],
        linewidth=0.8,
        zorder=2,
    )
    ax.add_patch(algo_box)
    ax.text(
        algo_center_x,
        y,
        algorithm,
        ha="center",
        va="center",
        fontsize=FONT_SIZE_DETAIL,
        fontfamily=FONT_FAMILY,
        fontweight="bold",
        color=COLORS["text"],
        zorder=3,
    )


def _category_block(ax, x, y, title, category_key, recommendations):
    """Draw a category answer box with algorithm recommendation rows inside.

    Parameters
    ----------
    recommendations : list of (condition_str, algorithm_str)
    """
    block_width = 3.0
    header_height = 0.45
    row_height = 0.40
    row_count = len(recommendations)
    body_height = row_height * row_count + 0.15
    total_height = header_height + body_height

    fill = CATEGORY_FILLS[category_key]
    edge = COLORS[category_key]

    # Outer container
    container = FancyBboxPatch(
        (x - block_width / 2, y - total_height / 2),
        block_width,
        total_height,
        boxstyle="round,pad=0.12",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.6,
        zorder=2,
    )
    ax.add_patch(container)

    # Header text
    header_y = y + total_height / 2 - header_height / 2 - 0.05
    ax.text(
        x,
        header_y,
        title,
        ha="center",
        va="center",
        fontsize=FONT_SIZE_MAIN,
        fontfamily=FONT_FAMILY,
        fontweight="bold",
        color=edge,
        zorder=3,
    )

    # Separator line
    sep_y = y + total_height / 2 - header_height
    ax.plot(
        [x - block_width / 2 + 0.15, x + block_width / 2 - 0.15],
        [sep_y, sep_y],
        color=edge,
        linewidth=0.6,
        alpha=0.5,
        zorder=3,
    )

    # Recommendation rows
    first_row_y = sep_y - 0.25
    for i, (condition, algorithm) in enumerate(recommendations):
        row_y = first_row_y - i * row_height
        _algorithm_row(ax, x, row_y, condition, algorithm, width=block_width - 0.3)

    # Return connection points (top center, bottom center)
    top_y = y + total_height / 2
    bottom_y = y - total_height / 2
    return (x, top_y), (x, bottom_y)


def generate_flowchart():
    """Build and save the algorithm selection flowchart."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    # -- Layout coordinates --------------------------------------------------
    # Decision nodes run vertically on the left side (x=2.0).
    # Category blocks sit to the right of each decision.

    decision_x = 2.0
    decision_w = 3.6
    decision_h = 0.7

    block_x = 8.5

    # Vertical positions (top to bottom)
    q1_y = 7.5
    q2_y = 5.0
    q3_y = 2.5

    # -- Title ---------------------------------------------------------------
    ax.text(
        6.5,
        8.8,
        "Algorithm Selection Guide",
        ha="center",
        va="center",
        fontsize=14,
        fontfamily=FONT_FAMILY,
        fontweight="bold",
        color=COLORS["text"],
    )

    # -- Q1: Expensive objective? -------------------------------------------
    _rounded_box(
        ax,
        decision_x,
        q1_y,
        decision_w,
        decision_h,
        "Is your objective function\nexpensive (> 1 s per eval)?",
        fill=COLORS["decision_fill"],
        edge_color=COLORS["border"],
        font_size=FONT_SIZE_MAIN,
        font_weight="bold",
    )

    smbo_top, _ = _category_block(
        ax,
        block_x,
        q1_y,
        "Use SMBO",
        "smbo",
        [
            ("Continuous params?", "Bayesian"),
            ("Many categoricals?", "TPE"),
            ("Many iterations?", "Forest"),
        ],
    )

    # Arrow Q1 -> SMBO (Yes)
    _arrow(
        ax, (decision_x + decision_w / 2, q1_y), (block_x - 1.5, q1_y), COLORS["smbo"]
    )
    _edge_label(
        ax,
        (decision_x + decision_w / 2 + block_x - 1.5) / 2,
        q1_y + 0.18,
        "Yes",
        COLORS["yes"],
    )

    # Arrow Q1 -> Q2 (No)
    _arrow(
        ax,
        (decision_x, q1_y - decision_h / 2),
        (decision_x, q2_y + decision_h / 2),
        COLORS["border"],
    )
    _edge_label(
        ax,
        decision_x - 0.3,
        (q1_y - decision_h / 2 + q2_y + decision_h / 2) / 2,
        "No",
        COLORS["no"],
    )

    # -- Q2: Good starting point? -------------------------------------------
    _rounded_box(
        ax,
        decision_x,
        q2_y,
        decision_w,
        decision_h,
        "Do you have a good\nstarting point?",
        fill=COLORS["decision_fill"],
        edge_color=COLORS["border"],
        font_size=FONT_SIZE_MAIN,
        font_weight="bold",
    )

    local_top, _ = _category_block(
        ax,
        block_x,
        q2_y,
        "Use Local Search",
        "local",
        [
            ("Smooth function?", "Hill Climbing"),
            ("Multiple local optima?", "Sim. Annealing"),
        ],
    )

    # Arrow Q2 -> Local (Yes)
    _arrow(
        ax, (decision_x + decision_w / 2, q2_y), (block_x - 1.5, q2_y), COLORS["local"]
    )
    _edge_label(
        ax,
        (decision_x + decision_w / 2 + block_x - 1.5) / 2,
        q2_y + 0.18,
        "Yes",
        COLORS["yes"],
    )

    # Arrow Q2 -> Q3 (No)
    _arrow(
        ax,
        (decision_x, q2_y - decision_h / 2),
        (decision_x, q3_y + decision_h / 2),
        COLORS["border"],
    )
    _edge_label(
        ax,
        decision_x - 0.3,
        (q2_y - decision_h / 2 + q3_y + decision_h / 2) / 2,
        "No",
        COLORS["no"],
    )

    # -- Q3: Parallel evaluation? -------------------------------------------
    _rounded_box(
        ax,
        decision_x,
        q3_y,
        decision_w,
        decision_h,
        "Can you evaluate\nin parallel?",
        fill=COLORS["decision_fill"],
        edge_color=COLORS["border"],
        font_size=FONT_SIZE_MAIN,
        font_weight="bold",
    )

    pop_top, _ = _category_block(
        ax,
        block_x,
        q3_y + 0.6,
        "Use Population-Based",
        "population",
        [
            ("Continuous?", "Particle Swarm"),
            ("Discrete?", "Genetic Algorithm"),
        ],
    )

    global_top, _ = _category_block(
        ax,
        block_x,
        q3_y - 1.3,
        "Use Global Search",
        "global",
        [
            ("Need baseline?", "Random Search"),
            ("Small space?", "Grid Search"),
            ("Unknown landscape?", "DIRECT"),
        ],
    )

    # Arrow Q3 -> Population (Yes)
    _arrow(
        ax,
        (decision_x + decision_w / 2, q3_y + 0.1),
        (block_x - 1.5, q3_y + 0.6),
        COLORS["population"],
    )
    _edge_label(
        ax,
        (decision_x + decision_w / 2 + block_x - 1.5) / 2 - 0.1,
        q3_y + 0.55,
        "Yes",
        COLORS["yes"],
    )

    # Arrow Q3 -> Global (No)
    _arrow(
        ax,
        (decision_x + decision_w / 2, q3_y - 0.1),
        (block_x - 1.5, q3_y - 1.3),
        COLORS["global"],
    )
    _edge_label(
        ax,
        (decision_x + decision_w / 2 + block_x - 1.5) / 2 - 0.1,
        q3_y - 0.95,
        "No",
        COLORS["no"],
    )

    # -- Save ----------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(
        OUTPUT_FILE, format="svg", bbox_inches="tight", dpi=150, transparent=True
    )
    plt.close(fig)
    print(f"Saved: {os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    generate_flowchart()
