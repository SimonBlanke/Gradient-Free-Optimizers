"""Generate SMBO loop diagram as SVG using matplotlib.

Produces a cyclic workflow diagram showing the six steps of
Sequential Model-Based Optimization, arranged in an oval layout.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams["svg.fonttype"] = "none"

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "..", "source", "_static", "diagrams")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "smbo_loop.svg")

# -- Colors ------------------------------------------------------------------

ORANGE = "#dd6b20"
TEXT_COLOR = "#2d3748"
BACKGROUND = "#f7fafc"
BORDER = "#e2e8f0"
INIT_COLOR = "#718096"
LOOP_ARROW_COLOR = "#805ad5"

# -- Step definitions ---------------------------------------------------------
# Each tuple: (label, description, is_entry_point)

STEPS = [
    ("Initialize", "Evaluate random points", True),
    ("Fit Model", "Train surrogate on observations", False),
    ("Predict", "Score all candidate points", False),
    ("Acquire", "Balance exploration vs exploitation", False),
    ("Evaluate", "Run objective function", False),
    ("Update", "Add new observation to dataset", False),
]

# -- Layout geometry ----------------------------------------------------------

BOX_WIDTH = 2.4
BOX_HEIGHT = 1.1
CORNER_RADIUS = 0.15


def _oval_positions(n_steps, cx, cy, rx, ry):
    """Return (x, y) center positions arranged on an oval.

    Steps are distributed starting from the top and going clockwise.
    """
    positions = []
    for i in range(n_steps):
        # Start at top (-pi/2) and go clockwise (positive angle direction
        # maps to clockwise when y-axis is inverted in display coords,
        # but we work in data coords so we go counter-clockwise in math
        # terms to get clockwise visual appearance).
        angle = -np.pi / 2 + 2 * np.pi * i / n_steps
        x = cx + rx * np.cos(angle)
        y = cy + ry * np.sin(angle)
        positions.append((x, y))
    return positions


def _draw_box(ax, cx, cy, label, description, number, is_entry):
    """Draw a single step box with number, label, and description."""
    x = cx - BOX_WIDTH / 2
    y = cy - BOX_HEIGHT / 2

    fill_color = "#f0f0f0" if is_entry else "white"
    edge_color = INIT_COLOR if is_entry else ORANGE

    box = FancyBboxPatch(
        (x, y),
        BOX_WIDTH,
        BOX_HEIGHT,
        boxstyle=f"round,pad={CORNER_RADIUS}",
        facecolor=fill_color,
        edgecolor=edge_color,
        linewidth=2.0,
    )
    ax.add_patch(box)

    number_color = INIT_COLOR if is_entry else ORANGE

    # Step number (small, top-left area of box)
    ax.text(
        cx - BOX_WIDTH / 2 + 0.22,
        cy + BOX_HEIGHT / 2 - 0.22,
        str(number),
        fontsize=9,
        fontweight="bold",
        color=number_color,
        fontfamily="sans-serif",
        ha="center",
        va="top",
    )

    # Label
    ax.text(
        cx,
        cy + 0.12,
        label,
        fontsize=12,
        fontweight="bold",
        color=TEXT_COLOR,
        fontfamily="sans-serif",
        ha="center",
        va="center",
    )

    # Description
    ax.text(
        cx,
        cy - 0.22,
        description,
        fontsize=8.5,
        color="#718096",
        fontfamily="sans-serif",
        ha="center",
        va="center",
    )


def _edge_point(cx, cy, target_x, target_y):
    """Find the point on the box edge closest to the target direction.

    Computes where a line from (cx, cy) toward (target_x, target_y)
    intersects the box boundary (approximated as a rectangle).
    """
    dx = target_x - cx
    dy = target_y - cy

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return cx, cy

    half_w = BOX_WIDTH / 2 + CORNER_RADIUS
    half_h = BOX_HEIGHT / 2 + CORNER_RADIUS

    # Scale factor to reach the box edge
    scale_x = half_w / abs(dx) if abs(dx) > 1e-9 else float("inf")
    scale_y = half_h / abs(dy) if abs(dy) > 1e-9 else float("inf")
    scale = min(scale_x, scale_y)

    return cx + dx * scale, cy + dy * scale


def _draw_arrow(ax, start_pos, end_pos, is_loop_back=False):
    """Draw an arrow between two step boxes."""
    sx, sy = _edge_point(start_pos[0], start_pos[1], end_pos[0], end_pos[1])
    ex, ey = _edge_point(end_pos[0], end_pos[1], start_pos[0], start_pos[1])

    linestyle = "dashed" if is_loop_back else "solid"
    color = LOOP_ARROW_COLOR if is_loop_back else ORANGE
    linewidth = 1.8 if is_loop_back else 1.5

    arrow = FancyArrowPatch(
        (sx, sy),
        (ex, ey),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)

    # Label on the loop-back arrow
    if is_loop_back:
        mid_x = (sx + ex) / 2
        mid_y = (sy + ey) / 2
        ax.text(
            mid_x - 0.45,
            mid_y,
            "repeat",
            fontsize=8,
            fontstyle="italic",
            color=LOOP_ARROW_COLOR,
            fontfamily="sans-serif",
            ha="center",
            va="center",
            rotation=0,
        )


def generate():
    """Generate the SMBO loop diagram and save as SVG."""
    fig, ax = plt.subplots(figsize=(10, 8))

    center_x, center_y = 5.0, 4.0
    radius_x, radius_y = 3.2, 2.6

    positions = _oval_positions(len(STEPS), center_x, center_y, radius_x, radius_y)

    # Draw arrows first so boxes render on top
    for i in range(len(STEPS)):
        next_i = (i + 1) % len(STEPS)

        # Arrow from step 6 (Update) back to step 2 (Fit Model) is the loop-back
        is_loop_back = i == len(STEPS) - 1 and next_i == 0

        if is_loop_back:
            # The loop-back goes from Update (index 5) to Fit Model (index 1),
            # skipping Initialize
            _draw_arrow(ax, positions[5], positions[1], is_loop_back=True)
        else:
            _draw_arrow(ax, positions[i], positions[next_i])

    # Draw boxes
    for i, (label, description, is_entry) in enumerate(STEPS):
        cx, cy = positions[i]
        _draw_box(ax, cx, cy, label, description, i + 1, is_entry)

    # Title
    ax.text(
        center_x,
        center_y,
        "SMBO Loop",
        fontsize=16,
        fontweight="bold",
        color=TEXT_COLOR,
        fontfamily="sans-serif",
        ha="center",
        va="center",
        alpha=0.4,
    )

    # Diagram border
    border = FancyBboxPatch(
        (0.3, 0.3),
        9.4,
        7.4,
        boxstyle="round,pad=0.1",
        facecolor="none",
        edgecolor=BORDER,
        linewidth=1.5,
    )
    ax.add_patch(border)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(OUTPUT_PATH, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)

    print(f"Saved SMBO loop diagram to {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    generate()
