"""Generate algorithm flowchart SVGs for documentation.

Produces four vertical flowchart diagrams:
  - Hill Climbing
  - Simulated Annealing
  - Particle Swarm Optimization
  - Bayesian Optimization

Each diagram is saved as an SVG file in docs/source/_static/diagrams/.
"""

import os

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

matplotlib.rcParams["svg.fonttype"] = "none"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "source", "_static", "diagrams")
)

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

COLOR_LOCAL = "#e53e3e"
COLOR_GLOBAL = "#38a169"
COLOR_POPULATION = "#805ad5"
COLOR_SMBO = "#dd6b20"

COLOR_TEXT = "#2d3748"
COLOR_BACKGROUND = "#ffffff"
COLOR_BORDER = "#e2e8f0"

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

FIGURE_WIDTH = 6
FIGURE_HEIGHT = 8

BOX_WIDTH = 3.2
BOX_HEIGHT = 0.52
DIAMOND_SIZE = 0.42

FONT_SIZE = 9
FONT_FAMILY = "sans-serif"

ARROW_STYLE = dict(
    arrowstyle="-|>",
    color=COLOR_TEXT,
    lw=1.2,
    mutation_scale=12,
)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _light_fill(hex_color, alpha=0.10):
    """Return an RGBA tuple with a very light version of *hex_color*."""
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    return (r, g, b, alpha)


def new_figure():
    """Create a clean figure with the standard size and white background."""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    ax.set_xlim(0, FIGURE_WIDTH)
    ax.set_ylim(0, FIGURE_HEIGHT)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def draw_start(ax, cx, cy, label, color):
    """Draw a pill-shaped start/end node centered at (*cx*, *cy*)."""
    pill = FancyBboxPatch(
        (cx - 0.7, cy - 0.22),
        1.4,
        0.44,
        boxstyle="round,pad=0.12",
        facecolor=color,
        edgecolor=color,
        linewidth=1.4,
    )
    ax.add_patch(pill)
    ax.text(
        cx,
        cy,
        label,
        ha="center",
        va="center",
        fontsize=FONT_SIZE,
        fontfamily=FONT_FAMILY,
        fontweight="bold",
        color=COLOR_BACKGROUND,
    )


def draw_box(ax, cx, cy, text, color, width=BOX_WIDTH, height=BOX_HEIGHT):
    """Draw a rounded-rectangle process box centered at (*cx*, *cy*)."""
    box = FancyBboxPatch(
        (cx - width / 2, cy - height / 2),
        width,
        height,
        boxstyle="round,pad=0.10",
        facecolor=_light_fill(color),
        edgecolor=color,
        linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=FONT_SIZE,
        fontfamily=FONT_FAMILY,
        color=COLOR_TEXT,
    )


def draw_diamond(ax, cx, cy, text, color, size=DIAMOND_SIZE):
    """Draw a decision diamond centered at (*cx*, *cy*)."""
    diamond = mpatches.RegularPolygon(
        (cx, cy),
        numVertices=4,
        radius=size,
        orientation=0,
        facecolor="white",
        edgecolor=color,
        linewidth=1.4,
    )
    ax.add_patch(diamond)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=FONT_SIZE - 1,
        fontfamily=FONT_FAMILY,
        color=COLOR_TEXT,
    )


def draw_arrow(ax, x_start, y_start, x_end, y_end):
    """Draw a straight arrow between two points."""
    ax.annotate(
        "",
        xy=(x_end, y_end),
        xytext=(x_start, y_start),
        arrowprops=ARROW_STYLE,
    )


def draw_label_on_arrow(ax, x, y, text, ha="center"):
    """Place a small label near an arrow (for Yes/No annotations)."""
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va="center",
        fontsize=FONT_SIZE - 1.5,
        fontfamily=FONT_FAMILY,
        fontstyle="italic",
        color=COLOR_TEXT,
    )


def draw_right_angle_arrow(ax, x_start, y_start, x_mid, y_end):
    """Draw an L-shaped arrow: horizontal then vertical."""
    ax.annotate(
        "",
        xy=(x_mid, y_end),
        xytext=(x_mid, y_start),
        arrowprops=ARROW_STYLE,
    )
    ax.plot(
        [x_start, x_mid],
        [y_start, y_start],
        color=COLOR_TEXT,
        lw=1.2,
    )


def draw_loop_arrow(ax, x_source, y_source, x_target, y_target, side="left"):
    """Draw a loop-back arrow that goes out to the side and back up/down.

    The arrow exits horizontally from *source*, travels vertically,
    and re-enters *target* horizontally.
    """
    offset = -1.1 if side == "left" else 1.1
    x_waypoint = x_source + offset

    # Horizontal line out from source
    ax.plot(
        [x_source, x_waypoint],
        [y_source, y_source],
        color=COLOR_TEXT,
        lw=1.2,
    )
    # Vertical line
    ax.plot(
        [x_waypoint, x_waypoint],
        [y_source, y_target],
        color=COLOR_TEXT,
        lw=1.2,
    )
    # Horizontal arrow into target
    ax.annotate(
        "",
        xy=(x_target, y_target),
        xytext=(x_waypoint, y_target),
        arrowprops=ARROW_STYLE,
    )


def save_figure(fig, filename):
    """Save figure as SVG to the output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(
        filepath, format="svg", bbox_inches="tight", pad_inches=0.2, transparent=True
    )
    plt.close(fig)
    print(f"  saved {filepath}")


# ---------------------------------------------------------------------------
# Flowchart 1: Hill Climbing
# ---------------------------------------------------------------------------


def generate_hill_climbing():
    color = COLOR_LOCAL
    fig, ax = new_figure()
    cx = FIGURE_WIDTH / 2

    # Vertical positions (top to bottom)
    y_start = 7.2
    y_gen = 6.2
    y_eval = 5.3
    y_dec = 4.2
    y_yes = 3.2
    y_no = 3.2

    draw_start(ax, cx, y_start, "Start", color)
    draw_arrow(ax, cx, y_start - 0.22, cx, y_gen + BOX_HEIGHT / 2)

    draw_box(ax, cx, y_gen, "Generate n neighbors\nwithin epsilon", color)
    draw_arrow(ax, cx, y_gen - BOX_HEIGHT / 2, cx, y_eval + BOX_HEIGHT / 2)

    draw_box(ax, cx, y_eval, "Evaluate all neighbors", color)
    draw_arrow(ax, cx, y_eval - BOX_HEIGHT / 2, cx, y_dec + DIAMOND_SIZE)

    draw_diamond(ax, cx, y_dec, "Better?", color, size=0.48)
    # Yes branch (left)
    draw_label_on_arrow(ax, cx - 0.65, y_dec - 0.08, "Yes", ha="right")
    draw_right_angle_arrow(
        ax, cx - 0.48, y_dec, cx - 1.5, y_yes - 0.26 + BOX_HEIGHT / 2
    )
    draw_box(
        ax,
        cx - 1.5,
        y_yes - 0.5,
        "Move to\nbest neighbor",
        color,
        width=1.8,
        height=0.52,
    )

    # No branch (right)
    draw_label_on_arrow(ax, cx + 0.65, y_dec - 0.08, "No", ha="left")
    draw_right_angle_arrow(ax, cx + 0.48, y_dec, cx + 1.5, y_no - 0.26 + BOX_HEIGHT / 2)
    draw_box(ax, cx + 1.5, y_no - 0.5, "Stay\nin place", color, width=1.8, height=0.52)

    # Both branches merge and loop back to generate
    y_merge = 2.2
    ax.plot(
        [cx - 1.5, cx - 1.5],
        [y_yes - 0.5 - BOX_HEIGHT / 2, y_merge],
        color=COLOR_TEXT,
        lw=1.2,
    )
    ax.plot(
        [cx + 1.5, cx + 1.5],
        [y_no - 0.5 - BOX_HEIGHT / 2, y_merge],
        color=COLOR_TEXT,
        lw=1.2,
    )
    ax.plot(
        [cx - 1.5, cx + 1.5],
        [y_merge, y_merge],
        color=COLOR_TEXT,
        lw=1.2,
    )
    # Arrow from merge back up to Generate (loop on left side)
    draw_loop_arrow(
        ax,
        cx - 1.5,
        y_merge,
        cx - BOX_WIDTH / 2,
        y_gen,
        side="left",
    )

    save_figure(fig, "hill_climbing_flowchart.svg")


# ---------------------------------------------------------------------------
# Flowchart 2: Simulated Annealing
# ---------------------------------------------------------------------------


def generate_simulated_annealing():
    color = COLOR_LOCAL
    fig, ax = new_figure()
    cx = FIGURE_WIDTH / 2

    y_start = 7.4
    y_gen = 6.6
    y_calc = 5.9
    y_dec1 = 5.1
    y_accept_direct = 4.3
    y_dec2 = 4.3
    y_accept_prob = 3.4
    y_reject = 3.4
    y_temp = 2.3

    draw_start(ax, cx, y_start, "Start", color)
    draw_arrow(ax, cx, y_start - 0.22, cx, y_gen + BOX_HEIGHT / 2)

    draw_box(ax, cx, y_gen, "Generate neighbor", color)
    draw_arrow(ax, cx, y_gen - BOX_HEIGHT / 2, cx, y_calc + BOX_HEIGHT / 2)

    draw_box(ax, cx, y_calc, "Calculate delta = new - current", color)
    draw_arrow(ax, cx, y_calc - BOX_HEIGHT / 2, cx, y_dec1 + DIAMOND_SIZE)

    draw_diamond(ax, cx, y_dec1, "d > 0?", color, size=0.44)

    # Yes: go left to Accept
    draw_label_on_arrow(ax, cx - 0.63, y_dec1 - 0.08, "Yes", ha="right")
    x_left = cx - 1.8
    draw_right_angle_arrow(
        ax, cx - 0.44, y_dec1, x_left, y_accept_direct + BOX_HEIGHT / 2
    )
    draw_box(ax, x_left, y_accept_direct, "Accept\nmove", color, width=1.3, height=0.52)

    # No: go right to second decision
    draw_label_on_arrow(ax, cx + 0.63, y_dec1 - 0.08, "No", ha="left")
    x_right = cx + 1.2
    draw_right_angle_arrow(ax, cx + 0.44, y_dec1, x_right, y_dec2 + DIAMOND_SIZE)

    # Second decision diamond (smaller text)
    draw_diamond(ax, x_right, y_dec2, "rand <\nexp?", color, size=0.44)

    # Yes: accept
    draw_label_on_arrow(ax, x_right - 0.63, y_dec2 - 0.08, "Yes", ha="right")
    x_prob_accept = x_right - 1.2
    draw_right_angle_arrow(
        ax, x_right - 0.44, y_dec2, x_prob_accept, y_accept_prob + BOX_HEIGHT / 2
    )
    draw_box(
        ax, x_prob_accept, y_accept_prob, "Accept\nmove", color, width=1.1, height=0.52
    )

    # No: reject
    draw_label_on_arrow(ax, x_right + 0.24, y_dec2 - 0.5, "No", ha="left")
    draw_arrow(ax, x_right, y_dec2 - DIAMOND_SIZE, x_right, y_reject + BOX_HEIGHT / 2)
    draw_box(ax, x_right, y_reject - 0.5, "Reject\nmove", color, width=1.1, height=0.52)

    # All three paths converge at Decrease Temperature
    y_converge = 2.55
    # Left (accept direct)
    ax.plot(
        [x_left, x_left],
        [y_accept_direct - BOX_HEIGHT / 2, y_converge],
        color=COLOR_TEXT,
        lw=1.2,
    )
    # Center (accept prob)
    ax.plot(
        [x_prob_accept, x_prob_accept],
        [y_accept_prob - BOX_HEIGHT / 2, y_converge],
        color=COLOR_TEXT,
        lw=1.2,
    )
    # Right (reject)
    ax.plot(
        [x_right, x_right],
        [y_reject - 0.5 - BOX_HEIGHT / 2, y_converge],
        color=COLOR_TEXT,
        lw=1.2,
    )
    # Horizontal merge
    ax.plot(
        [x_left, x_right],
        [y_converge, y_converge],
        color=COLOR_TEXT,
        lw=1.2,
    )
    # Down to temperature box
    draw_arrow(ax, cx, y_converge, cx, y_temp + BOX_HEIGHT / 2)

    draw_box(ax, cx, y_temp, "Decrease temperature", color)

    # Loop back to Generate
    draw_loop_arrow(
        ax,
        cx - BOX_WIDTH / 2,
        y_temp,
        cx - BOX_WIDTH / 2,
        y_gen,
        side="left",
    )

    save_figure(fig, "simulated_annealing_flowchart.svg")


# ---------------------------------------------------------------------------
# Flowchart 3: Particle Swarm Optimization
# ---------------------------------------------------------------------------


def generate_particle_swarm():
    color = COLOR_POPULATION
    fig, ax = new_figure()
    cx = FIGURE_WIDTH / 2

    y_start = 7.4
    y_init = 6.6
    y_loop = 5.9
    y_vel = 5.1
    y_pos = 4.3
    y_eval = 3.55
    y_dec_p = 2.7
    y_upd_p = 2.7
    y_dec_g = 1.7
    y_upd_g = 1.7

    draw_start(ax, cx, y_start, "Start", color)
    draw_arrow(ax, cx, y_start - 0.22, cx, y_init + BOX_HEIGHT / 2)

    draw_box(
        ax,
        cx,
        y_init,
        "Initialize particles with\nrandom positions and velocities",
        color,
    )
    draw_arrow(ax, cx, y_init - BOX_HEIGHT / 2, cx, y_loop + BOX_HEIGHT / 2)

    draw_box(ax, cx, y_loop, "For each particle:", color, width=2.4, height=0.40)
    draw_arrow(ax, cx, y_loop - 0.20, cx, y_vel + BOX_HEIGHT / 2)

    draw_box(
        ax,
        cx,
        y_vel,
        "v = w*v + c1*r1*(pbest-x)\n     + c2*r2*(gbest-x)",
        color,
    )
    draw_arrow(ax, cx, y_vel - BOX_HEIGHT / 2, cx, y_pos + BOX_HEIGHT / 2)

    draw_box(ax, cx, y_pos, "Update position: x = x + v", color)
    draw_arrow(ax, cx, y_pos - BOX_HEIGHT / 2, cx, y_eval + BOX_HEIGHT / 2)

    draw_box(ax, cx, y_eval, "Evaluate fitness", color)
    draw_arrow(ax, cx, y_eval - BOX_HEIGHT / 2, cx, y_dec_p + DIAMOND_SIZE)

    # Decision: better than personal best?
    x_dec_p = cx
    draw_diamond(ax, x_dec_p, y_dec_p, "pbest?", color, size=0.42)

    # Yes branch: right
    draw_label_on_arrow(ax, x_dec_p + 0.6, y_dec_p - 0.05, "Yes", ha="left")
    x_upd_p = cx + 1.6
    draw_right_angle_arrow(
        ax, x_dec_p + 0.42, y_dec_p, x_upd_p, y_upd_p - 0.05 + BOX_HEIGHT / 2
    )
    draw_box(
        ax, x_upd_p, y_upd_p - 0.25, "Update\npbest", color, width=1.2, height=0.44
    )
    # Rejoin: down from update_pbest to dec_g level
    ax.plot(
        [x_upd_p, x_upd_p],
        [y_upd_p - 0.25 - 0.22, y_dec_g],
        color=COLOR_TEXT,
        lw=1.2,
    )
    ax.plot(
        [x_upd_p, cx + 0.42],
        [y_dec_g, y_dec_g],
        color=COLOR_TEXT,
        lw=1.2,
    )

    # No branch: straight down
    draw_label_on_arrow(ax, x_dec_p - 0.18, y_dec_p - 0.52, "No", ha="right")
    draw_arrow(ax, x_dec_p, y_dec_p - DIAMOND_SIZE, cx, y_dec_g + DIAMOND_SIZE)

    # Decision: better than global best?
    draw_diamond(ax, cx, y_dec_g, "gbest?", color, size=0.42)

    # Yes branch: right
    draw_label_on_arrow(ax, cx + 0.6, y_dec_g - 0.05, "Yes", ha="left")
    x_upd_g = cx + 1.6
    draw_right_angle_arrow(
        ax, cx + 0.42, y_dec_g, x_upd_g, y_upd_g - 0.05 + BOX_HEIGHT / 2
    )
    draw_box(
        ax, x_upd_g, y_upd_g - 0.25, "Update\ngbest", color, width=1.2, height=0.44
    )
    # Rejoin from update_gbest to loop-back path
    y_bottom = 0.9
    ax.plot(
        [x_upd_g, x_upd_g],
        [y_upd_g - 0.25 - 0.22, y_bottom],
        color=COLOR_TEXT,
        lw=1.2,
    )
    ax.plot(
        [x_upd_g, cx],
        [y_bottom, y_bottom],
        color=COLOR_TEXT,
        lw=1.2,
    )

    # No branch: straight down to loop
    draw_label_on_arrow(ax, cx - 0.18, y_dec_g - 0.52, "No", ha="right")
    draw_arrow(ax, cx, y_dec_g - DIAMOND_SIZE, cx, y_bottom + 0.08)

    # Loop back to "For each particle"
    draw_loop_arrow(
        ax,
        cx,
        y_bottom,
        cx - BOX_WIDTH / 2 + 0.4,
        y_loop,
        side="left",
    )

    save_figure(fig, "particle_swarm_flowchart.svg")


# ---------------------------------------------------------------------------
# Flowchart 4: Bayesian Optimization
# ---------------------------------------------------------------------------


def generate_bayesian_optimization():
    color = COLOR_SMBO
    fig, ax = new_figure()
    cx = FIGURE_WIDTH / 2

    y_start = 7.3
    y_init = 6.4
    y_gp = 5.4
    y_acq = 4.5
    y_sel = 3.6
    y_eval = 2.7
    y_add = 1.8

    draw_start(ax, cx, y_start, "Start", color)
    draw_arrow(ax, cx, y_start - 0.22, cx, y_init + BOX_HEIGHT / 2)

    draw_box(ax, cx, y_init, "Evaluate initial random points", color)
    draw_arrow(ax, cx, y_init - BOX_HEIGHT / 2, cx, y_gp + BOX_HEIGHT / 2)

    draw_box(ax, cx, y_gp, "Fit Gaussian Process\nto observations", color)
    draw_arrow(ax, cx, y_gp - BOX_HEIGHT / 2, cx, y_acq + BOX_HEIGHT / 2)

    draw_box(ax, cx, y_acq, "Compute acquisition\nfunction (EI)", color)
    draw_arrow(ax, cx, y_acq - BOX_HEIGHT / 2, cx, y_sel + BOX_HEIGHT / 2)

    draw_box(ax, cx, y_sel, "Select point with\nhighest EI", color)
    draw_arrow(ax, cx, y_sel - BOX_HEIGHT / 2, cx, y_eval + BOX_HEIGHT / 2)

    draw_box(ax, cx, y_eval, "Evaluate objective function", color)
    draw_arrow(ax, cx, y_eval - BOX_HEIGHT / 2, cx, y_add + BOX_HEIGHT / 2)

    draw_box(ax, cx, y_add, "Add observation to dataset", color)

    # Loop back to Fit GP
    draw_loop_arrow(
        ax,
        cx - BOX_WIDTH / 2,
        y_add,
        cx - BOX_WIDTH / 2,
        y_gp,
        side="left",
    )

    save_figure(fig, "bayesian_optimization_flowchart.svg")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Generating algorithm flowcharts...")
    generate_hill_climbing()
    generate_simulated_annealing()
    generate_particle_swarm()
    generate_bayesian_optimization()
    print("Done.")


if __name__ == "__main__":
    main()
