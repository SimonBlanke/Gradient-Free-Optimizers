"""Generate conceptual comparison SVG diagrams for each algorithm category.

Produces four diagrams illustrating the behavioral differences between
algorithms within each category: local search, global search, population-based,
and sequential model-based optimization.
"""

import os

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams["svg.fonttype"] = "none"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "source", "_static", "diagrams")
)

# Category accent colors
COLOR_LOCAL = "#e53e3e"
COLOR_GLOBAL = "#38a169"
COLOR_POPULATION = "#805ad5"
COLOR_SMBO = "#dd6b20"

# Shared palette
COLOR_TEXT = "#2d3748"
COLOR_BG = "#f7fafc"
COLOR_BORDER = "#e2e8f0"


def _apply_style(ax, title, accent_color):
    """Apply shared visual styling to a subplot axis."""
    ax.set_title(title, fontsize=11, fontweight="bold", color=accent_color, pad=10)
    ax.tick_params(colors=COLOR_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(COLOR_BORDER)


def _save(fig, filename):
    """Save figure as SVG with consistent settings."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  saved {path}")


# ──────────────────────────────────────────────
# Diagram 1 -- Local Search Comparison
# ──────────────────────────────────────────────


def _landscape(x):
    """Multi-peak 1D landscape used for local-search illustrations."""
    return (
        np.sin(2.0 * x) * 0.7 + np.sin(4.5 * x) * 0.4 + np.cos(1.2 * x) * 0.3 + 0.05 * x
    )


def generate_local_search():
    x = np.linspace(0, 6, 500)
    y = _landscape(x)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, y, color=COLOR_TEXT, linewidth=2, zorder=2)
    ax.fill_between(x, y, y.min() - 0.3, alpha=0.06, color=COLOR_LOCAL)
    _apply_style(ax, "Local Search Algorithms -- Behavioral Comparison", COLOR_LOCAL)

    # --- Hill Climbing: stuck at local peak ---
    peak_idx = 95 + np.argmax(y[95:160])
    px, py = x[peak_idx], y[peak_idx]
    ax.annotate(
        "Gets stuck",
        xy=(px, py),
        xytext=(px - 0.55, py + 0.45),
        fontsize=9,
        fontweight="bold",
        color=COLOR_LOCAL,
        arrowprops=dict(arrowstyle="-|>", color=COLOR_LOCAL, lw=1.5),
    )
    ax.plot(px, py, "o", color=COLOR_LOCAL, markersize=7, zorder=3)
    ax.text(px - 0.55, py + 0.55, "Hill Climbing", fontsize=8, color=COLOR_TEXT)

    # --- Stochastic HC: probabilistic jump ---
    sp_x = px
    sp_y = py
    land_target_idx = peak_idx + 55
    target_x = x[land_target_idx]
    target_y = y[land_target_idx]
    ax.annotate(
        "",
        xy=(target_x, target_y + 0.05),
        xytext=(sp_x, sp_y),
        arrowprops=dict(
            arrowstyle="-|>",
            color=COLOR_LOCAL,
            lw=1.5,
            linestyle="dotted",
        ),
    )
    ax.plot(target_x, target_y, "o", color=COLOR_LOCAL, markersize=6, zorder=3)
    mid_x = (sp_x + target_x) / 2
    mid_y = max(sp_y, target_y) + 0.35
    ax.text(
        mid_x,
        mid_y,
        "Probabilistic escape",
        fontsize=8,
        ha="center",
        color=COLOR_TEXT,
        style="italic",
    )
    ax.text(
        mid_x, mid_y + 0.15, "Stochastic HC", fontsize=8, ha="center", color=COLOR_TEXT
    )

    # --- Repulsing HC: expanding step size arrows ---
    rep_x = 4.0
    rep_y_val = _landscape(rep_x)
    for i, dx in enumerate([0.15, 0.35, 0.6]):
        alpha = 0.4 + 0.2 * i
        ax.annotate(
            "",
            xy=(rep_x + dx, rep_y_val),
            xytext=(rep_x, rep_y_val),
            arrowprops=dict(arrowstyle="-|>", color=COLOR_LOCAL, lw=1.2, alpha=alpha),
        )
        ax.annotate(
            "",
            xy=(rep_x - dx, rep_y_val),
            xytext=(rep_x, rep_y_val),
            arrowprops=dict(arrowstyle="-|>", color=COLOR_LOCAL, lw=1.2, alpha=alpha),
        )
    ax.plot(rep_x, rep_y_val, "o", color=COLOR_LOCAL, markersize=6, zorder=3)
    ax.text(
        rep_x,
        rep_y_val + 0.40,
        "Growing step size",
        fontsize=8,
        ha="center",
        color=COLOR_TEXT,
        style="italic",
    )
    ax.text(
        rep_x,
        rep_y_val + 0.55,
        "Repulsing HC",
        fontsize=8,
        ha="center",
        color=COLOR_TEXT,
    )

    # --- Simulated Annealing: temperature bar ---
    bar_left = 0.3
    bar_bottom = y.min() - 0.22
    bar_w = 1.2
    bar_h = 0.12
    hot_cold = LinearSegmentedColormap.from_list("hc", ["#e53e3e", "#3182ce"])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(
        gradient,
        aspect="auto",
        cmap=hot_cold,
        extent=[bar_left, bar_left + bar_w, bar_bottom, bar_bottom + bar_h],
        zorder=3,
    )
    ax.text(
        bar_left,
        bar_bottom - 0.1,
        "hot",
        fontsize=7,
        color="#e53e3e",
        ha="left",
    )
    ax.text(
        bar_left + bar_w,
        bar_bottom - 0.1,
        "cold",
        fontsize=7,
        color="#3182ce",
        ha="right",
    )
    ax.text(
        bar_left + bar_w / 2,
        bar_bottom + bar_h + 0.08,
        "Cooling schedule",
        fontsize=8,
        ha="center",
        color=COLOR_TEXT,
        style="italic",
    )
    ax.text(
        bar_left + bar_w / 2,
        bar_bottom + bar_h + 0.23,
        "Simulated Annealing",
        fontsize=8,
        ha="center",
        color=COLOR_TEXT,
    )

    # --- Downhill Simplex: triangle on the landscape ---
    sx = 5.0
    tri_pts = np.array(
        [
            [sx - 0.25, _landscape(sx - 0.25)],
            [sx + 0.25, _landscape(sx + 0.25)],
            [sx, _landscape(sx) + 0.30],
        ]
    )
    triangle = plt.Polygon(
        tri_pts,
        closed=True,
        fill=False,
        edgecolor=COLOR_LOCAL,
        linewidth=2,
        zorder=4,
    )
    ax.add_patch(triangle)
    ax.text(
        sx,
        tri_pts[:, 1].max() + 0.15,
        "Simplex operations",
        fontsize=8,
        ha="center",
        color=COLOR_TEXT,
        style="italic",
    )
    ax.text(
        sx,
        tri_pts[:, 1].max() + 0.30,
        "Downhill Simplex",
        fontsize=8,
        ha="center",
        color=COLOR_TEXT,
    )

    ax.set_xlabel("x", fontsize=9, color=COLOR_TEXT)
    ax.set_ylabel("f(x)", fontsize=9, color=COLOR_TEXT)
    ax.set_xlim(x[0] - 0.1, x[-1] + 0.1)
    ax.set_ylim(y.min() - 0.5, y.max() + 0.9)

    _save(fig, "local_search_comparison.svg")


# ──────────────────────────────────────────────
# Diagram 2 -- Global Search Comparison
# ──────────────────────────────────────────────


def generate_global_search():
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle(
        "Global Search Algorithms -- Coverage Patterns",
        fontsize=13,
        fontweight="bold",
        color=COLOR_GLOBAL,
        y=1.02,
    )
    rng = np.random.RandomState(42)

    titles = ["Random Search", "Grid Search", "Pattern Search", "DIRECT"]

    for ax, title in zip(axes, titles):
        _apply_style(ax, title, COLOR_GLOBAL)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        # Search space border
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color(COLOR_BORDER)

    # Random Search
    rx, ry = rng.rand(40), rng.rand(40)
    axes[0].scatter(rx, ry, s=25, color=COLOR_GLOBAL, alpha=0.7, zorder=3)

    # Grid Search
    gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 7), np.linspace(0.1, 0.9, 7))
    axes[1].scatter(
        gx.ravel(),
        gy.ravel(),
        s=25,
        color=COLOR_GLOBAL,
        alpha=0.7,
        zorder=3,
    )

    # Pattern Search -- cross-shaped probes around several centers
    centers = [(0.3, 0.3), (0.7, 0.5), (0.5, 0.75)]
    for cx, cy in centers:
        axes[2].plot(cx, cy, "o", color=COLOR_GLOBAL, markersize=6, zorder=4)
        for step in [0.08, 0.15]:
            offsets = [(step, 0), (-step, 0), (0, step), (0, -step)]
            for dx, dy in offsets:
                axes[2].plot(
                    cx + dx,
                    cy + dy,
                    "o",
                    color=COLOR_GLOBAL,
                    markersize=4,
                    alpha=0.5,
                    zorder=3,
                )
                axes[2].plot(
                    [cx, cx + dx],
                    [cy, cy + dy],
                    color=COLOR_GLOBAL,
                    lw=0.7,
                    alpha=0.4,
                )

    # DIRECT -- recursive rectangle partitioning
    def _draw_rect(ax, x0, y0, w, h, depth, max_depth):
        rect = mpatches.FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="square,pad=0",
            facecolor="none",
            edgecolor=COLOR_GLOBAL,
            linewidth=max(0.5, 1.5 - 0.3 * depth),
            alpha=max(0.3, 1.0 - 0.15 * depth),
        )
        ax.add_patch(rect)
        cx, cy = x0 + w / 2, y0 + h / 2
        ax.plot(
            cx,
            cy,
            "o",
            color=COLOR_GLOBAL,
            markersize=max(2, 5 - depth),
            alpha=0.7,
            zorder=3,
        )
        if depth < max_depth:
            # Split the larger dimension into thirds, recurse on center
            if w >= h:
                third = w / 3
                _draw_rect(ax, x0, y0, third, h, depth + 1, max_depth)
                _draw_rect(ax, x0 + third, y0, third, h, depth + 1, max_depth)
                _draw_rect(ax, x0 + 2 * third, y0, third, h, depth + 1, max_depth)
            else:
                third = h / 3
                _draw_rect(ax, x0, y0, w, third, depth + 1, max_depth)
                _draw_rect(ax, x0, y0 + third, w, third, depth + 1, max_depth)
                _draw_rect(ax, x0, y0 + 2 * third, w, third, depth + 1, max_depth)

    _draw_rect(axes[3], 0.02, 0.02, 0.96, 0.96, 0, 2)

    fig.tight_layout()
    _save(fig, "global_search_comparison.svg")


# ──────────────────────────────────────────────
# Diagram 3 -- Population Comparison
# ──────────────────────────────────────────────


def generate_population():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Population-Based Algorithms -- Movement Patterns",
        fontsize=13,
        fontweight="bold",
        color=COLOR_POPULATION,
        y=1.02,
    )

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color(COLOR_BORDER)

    # --- PSO ---
    _apply_style(axes[0], "Particle Swarm Optimization", COLOR_POPULATION)
    particle = np.array([0.35, 0.30])
    p_best = np.array([0.50, 0.55])
    g_best = np.array([0.75, 0.80])

    axes[0].plot(*particle, "o", color=COLOR_POPULATION, markersize=9, zorder=4)
    axes[0].text(
        particle[0] - 0.02,
        particle[1] - 0.06,
        "particle",
        fontsize=8,
        color=COLOR_TEXT,
        ha="center",
    )

    # Inertia component
    inertia_vec = np.array([0.12, 0.04])
    axes[0].annotate(
        "",
        xy=particle + inertia_vec,
        xytext=particle,
        arrowprops=dict(arrowstyle="-|>", color="#718096", lw=1.8),
    )
    axes[0].text(
        particle[0] + inertia_vec[0] / 2 - 0.08,
        particle[1] + inertia_vec[1] / 2 + 0.03,
        "inertia",
        fontsize=7,
        color="#718096",
        style="italic",
    )

    # Cognitive (toward personal best)
    axes[0].plot(*p_best, "s", color="#38a169", markersize=8, zorder=4)
    axes[0].text(p_best[0] + 0.04, p_best[1], "p_best", fontsize=8, color="#38a169")
    cog_dir = p_best - particle
    cog_dir = cog_dir / np.linalg.norm(cog_dir) * 0.15
    axes[0].annotate(
        "",
        xy=particle + cog_dir,
        xytext=particle,
        arrowprops=dict(arrowstyle="-|>", color="#38a169", lw=1.5, linestyle="dashed"),
    )

    # Social (toward global best)
    axes[0].plot(*g_best, "*", color="#e53e3e", markersize=12, zorder=4)
    axes[0].text(
        g_best[0] + 0.03, g_best[1] - 0.06, "g_best", fontsize=8, color="#e53e3e"
    )
    soc_dir = g_best - particle
    soc_dir = soc_dir / np.linalg.norm(soc_dir) * 0.15
    axes[0].annotate(
        "",
        xy=particle + soc_dir,
        xytext=particle,
        arrowprops=dict(arrowstyle="-|>", color="#e53e3e", lw=1.5, linestyle="dashed"),
    )

    # Resultant velocity
    resultant = inertia_vec + 0.6 * cog_dir + 0.6 * soc_dir
    axes[0].annotate(
        "",
        xy=particle + resultant,
        xytext=particle,
        arrowprops=dict(arrowstyle="-|>", color=COLOR_POPULATION, lw=2.5),
    )
    axes[0].text(
        particle[0] + resultant[0] + 0.02,
        particle[1] + resultant[1] + 0.02,
        "velocity",
        fontsize=8,
        fontweight="bold",
        color=COLOR_POPULATION,
    )

    # --- Genetic Algorithm ---
    _apply_style(axes[1], "Genetic Algorithm", COLOR_POPULATION)
    # Parent chromosomes represented as colored blocks
    block_w = 0.08
    block_h = 0.06
    parent_colors_a = ["#805ad5", "#805ad5", "#805ad5", "#e53e3e", "#e53e3e", "#e53e3e"]
    parent_colors_b = ["#38a169", "#38a169", "#38a169", "#dd6b20", "#dd6b20", "#dd6b20"]
    # Crossover point at index 3
    child_colors = ["#805ad5", "#805ad5", "#805ad5", "#dd6b20", "#dd6b20", "#dd6b20"]

    y_p1 = 0.75
    y_p2 = 0.55
    y_child = 0.25
    x_start = 0.18

    axes[1].text(
        0.08,
        y_p1 + block_h / 2,
        "P1",
        fontsize=9,
        va="center",
        fontweight="bold",
        color=COLOR_TEXT,
    )
    axes[1].text(
        0.08,
        y_p2 + block_h / 2,
        "P2",
        fontsize=9,
        va="center",
        fontweight="bold",
        color=COLOR_TEXT,
    )
    axes[1].text(
        0.04,
        y_child + block_h / 2,
        "Child",
        fontsize=9,
        va="center",
        fontweight="bold",
        color=COLOR_TEXT,
    )

    for i in range(6):
        xi = x_start + i * (block_w + 0.02)
        # Parent 1
        axes[1].add_patch(
            mpatches.FancyBboxPatch(
                (xi, y_p1),
                block_w,
                block_h,
                boxstyle="round,pad=0.01",
                facecolor=parent_colors_a[i],
                edgecolor="white",
                linewidth=1,
            )
        )
        # Parent 2
        axes[1].add_patch(
            mpatches.FancyBboxPatch(
                (xi, y_p2),
                block_w,
                block_h,
                boxstyle="round,pad=0.01",
                facecolor=parent_colors_b[i],
                edgecolor="white",
                linewidth=1,
            )
        )
        # Child
        axes[1].add_patch(
            mpatches.FancyBboxPatch(
                (xi, y_child),
                block_w,
                block_h,
                boxstyle="round,pad=0.01",
                facecolor=child_colors[i],
                edgecolor="white",
                linewidth=1,
            )
        )

    # Crossover line
    cross_x = x_start + 3 * (block_w + 0.02) - 0.01
    axes[1].plot(
        [cross_x, cross_x],
        [y_p1 - 0.02, y_p1 + block_h + 0.02],
        color=COLOR_LOCAL,
        lw=2,
        linestyle="--",
        zorder=5,
    )
    axes[1].text(
        cross_x,
        y_p1 + block_h + 0.05,
        "crossover",
        fontsize=8,
        ha="center",
        color=COLOR_LOCAL,
        style="italic",
    )

    # Arrows from parents to child
    mid_x = x_start + 2.5 * (block_w + 0.02)
    axes[1].annotate(
        "",
        xy=(mid_x, y_child + block_h + 0.02),
        xytext=(mid_x, y_p2 - 0.02),
        arrowprops=dict(arrowstyle="-|>", color=COLOR_POPULATION, lw=1.5),
    )

    # --- Differential Evolution ---
    _apply_style(axes[2], "Differential Evolution", COLOR_POPULATION)
    # Three vectors: x_r1, x_r2, x_r3 and the mutation
    xr1 = np.array([0.25, 0.60])
    xr2 = np.array([0.65, 0.75])
    xr3 = np.array([0.55, 0.35])

    for pt, label in [(xr1, "$x_{r1}$"), (xr2, "$x_{r2}$"), (xr3, "$x_{r3}$")]:
        axes[2].plot(*pt, "o", color=COLOR_POPULATION, markersize=8, zorder=4)
        axes[2].text(pt[0] + 0.03, pt[1] + 0.03, label, fontsize=10, color=COLOR_TEXT)

    # Difference vector x_r2 - x_r3
    diff = xr2 - xr3
    axes[2].annotate(
        "",
        xy=xr2,
        xytext=xr3,
        arrowprops=dict(arrowstyle="-|>", color="#718096", lw=1.5, linestyle="dashed"),
    )
    axes[2].text(
        (xr2[0] + xr3[0]) / 2 + 0.04,
        (xr2[1] + xr3[1]) / 2 - 0.04,
        "$x_{r2} - x_{r3}$",
        fontsize=9,
        color="#718096",
    )

    # Mutant vector: x_r1 + F * diff
    f_scale = 0.7
    mutant = xr1 + f_scale * diff
    axes[2].plot(*mutant, "D", color=COLOR_POPULATION, markersize=8, zorder=4)
    axes[2].text(
        mutant[0] + 0.03,
        mutant[1] - 0.06,
        "mutant",
        fontsize=9,
        fontweight="bold",
        color=COLOR_POPULATION,
    )
    axes[2].annotate(
        "",
        xy=mutant,
        xytext=xr1,
        arrowprops=dict(arrowstyle="-|>", color=COLOR_POPULATION, lw=2),
    )
    axes[2].text(
        (xr1[0] + mutant[0]) / 2 - 0.12,
        (xr1[1] + mutant[1]) / 2,
        "$x_{r1} + F \\cdot \\Delta$",
        fontsize=9,
        color=COLOR_POPULATION,
    )

    fig.tight_layout()
    _save(fig, "population_comparison.svg")


# ──────────────────────────────────────────────
# Diagram 4 -- SMBO Comparison
# ──────────────────────────────────────────────


def generate_smbo():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Sequential Model-Based Optimization -- Surrogate Models",
        fontsize=13,
        fontweight="bold",
        color=COLOR_SMBO,
        y=1.02,
    )
    rng = np.random.RandomState(7)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color(COLOR_BORDER)

    # --- Bayesian Optimization (Gaussian Process) ---
    _apply_style(axes[0], "Bayesian Optimization (GP)", COLOR_SMBO)
    x = np.linspace(0, 5, 200)
    mean = np.sin(x) * 0.5 + 0.3 * x
    std = 0.15 + 0.25 * np.abs(np.sin(0.8 * x + 1))

    axes[0].plot(x, mean, color=COLOR_SMBO, lw=2, label="mean $\\mu(x)$")
    axes[0].fill_between(
        x,
        mean - 2 * std,
        mean + 2 * std,
        alpha=0.18,
        color=COLOR_SMBO,
        label="$\\pm 2\\sigma$",
    )
    axes[0].fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.25,
        color=COLOR_SMBO,
        label="$\\pm 1\\sigma$",
    )

    # Observed points
    obs_x = np.array([0.5, 1.5, 2.8, 4.2])
    obs_y = np.sin(obs_x) * 0.5 + 0.3 * obs_x + rng.randn(4) * 0.05
    axes[0].scatter(obs_x, obs_y, color=COLOR_TEXT, s=40, zorder=5, label="observed")

    axes[0].set_xlabel("x", fontsize=9, color=COLOR_TEXT)
    axes[0].set_ylabel("f(x)", fontsize=9, color=COLOR_TEXT)
    axes[0].legend(fontsize=7, loc="upper left", framealpha=0.8)
    axes[0].tick_params(labelsize=7)

    # --- TPE ---
    _apply_style(axes[1], "Tree-Structured Parzen Estimators", COLOR_SMBO)
    x_tpe = np.linspace(0, 5, 200)

    # l(x): density of good observations (narrow peak)
    l_x = 1.2 * np.exp(-0.5 * ((x_tpe - 2.0) / 0.5) ** 2)
    l_x += 0.4 * np.exp(-0.5 * ((x_tpe - 3.8) / 0.4) ** 2)

    # g(x): density of bad observations (broader, shifted)
    g_x = 0.5 * np.exp(-0.5 * ((x_tpe - 1.0) / 0.8) ** 2)
    g_x += 0.6 * np.exp(-0.5 * ((x_tpe - 3.0) / 1.0) ** 2)
    g_x += 0.3 * np.exp(-0.5 * ((x_tpe - 4.5) / 0.5) ** 2)

    axes[1].plot(x_tpe, l_x, color="#38a169", lw=2, label="$\\ell(x)$ good")
    axes[1].fill_between(x_tpe, l_x, alpha=0.15, color="#38a169")
    axes[1].plot(x_tpe, g_x, color="#e53e3e", lw=2, label="$g(x)$ bad")
    axes[1].fill_between(x_tpe, g_x, alpha=0.15, color="#e53e3e")

    axes[1].set_xlabel("x", fontsize=9, color=COLOR_TEXT)
    axes[1].set_ylabel("density", fontsize=9, color=COLOR_TEXT)
    axes[1].legend(fontsize=8, loc="upper right", framealpha=0.8)
    axes[1].tick_params(labelsize=7)
    axes[1].text(
        2.0,
        max(l_x) + 0.12,
        "maximize $\\ell(x) / g(x)$",
        fontsize=8,
        ha="center",
        color=COLOR_SMBO,
        style="italic",
    )

    # --- Random Forest ---
    _apply_style(axes[2], "Forest (Random Forest Surrogate)", COLOR_SMBO)
    x_rf = np.linspace(0, 5, 300)

    # Individual tree predictions as step functions
    tree_predictions = []
    for i in range(5):
        n_splits = rng.randint(4, 7)
        splits = np.sort(rng.uniform(0.3, 4.7, n_splits))
        splits = np.concatenate([[0], splits, [5]])
        pred = np.zeros_like(x_rf)
        for j in range(len(splits) - 1):
            mask = (x_rf >= splits[j]) & (x_rf < splits[j + 1])
            pred[mask] = rng.uniform(-0.5, 2.0)
        tree_predictions.append(pred)
        axes[2].step(
            x_rf,
            pred,
            color=COLOR_SMBO,
            alpha=0.25,
            lw=1,
            where="post",
        )

    # Average prediction
    mean_pred = np.mean(tree_predictions, axis=0)
    axes[2].step(
        x_rf,
        mean_pred,
        color=COLOR_SMBO,
        lw=2.5,
        where="post",
        label="ensemble mean",
    )

    axes[2].set_xlabel("x", fontsize=9, color=COLOR_TEXT)
    axes[2].set_ylabel("f(x)", fontsize=9, color=COLOR_TEXT)
    axes[2].legend(fontsize=8, loc="upper left", framealpha=0.8)
    axes[2].tick_params(labelsize=7)

    # Label individual trees
    axes[2].text(
        4.5,
        -0.7,
        "individual trees",
        fontsize=7,
        color=COLOR_SMBO,
        alpha=0.6,
        style="italic",
    )

    fig.tight_layout()
    _save(fig, "smbo_comparison.svg")


# ──────────────────────────────────────────────


def main():
    print(f"Output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generate_local_search()
    generate_global_search()
    generate_population()
    generate_smbo()

    print("All diagrams generated.")


if __name__ == "__main__":
    main()
