"""PlotAccessor - provides plotting methods for SearchTracker."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ..tracker import SearchTracker


class PlotAccessor:
    """
    Provides plotting methods for SearchTracker.

    This class is accessed via the `plot` property of SearchTracker.
    All methods return (fig, ax) tuples for further customization.

    Example:
        tracker.plot.convergence()
        tracker.plot.search_space(dimensions=["x", "y"])
        fig, ax = tracker.plot.parameter_importance()
        ax.set_title("My Custom Title")
        plt.savefig("importance.png")
    """

    def __init__(self, tracker: SearchTracker):
        self._tracker = tracker

    def _get_arrays(
        self, run_id: str | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract arrays from tracker records."""
        records = self._tracker.get_records(run_id)

        iterations = np.array([r.iteration for r in records])
        scores = np.array([r.score for r in records])
        times = np.array([r.evaluation_time for r in records])

        return iterations, scores, times

    def _get_parameter_names(self) -> list[str]:
        """Get list of parameter names from records."""
        records = self._tracker.records
        if not records:
            return []
        return list(records[0].parameters.keys())

    # ─────────────────────────────────────────────────────────
    # Convergence Plots
    # ─────────────────────────────────────────────────────────

    def convergence(
        self,
        run_id: str | None = None,
        show_all: bool = True,
        show_best: bool = True,
        log_scale: bool = False,
        figsize: tuple[float, float] = (10, 6),
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """
        Plot score convergence over iterations.

        Shows how the objective function score improves over the
        course of optimization.

        Args:
            run_id: If provided, only plot this run
            show_all: Show scatter of all evaluation scores
            show_best: Show line of best score so far
            log_scale: Use logarithmic y-axis
            figsize: Figure size (width, height)
            ax: Existing axes to plot on (creates new figure if None)

        Returns
        -------
            Tuple of (figure, axes)
        """
        import matplotlib.pyplot as plt

        iterations, scores, _ = self._get_arrays(run_id)

        if len(iterations) == 0:
            raise ValueError("No data to plot")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        if show_all:
            ax.scatter(
                iterations,
                scores,
                alpha=0.4,
                s=15,
                c="steelblue",
                label="Evaluations",
            )

        if show_best:
            best_so_far = np.maximum.accumulate(scores)
            ax.plot(
                iterations,
                best_so_far,
                color="crimson",
                linewidth=2,
                label="Best so far",
            )

            # Mark the best point
            best_idx = np.argmax(scores)
            ax.scatter(
                [iterations[best_idx]],
                [scores[best_idx]],
                color="gold",
                s=100,
                marker="*",
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
                label=f"Best: {scores[best_idx]:.4f}",
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")
        ax.set_title(f"Convergence - {self._tracker.name}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        if log_scale:
            ax.set_yscale("log")

        fig.tight_layout()
        return fig, ax

    def convergence_by_run(
        self,
        figsize: tuple[float, float] = (10, 6),
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """
        Plot convergence curves for all runs overlaid.

        Useful for comparing different optimizers or configurations.

        Args:
            figsize: Figure size (width, height)
            ax: Existing axes to plot on

        Returns
        -------
            Tuple of (figure, axes)
        """
        import matplotlib.pyplot as plt

        run_ids = self._tracker.run_ids

        if len(run_ids) <= 1:
            return self.convergence()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        colors = plt.cm.tab10(np.linspace(0, 1, len(run_ids)))

        for run_id, color in zip(run_ids, colors):
            iterations, scores, _ = self._get_arrays(run_id)
            best_so_far = np.maximum.accumulate(scores)
            ax.plot(iterations, best_so_far, label=run_id, color=color, linewidth=2)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Score")
        ax.set_title(f"Convergence Comparison - {self._tracker.name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig, ax

    # ─────────────────────────────────────────────────────────
    # Search Space Visualization
    # ─────────────────────────────────────────────────────────

    def search_space(
        self,
        dimensions: list[str] | None = None,
        run_id: str | None = None,
        color_by: str = "score",
        cmap: str = "viridis",
        figsize: tuple[float, float] = (10, 8),
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """
        Scatter plot of evaluated points in parameter space.

        Shows where in the search space evaluations occurred,
        with color indicating the score or iteration.

        Args:
            dimensions: Which parameters to plot (default: first 2)
            run_id: If provided, only plot this run
            color_by: Color points by "score" or "iteration"
            cmap: Matplotlib colormap name
            figsize: Figure size (width, height)
            ax: Existing axes to plot on

        Returns
        -------
            Tuple of (figure, axes)
        """
        import matplotlib.pyplot as plt

        records = self._tracker.get_records(run_id)

        if not records:
            raise ValueError("No data to plot")

        param_names = self._get_parameter_names()

        if dimensions is None:
            # Use first two numeric parameters
            numeric_params = []
            for name in param_names:
                if isinstance(records[0].parameters[name], int | float):
                    numeric_params.append(name)
                if len(numeric_params) >= 2:
                    break
            dimensions = numeric_params[:2]

        if len(dimensions) < 2:
            raise ValueError(
                f"Need at least 2 numeric dimensions, got {len(dimensions)}"
            )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        x_vals = [r.parameters[dimensions[0]] for r in records]
        y_vals = [r.parameters[dimensions[1]] for r in records]

        if color_by == "score":
            colors = [r.score for r in records]
            label = "Score"
        else:
            colors = [r.iteration for r in records]
            label = "Iteration"

        scatter = ax.scatter(x_vals, y_vals, c=colors, cmap=cmap, alpha=0.6, s=30)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(label)

        # Mark the best point
        best = self._tracker.best_record
        if best:
            ax.scatter(
                [best.parameters[dimensions[0]]],
                [best.parameters[dimensions[1]]],
                color="red",
                s=150,
                marker="*",
                edgecolors="white",
                linewidths=1,
                zorder=5,
                label=f"Best ({best.score:.4f})",
            )

        ax.set_xlabel(dimensions[0])
        ax.set_ylabel(dimensions[1])
        ax.set_title(f"Search Space - {self._tracker.name}")
        ax.legend()

        fig.tight_layout()
        return fig, ax

    def search_space_3d(
        self,
        dimensions: list[str] | None = None,
        run_id: str | None = None,
        color_by: str = "score",
        cmap: str = "viridis",
        figsize: tuple[float, float] = (10, 8),
    ) -> tuple[Figure, Axes]:
        """
        3D scatter plot of evaluated points.

        Args:
            dimensions: Which 3 parameters to plot (default: first 3)
            run_id: If provided, only plot this run
            color_by: Color points by "score" or "iteration"
            cmap: Matplotlib colormap name
            figsize: Figure size

        Returns
        -------
            Tuple of (figure, axes)
        """
        import matplotlib.pyplot as plt

        records = self._tracker.get_records(run_id)

        if not records:
            raise ValueError("No data to plot")

        param_names = self._get_parameter_names()

        if dimensions is None:
            numeric_params = []
            for name in param_names:
                if isinstance(records[0].parameters[name], int | float):
                    numeric_params.append(name)
                if len(numeric_params) >= 3:
                    break
            dimensions = numeric_params[:3]

        if len(dimensions) < 3:
            raise ValueError(
                f"Need at least 3 numeric dimensions, got {len(dimensions)}"
            )

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        x_vals = [r.parameters[dimensions[0]] for r in records]
        y_vals = [r.parameters[dimensions[1]] for r in records]
        z_vals = [r.parameters[dimensions[2]] for r in records]

        if color_by == "score":
            colors = [r.score for r in records]
        else:
            colors = [r.iteration for r in records]

        scatter = ax.scatter(x_vals, y_vals, z_vals, c=colors, cmap=cmap, alpha=0.6)
        plt.colorbar(scatter, ax=ax, label=color_by.capitalize())

        ax.set_xlabel(dimensions[0])
        ax.set_ylabel(dimensions[1])
        ax.set_zlabel(dimensions[2])
        ax.set_title(f"Search Space 3D - {self._tracker.name}")

        fig.tight_layout()
        return fig, ax

    # ─────────────────────────────────────────────────────────
    # Parameter Analysis
    # ─────────────────────────────────────────────────────────

    def parameter_importance(
        self,
        method: str = "correlation",
        run_id: str | None = None,
        figsize: tuple[float, float] = (10, 6),
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """
        Bar chart of parameter importance.

        Shows which parameters have the most influence on the score.

        Args:
            method: Importance calculation method:
                - "correlation": Absolute Pearson correlation (fast)
                - "variance": Variance of parameter across top scores
            run_id: If provided, only use this run's data
            figsize: Figure size
            ax: Existing axes to plot on

        Returns
        -------
            Tuple of (figure, axes)
        """
        import matplotlib.pyplot as plt

        records = self._tracker.get_records(run_id)

        if not records:
            raise ValueError("No data to plot")

        param_names = self._get_parameter_names()
        scores = np.array([r.score for r in records])

        importances = {}

        for name in param_names:
            values = [r.parameters[name] for r in records]

            # Skip non-numeric parameters
            if not all(isinstance(v, int | float) for v in values):
                continue

            values = np.array(values)

            if method == "correlation":
                # Absolute Pearson correlation
                if np.std(values) > 0 and np.std(scores) > 0:
                    corr = np.corrcoef(values, scores)[0, 1]
                    importances[name] = abs(corr) if not np.isnan(corr) else 0
                else:
                    importances[name] = 0
            elif method == "variance":
                # Variance in top 20% of scores
                threshold = np.percentile(scores, 80)
                top_mask = scores >= threshold
                if top_mask.sum() > 1:
                    importances[name] = np.std(values[top_mask])
                else:
                    importances[name] = 0
            else:
                raise ValueError(f"Unknown method: {method}")

        if not importances:
            raise ValueError("No numeric parameters found")

        # Sort by importance
        sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        names = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))[::-1]
        bars = ax.barh(names, values, color=colors)

        ax.set_xlabel(f"Importance ({method})")
        ax.set_title(f"Parameter Importance - {self._tracker.name}")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                val + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=9,
            )

        fig.tight_layout()
        return fig, ax

    # ─────────────────────────────────────────────────────────
    # Distribution Plots
    # ─────────────────────────────────────────────────────────

    def score_distribution(
        self,
        run_id: str | None = None,
        bins: int = 30,
        figsize: tuple[float, float] = (10, 6),
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """
        Histogram of score distribution.

        Shows the distribution of scores across all evaluations.

        Args:
            run_id: If provided, only use this run's data
            bins: Number of histogram bins
            figsize: Figure size
            ax: Existing axes to plot on

        Returns
        -------
            Tuple of (figure, axes)
        """
        import matplotlib.pyplot as plt

        _, scores, _ = self._get_arrays(run_id)

        if len(scores) == 0:
            raise ValueError("No data to plot")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ax.hist(scores, bins=bins, color="steelblue", edgecolor="white", alpha=0.7)

        # Add vertical lines for statistics
        ax.axvline(np.mean(scores), color="orange", linestyle="--", label="Mean")
        ax.axvline(np.max(scores), color="green", linestyle="-", label="Best")

        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_title(f"Score Distribution - {self._tracker.name}")
        ax.legend()

        fig.tight_layout()
        return fig, ax

    def evaluation_time(
        self,
        run_id: str | None = None,
        figsize: tuple[float, float] = (10, 6),
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """
        Plot evaluation time over iterations.

        Useful for identifying slow evaluations or performance trends.

        Args:
            run_id: If provided, only use this run's data
            figsize: Figure size
            ax: Existing axes to plot on

        Returns
        -------
            Tuple of (figure, axes)
        """
        import matplotlib.pyplot as plt

        iterations, _, times = self._get_arrays(run_id)

        if len(iterations) == 0:
            raise ValueError("No data to plot")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ax.scatter(iterations, times * 1000, alpha=0.5, s=15, c="steelblue")
        ax.axhline(
            np.mean(times) * 1000,
            color="orange",
            linestyle="--",
            label=f"Mean: {np.mean(times)*1000:.1f}ms",
        )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Evaluation Time (ms)")
        ax.set_title(f"Evaluation Time - {self._tracker.name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig, ax

    # ─────────────────────────────────────────────────────────
    # Multi-dimensional Views
    # ─────────────────────────────────────────────────────────

    def parallel_coordinates(
        self,
        run_id: str | None = None,
        top_n: int | None = None,
        figsize: tuple[float, float] = (12, 6),
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """
        Parallel coordinates plot for all parameters.

        Each vertical axis represents a parameter, and each line
        represents an evaluation. Lines are colored by score.

        Args:
            run_id: If provided, only use this run's data
            top_n: If provided, only show top N evaluations by score
            figsize: Figure size
            ax: Existing axes to plot on

        Returns
        -------
            Tuple of (figure, axes)
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        records = self._tracker.get_records(run_id)

        if not records:
            raise ValueError("No data to plot")

        # Sort by score and optionally limit
        records = sorted(records, key=lambda r: r.score, reverse=True)
        if top_n:
            records = records[:top_n]

        # Get numeric parameters
        param_names = []
        for name in self._get_parameter_names():
            if isinstance(records[0].parameters[name], int | float):
                param_names.append(name)

        if len(param_names) < 2:
            raise ValueError("Need at least 2 numeric parameters")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Normalize each parameter to [0, 1]
        param_data = {name: [] for name in param_names}
        scores = []

        for r in records:
            scores.append(r.score)
            for name in param_names:
                param_data[name].append(r.parameters[name])

        normalized_data = {}
        for name in param_names:
            values = np.array(param_data[name])
            min_val, max_val = values.min(), values.max()
            if max_val > min_val:
                normalized_data[name] = (values - min_val) / (max_val - min_val)
            else:
                normalized_data[name] = np.zeros_like(values)

        # Create colormap
        norm = Normalize(vmin=min(scores), vmax=max(scores))
        cmap = plt.cm.viridis

        # Plot lines
        x = np.arange(len(param_names))
        for i, score in enumerate(scores):
            y = [normalized_data[name][i] for name in param_names]
            ax.plot(x, y, color=cmap(norm(score)), alpha=0.5, linewidth=1)

        # Customize axes
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45, ha="right")
        ax.set_ylabel("Normalized Value")
        ax.set_title(f"Parallel Coordinates - {self._tracker.name}")

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Score")

        # Add min/max labels for each parameter
        for i, name in enumerate(param_names):
            values = param_data[name]
            ax.text(i, -0.1, f"{min(values):.2g}", ha="center", va="top", fontsize=8)
            ax.text(i, 1.1, f"{max(values):.2g}", ha="center", va="bottom", fontsize=8)

        ax.set_ylim(-0.15, 1.15)
        fig.tight_layout()
        return fig, ax
