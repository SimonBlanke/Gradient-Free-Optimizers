from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


optimizer_keys = ["HillClimbing", "StochasticHillClimbing"]


def create_convergence_plots(optimizer_keys, plot_name):
    plt.figure(figsize=(10, 8))

    mean_plts = []
    std_plts = []

    for idx, optimizer_key in enumerate(optimizer_keys):
        convergence_data = pd.read_csv(
            "./data/" + optimizer_key + "_convergence_data.csv"
        )

        x_range = range(len(convergence_data))
        scores_mean = convergence_data["scores_mean"]
        scores_std = convergence_data["scores_std"]

        (mean_plt,) = plt.plot(
            x_range,
            scores_mean,
            linestyle="--",
            marker=",",
            alpha=0.9,
            label=optimizer_key,
            linewidth=1,
        )
        std_plt = plt.fill_between(
            x_range,
            scores_mean - scores_std,
            scores_mean + scores_std,
            label=optimizer_key,
            alpha=0.3,
        )

        mean_plts.append(mean_plt)
        std_plts.append(std_plt)

    plt.tight_layout()
    leg1 = plt.legend(
        mean_plts, optimizer_keys, loc="lower center", title="average score"
    )
    plt.legend(
        std_plts, optimizer_keys, loc="lower right", title="standard deviation"
    )
    plt.gca().add_artist(leg1)

    plt.savefig(
        "./plots/" + plot_name + "_convergence.png", dpi=300,
    )
    plt.close()


create_convergence_plots(optimizer_keys, "local")
