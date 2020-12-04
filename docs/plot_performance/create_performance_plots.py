from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_performance_plots(study_name):
    results = pd.read_csv("./data/" + study_name + ".csv", index_col=0)

    total_time = results.loc["total_time_mean"].values
    eval_time = results.loc["eval_time_mean"].values
    iter_time = results.loc["iter_time_mean"].values

    ind = np.arange(total_time.shape[0])  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    plt.figure(figsize=(15, 5))

    p2 = plt.bar(ind, iter_time, width, bottom=eval_time)
    p1 = plt.bar(ind, eval_time, width)

    plt.ylabel("Time [s]")
    # plt.title(title)
    plt.xticks(ind, results.columns, rotation=75)
    # plt.yticks()
    plt.legend((p1[0], p2[0]), ("Eval time", "Opt time"))

    plt.tight_layout()
    plt.savefig("./plots/performance.png", dpi=300)


create_performance_plots("simple function")
