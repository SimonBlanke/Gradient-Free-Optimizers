import os
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from gradient_free_optimizers.converter import Converter


def plot_search_paths(
    opt_name,
    optimizer,
    opt_para,
    n_iter_list,
    objective_function,
    objective_function_np,
    search_space,
    initialize,
    random_state,
):
    for n_iter in tqdm(n_iter_list):
        opt = optimizer(search_space, **opt_para)

        opt.search(
            objective_function,
            n_iter=n_iter,
            random_state=random_state,
            memory=False,
            verbosity=False,
            initialize=initialize,
        )

        conv = Converter(search_space)

        plt.figure(figsize=(10, 8))
        plt.set_cmap("jet_r")

        x_all, y_all = search_space["x"], search_space["y"]
        xi, yi = np.meshgrid(x_all, y_all)
        zi = objective_function_np(xi, yi)

        plt.imshow(
            zi,
            alpha=0.15,
            # vmin=z.min(),
            # vmax=z.max(),
            # origin="lower",
            extent=[x_all.min(), x_all.max(), y_all.min(), y_all.max()],
        )

        # print("\n Results \n", opt.results)

        for n, opt_ in enumerate(opt.optimizers):
            pos_list = np.array(opt_.pos_new_list)
            score_list = np.array(opt_.score_new_list)

            values_list = conv.positions2values(pos_list)
            values_list = np.array(values_list)

            plt.plot(
                values_list[:, 0],
                values_list[:, 1],
                linestyle="--",
                marker=",",
                color="black",
                alpha=0.33,
                label=n,
                linewidth=0.5,
            )
            plt.scatter(
                values_list[:, 0],
                values_list[:, 1],
                c=score_list,
                marker="H",
                s=15,
                vmin=-20000,
                vmax=0,
                label=n,
                edgecolors="black",
                linewidth=0.3,
            )

        plt.xlabel("x")
        plt.ylabel("y")

        nth_iteration = "\n\nnth Iteration: " + str(n_iter)

        plt.title(opt_name + nth_iteration)

        plt.xlim((-101, 101))
        plt.ylim((-101, 101))
        plt.colorbar()
        # plt.legend(loc="upper left", bbox_to_anchor=(-0.10, 1.2))

        plt.tight_layout()
        plt.savefig(
            "./_plots/"
            + str(opt.__class__.__name__)
            + "_"
            + "{0:0=3d}".format(n_iter)
            + ".jpg",
            dpi=300,
        )
        plt.close()


def create_search_path_gif(
    opt_name,
    optimizer,
    opt_para,
    n_iter,
    objective_function,
    objective_function_np,
    search_space,
):
    pass


########################################################################


from gradient_free_optimizers import HillClimbingOptimizer

optimizer_keys = ["HillClimbingOptimizer"]
n_iter_list = range(1, 51)


def get_path(optimizer_key, nth_iteration):
    return (
        "./_plots/"
        + str(optimizer_key)
        + "_"
        + "{0:0=2d}".format(nth_iteration)
        + ".jpg"
    )


def objective_function(pos_new):
    score = -(pos_new["x"] * pos_new["x"] + pos_new["y"] * pos_new["y"])
    return score


def objective_function_np(x1, x2):
    score = -(x1 * x1 + x2 * x2)
    return score


search_space = {"x": np.arange(-100, 101, 1), "y": np.arange(-100, 101, 1)}

n_iter_list = range(1, 3)
opt_para = {}
initialize = {"vertices": 2}
random_state = 0

para_dict = (
    "hill_climbing.gif",
    {
        "opt_name": "Hill climbing",
        "optimizer": HillClimbingOptimizer,
        "opt_para": opt_para,
        "n_iter_list": n_iter_list,
        "objective_function": objective_function,
        "objective_function_np": objective_function_np,
        "search_space": search_space,
        "initialize": initialize,
        "random_state": random_state,
    },
)


for _para_dict in tqdm([para_dict]):
    plot_search_paths(**_para_dict[1])

    _framerate = " -framerate 3 "
    _input = " -i ./_plots/HillClimbingOptimizer_%03d.jpg "
    _scale = " -vf scale=1200:-1 "
    _output = " ./_gifs/" + _para_dict[0]

    ffmpeg_command = "ffmpeg -y" + _framerate + _input + _scale + _output
    print("\n\n -----> ffmpeg_command \n", ffmpeg_command, "\n\n")
    print(_para_dict[0])

    os.system(ffmpeg_command)

    rm_files = glob.glob("./_plots/*.jpg")

    for f in rm_files:
        os.remove(f)

