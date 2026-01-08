def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

import gc
import glob
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

plt.rcParams["figure.facecolor"] = "w"

mpl.use("agg")

from gradient_free_optimizers.optimizers.core_optimizer.converter import Converter


def plot_search_paths(
    path,
    optimizer,
    opt_para,
    n_iter_max,
    objective_function,
    search_space,
    initialize,
    random_state,
    title,
):
    if opt_para == {}:
        show_opt_para = False
    else:
        show_opt_para = True

    opt = optimizer(
        search_space, initialize=initialize, random_state=random_state, **opt_para
    )
    opt.search(
        objective_function,
        n_iter=n_iter_max,
        # memory=False,
        verbosity=False,
    )

    conv = Converter(search_space)
    for n_iter in tqdm(range(1, n_iter_max + 1)):

        def objective_function_np(args):
            params = {}
            for i, para_name in enumerate(search_space):
                params[para_name] = args[i]

            return objective_function(params)

        plt.figure(figsize=(7, 7))
        plt.set_cmap("jet_r")
        # jet_r

        x_all, y_all = search_space["x0"], search_space["x1"]
        xi, yi = np.meshgrid(x_all, y_all)
        zi = objective_function_np((xi, yi))

        zi = np.rot90(zi, k=1)

        plt.imshow(
            zi,
            alpha=0.15,
            # interpolation="antialiased",
            # vmin=z.min(),
            # vmax=z.max(),
            # origin="lower",
            extent=[x_all.min(), x_all.max(), y_all.min(), y_all.max()],
        )

        for n, opt_ in enumerate(opt.optimizers):
            n_optimizers = len(opt.optimizers)
            n_iter_tmp = int(n_iter / n_optimizers)
            n_iter_mod = n_iter % n_optimizers

            if n_iter_mod > n:
                n_iter_tmp += 1
            if n_iter_tmp == 0:
                continue

            pos_list = np.array(opt_.pos_new_list)
            score_list = np.array(opt_.score_new_list)

            # print("\n pos_list \n", pos_list, "\n")
            # print("\n score_list \n", score_list, "\n")

            if len(pos_list) == 0:
                continue

            values_list = conv.positions2values(pos_list)
            values_list = np.array(values_list)

            plt.plot(
                values_list[:n_iter_tmp, 0],
                values_list[:n_iter_tmp, 1],
                linestyle="--",
                marker=",",
                color="black",
                alpha=0.33,
                label=n,
                linewidth=0.5,
            )
            plt.scatter(
                values_list[:n_iter_tmp, 0],
                values_list[:n_iter_tmp, 1],
                c=score_list[:n_iter_tmp],
                marker="H",
                s=15,
                vmin=np.amin(score_list[:n_iter_tmp]),
                vmax=np.amax(score_list[:n_iter_tmp]),
                label=n,
                edgecolors="black",
                linewidth=0.3,
            )

        plt.xlabel("x")
        plt.ylabel("y")

        nth_iteration = "\n\nnth Iteration: " + str(n_iter)
        opt_para_name = ""
        opt_para_value = "\n\n"

        if show_opt_para:
            opt_para_name += "\n Parameter:"
            for para_name, para_value in opt_para.items():
                opt_para_name += "\n " + "     " + para_name + ": "
                opt_para_value += "\n " + str(para_value) + "                "

        if title is True:
            title_name = opt.name + "\n" + opt_para_name
            plt.title(title_name, loc="left")
            plt.title(opt_para_value, loc="center")
        elif isinstance(title, str):
            plt.title(title, loc="left")

        plt.title(nth_iteration, loc="right", fontsize=8)

        # plt.xlim((-101, 201))
        # plt.ylim((-101, 201))
        clb = plt.colorbar()
        clb.set_label("score", labelpad=-50, y=1.03, rotation=0)

        # plt.legend(loc="upper left", bbox_to_anchor=(-0.10, 1.2))

        # plt.axis("off")

        if show_opt_para:
            plt.subplots_adjust(top=0.75)
        plt.tight_layout()

        # plt.margins(0, 0)
        plt.savefig(
            path + "/_plots/" + opt._name_ + "_" + f"{n_iter:0=3d}" + ".jpg",
            dpi=150,
            pad_inches=0,
            # bbox_inches="tight",
        )

        plt.ioff()
        # Clear the current axes.
        plt.cla()
        # Clear the current figure.
        plt.clf()
        # Closes all the figure windows.
        plt.close("all")

        gc.collect()


def search_path_gif(
    path,
    optimizer,
    opt_para,
    name,
    n_iter,
    objective_function,
    search_space,
    initialize,
    random_state=0,
    title=True,
):
    path = os.path.join(os.getcwd(), path)

    print("\n\nname", name)
    plots_dir = path + "/_plots/"
    print("plots_dir", plots_dir)
    os.makedirs(plots_dir, exist_ok=True)

    plot_search_paths(
        path=path,
        optimizer=optimizer,
        opt_para=opt_para,
        n_iter_max=n_iter,
        objective_function=objective_function,
        search_space=search_space,
        initialize=initialize,
        random_state=random_state,
        title=title,
    )

    ### ffmpeg
    framerate = str(n_iter / 10)
    # framerate = str(10)
    _framerate = " -framerate " + framerate + " "

    _opt_ = optimizer(search_space)
    _input = " -i " + path + "/_plots/" + str(_opt_._name_) + "_" + "%03d.jpg "
    _scale = " -vf scale=1200:-1:flags=lanczos "
    _output = os.path.join(path, name)

    ffmpeg_command = (
        "ffmpeg -hide_banner -loglevel error -y"
        + _framerate
        + _input
        + _scale
        + _output
    )
    print("\n -----> ffmpeg_command \n", ffmpeg_command, "\n")
    print("create " + name)

    os.system(ffmpeg_command)

    ### remove _plots
    rm_files = glob.glob(path + "/_plots/*.jpg")
    for f in rm_files:
        os.remove(f)
    os.rmdir(plots_dir)
