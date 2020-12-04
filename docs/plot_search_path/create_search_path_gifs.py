import imageio


optimizer_keys = ["HillClimbingOptimizer"]
n_iter_list = range(1, 51)


def get_path(optimizer_key, nth_iteration):
    return (
        "./plots/"
        + str(optimizer_key)
        + "_"
        + "{0:0=2d}".format(nth_iteration)
        + ".jpg"
    )


filepaths = [
    get_path(opt, nth_iter)
    for opt in optimizer_keys
    for nth_iter in n_iter_list
]


images = []
for filename in filepaths:
    images.append(imageio.imread(filename))

print("Save gif...")
imageio.mimsave("movie.gif", images, duration=0.3)

