from keras.models import Sequential
from keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    Activation,
)
from keras.datasets import cifar10
from keras.utils import to_categorical

from gradient_free_optimizers import BayesianOptimizer

import numpy as np

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


def cnn(para):
    nn = Sequential()
    nn.add(
        Conv2D(
            para["filter.0"],
            (3, 3),
            padding="same",
            input_shape=X_train.shape[1:],
        )
    )
    nn.add(Activation("relu"))
    nn.add(Conv2D(para["filter.0"], (3, 3)))
    nn.add(Activation("relu"))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Dropout(0.25))

    nn.add(Conv2D(para["filter.0"], (3, 3), padding="same"))
    nn.add(Activation("relu"))
    nn.add(Conv2D(para["filter.0"], (3, 3)))
    nn.add(Activation("relu"))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Dropout(0.25))

    nn.add(Flatten())
    nn.add(Dense(para["dense.0"]))
    nn.add(Activation("relu"))
    nn.add(Dropout(0.5))
    nn.add(Dense(10))
    nn.add(Activation("softmax"))

    nn.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    nn.fit(X_train, y_train, epochs=25, batch_size=128)

    _, score = nn.evaluate(x=X_test, y=y_test)

    return score


search_space = {
    "filter.0": np.array([16, 32, 64, 128]),
    "dense.0": np.arange(100, 1000, 100),
}


opt = BayesianOptimizer(search_space)
opt.search(cnn, n_iter=5)
