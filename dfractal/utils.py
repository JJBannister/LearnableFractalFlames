from typing import Tuple

import taichi as ti
import taichi.math as tm

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread, imsave
from skimage.transform import resize


def plot_image(
    image: np.array, colorbar=False, title=False, cmap="viridis", save_file=None
):
    plt.figure()
    plt.imshow(
        np.swapaxes(image, 0, 1), cmap=cmap, origin="lower", interpolation="nearest"
    )
    plt.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )

    if colorbar:
        plt.colorbar()
    if title:
        plt.title(title)

    plt.tight_layout()

    if save_file != None:
        plt.savefig(save_file, dpi=500)


def show_plots():
    plt.show()


def read_image(file_path: str, resolution: Tuple[int]):
    image = np.swapaxes(np.flip(imread(file_path), 0), 0, 1)
    image = resize(image, resolution)
    return image


def save_image(file_path: str, image: np.array):
    imsave(file_path, np.flip(np.swapaxes(image, 0, 1), 0))


@ti.func
def logistic(x):
    return tm.pow((1.0 + tm.exp(-x)), -1.0)
